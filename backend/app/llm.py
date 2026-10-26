"""Claude access layer: structured outputs and a traced, permission-checked tool loop.

When no Anthropic credentials are configured the app runs fully offline: every
agent has a deterministic policy that uses the same tools, so tests and evals
run in CI for free. With credentials, Claude drives the same tools.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from .config import settings
from .guardrails import ToolPermissionError, check_tool_permission
from .tracing import Span, Tracer

# Server-side refusal fallback is available for these models on the Claude API.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class LLMUnavailable(RuntimeError):
    pass


class LLMRefusal(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., object]

    def definition(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCall:
    name: str
    input: dict
    output: object
    is_error: bool = False


def _has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


class ClaudeLLM:
    def __init__(self) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def _create(self, *, model: str, **kwargs):
        output_config = {"effort": settings.effort, **kwargs.pop("output_config", {})}
        params = {"model": model, "max_tokens": 16000, "output_config": output_config, **kwargs}
        if model in _FALLBACK_MODELS:
            params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            response = self.client.beta.messages.create(**params)
        else:
            response = self.client.messages.create(**params)
        if response.stop_reason == "refusal":
            raise LLMRefusal(getattr(response.stop_details, "category", None) or "refused")
        return response

    def structured(
        self,
        *,
        agent: str,
        name: str,
        system: str,
        content: list[dict] | str,
        schema: dict,
        tracer: Tracer,
        model: str | None = None,
    ) -> dict:
        model = model or settings.model
        start = time.perf_counter()
        span = Span(
            agent=agent, kind="llm", name=name, model=model, input=content if isinstance(content, str) else "[multimodal]"
        )
        try:
            response = self._create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            span.input_tokens = response.usage.input_tokens
            span.output_tokens = response.usage.output_tokens
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            span.output = data
            return data
        except Exception as exc:
            span.status = "error"
            span.output = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.latency_ms = (time.perf_counter() - start) * 1000
            tracer.record(span)

    def run_tools(
        self,
        *,
        agent: str,
        system: str,
        prompt: str,
        tools: list[Tool],
        tracer: Tracer,
        model: str | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Manual agent loop. Owned here (not the SDK tool runner) so every call is
        permission-checked against the agent's allowlist and traced as a span."""
        model = model or settings.model
        by_name = {t.name: t for t in tools}
        for t in tools:
            check_tool_permission(agent, t.name)  # a mis-wired agent fails loudly at start
        messages: list[dict] = [{"role": "user", "content": prompt}]
        calls: list[ToolCall] = []

        for turn in range(settings.max_agent_turns):
            start = time.perf_counter()
            span = Span(agent=agent, kind="llm", name=f"turn_{turn}", model=model)
            try:
                response = self._create(model=model, system=system, messages=messages, tools=[t.definition() for t in tools])
            except Exception as exc:
                span.status, span.output = "error", f"{type(exc).__name__}: {exc}"
                span.latency_ms = (time.perf_counter() - start) * 1000
                tracer.record(span)
                raise
            span.latency_ms = (time.perf_counter() - start) * 1000
            span.input_tokens = response.usage.input_tokens
            span.output_tokens = response.usage.output_tokens
            span.output = {"stop_reason": response.stop_reason}
            tracer.record(span)

            # Append the full content unchanged: thinking blocks must round-trip as-is.
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = "\n".join(b.text for b in response.content if b.type == "text")
                return text, calls

            results = []
            for block in tool_uses:
                results.append(self._execute(agent, block, by_name, tracer, calls))
            # All results for one turn go back in a single user message.
            messages.append({"role": "user", "content": results})

        return "stopped: max agent turns reached", calls

    def _execute(self, agent: str, block, by_name: dict[str, Tool], tracer: Tracer, calls: list[ToolCall]) -> dict:
        args = block.input if isinstance(block.input, dict) else {}
        with tracer.span(agent, "tool", block.name, input=args) as span:
            try:
                check_tool_permission(agent, block.name)
                tool = by_name.get(block.name)
                if tool is None:
                    raise KeyError(f"unknown tool {block.name}")
                output = tool.fn(**args)
                is_error = False
            except ToolPermissionError as exc:
                output, is_error, span.status = str(exc), True, "denied"
            except Exception as exc:  # tool failures go back to the model, not up the stack
                output, is_error, span.status = f"{type(exc).__name__}: {exc}", True, "error"
            span.output = output
        calls.append(ToolCall(block.name, args, output, is_error))
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output if isinstance(output, str) else json.dumps(output, default=str),
            "is_error": is_error,
        }


_llm: ClaudeLLM | None = None


def get_llm() -> ClaudeLLM | None:
    """Return a Claude client, or None when running offline."""
    global _llm
    mode = settings.llm_mode
    if mode == "offline" or (mode == "auto" and not _has_credentials()):
        return None
    if _llm is None:
        try:
            _llm = ClaudeLLM()
        except Exception as exc:
            if mode == "claude":
                raise LLMUnavailable(str(exc)) from exc
            return None
    return _llm
