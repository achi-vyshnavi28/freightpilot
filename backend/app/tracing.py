"""Per-run tracing: every node, LLM call, tool call and guardrail decision becomes a span.

Spans are stored in the database so the UI can replay exactly what an agent did,
what it cost and how long it took.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from .db import TraceSpan, session_scope

log = logging.getLogger("freightpilot.trace")

# USD per million tokens (input, output). Unknown models are costed at zero, not guessed.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICING.get(model or "", (0.0, 0.0))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def _summarise(value, limit: int = 600) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class Span:
    agent: str
    kind: str
    name: str
    input: object = None
    output: object = None
    status: str = "ok"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class Tracer:
    document_id: int | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    persist: bool = True
    spans: list[Span] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(cost_usd(s.model, s.input_tokens, s.output_tokens) for s in self.spans)

    @property
    def total_tokens(self) -> tuple[int, int]:
        return sum(s.input_tokens for s in self.spans), sum(s.output_tokens for s in self.spans)

    def record(self, span: Span) -> Span:
        self.spans.append(span)
        log.info("%s/%s %s %s %.0fms", span.agent, span.kind, span.name, span.status, span.latency_ms)
        if self.persist:
            with session_scope() as s:
                s.add(
                    TraceSpan(
                        run_id=self.run_id,
                        document_id=self.document_id,
                        agent=span.agent,
                        kind=span.kind,
                        name=span.name,
                        status=span.status,
                        input_summary=_summarise(span.input),
                        output_summary=_summarise(span.output),
                        model=span.model,
                        input_tokens=span.input_tokens,
                        output_tokens=span.output_tokens,
                        cost_usd=cost_usd(span.model, span.input_tokens, span.output_tokens),
                        latency_ms=span.latency_ms,
                    )
                )
        return span

    @contextmanager
    def span(self, agent: str, kind: str, name: str, input=None):
        span = Span(agent=agent, kind=kind, name=name, input=input)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.output = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.latency_ms = (time.perf_counter() - start) * 1000
            self.record(span)
