"""Shared plumbing for agents' deterministic (code-driven) tool calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ..guardrails import check_tool_permission
from ..tracing import Tracer

T = TypeVar("T")


def call_tool(tracer: Tracer, agent: str, name: str, fn: Callable[..., T], /, **kwargs) -> T:
    """Same permission check and trace span as a model-initiated tool call."""
    check_tool_permission(agent, name)
    with tracer.span(agent, "tool", name, input=kwargs) as span:
        result = fn(**kwargs)
        span.output = result
        return result


def decision(tracer: Tracer, agent: str, name: str, detail: object) -> None:
    """Record why an agent took a branch, so the trace explains itself."""
    with tracer.span(agent, "decision", name) as span:
        span.output = detail
