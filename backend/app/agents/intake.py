"""Intake agent: screen the document, then decide what kind of paper it is."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings
from ..guardrails import UNTRUSTED_DATA_RULE, redact_pii, scan_for_injection, wrap_untrusted
from ..llm import ClaudeLLM, LLMRefusal
from ..schemas import DOC_TYPES
from ..tools.extractors import classify_by_keywords
from ..tracing import Tracer
from .common import call_tool, decision

AGENT = "intake"

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": list(DOC_TYPES)},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["doc_type", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass
class IntakeResult:
    doc_type: str
    confidence: float
    llm_text: str  # PII-redacted text, the only version an external model ever sees
    flags: dict = field(default_factory=dict)


def run_intake(text: str, tracer: Tracer, llm: ClaudeLLM | None) -> IntakeResult:
    with tracer.span(AGENT, "guardrail", "screen_document") as span:
        injection = scan_for_injection(text)
        redacted, pii = redact_pii(text) if settings.redact_pii else (text, {})
        flags = {"prompt_injection": injection.matches} if injection.suspicious else {}
        if pii:
            flags["pii_redacted"] = pii
        span.output = flags or "clean"
        span.status = "flagged" if injection.suspicious else "ok"

    doc_type, conf, scores = call_tool(tracer, AGENT, "classify_document", lambda text: classify_by_keywords(text), text=text)

    # Keywords settle most documents for free; ask the cheap model only when they don't.
    if (doc_type == "unknown" or conf < 0.65) and llm is not None and text.strip():
        try:
            out = llm.structured(
                agent=AGENT,
                name="classify_llm",
                system=f"Classify Indian logistics documents. {UNTRUSTED_DATA_RULE}",
                content=wrap_untrusted(redacted[:6000]),
                schema=CLASSIFY_SCHEMA,
                tracer=tracer,
                model=settings.fast_model,
            )
            doc_type, conf = out["doc_type"], float(out["confidence"])
        except LLMRefusal as exc:
            decision(tracer, AGENT, "llm_refused", str(exc))

    decision(tracer, AGENT, "classified", {"doc_type": doc_type, "confidence": conf, "keyword_scores": scores})
    return IntakeResult(doc_type, conf, redacted, flags)
