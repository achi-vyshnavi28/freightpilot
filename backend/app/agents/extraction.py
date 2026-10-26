"""Extraction agent.

Escalation ladder (code-driven, because the order never changes):
    rules -> fine-tuned ML -> LLM, stopping as soon as the validators pass.

Repair loop (model-driven, because the fix depends on the document): when the
best attempt still fails validation, Claude gets the issues and may propose
corrections, re-checking them with `validate_fields`. A correction is accepted
only if its value literally appears in the source text, so the agent can fix
OCR misreads but can never invent numbers to make a checksum pass.
"""

from __future__ import annotations

import base64
import json
import re

from ..config import settings
from ..guardrails import UNTRUSTED_DATA_RULE, wrap_untrusted
from ..llm import ClaudeLLM, LLMRefusal, Tool
from ..schemas import FIELD_MODELS, ExtractionResult
from ..tools.extractors import EXTRACTION_SYSTEM, llm_schema, ml_available, ml_extract, rule_extract
from ..tools.ocr import TextResult
from ..tracing import Tracer
from ..validators import validate
from .common import call_tool, decision

AGENT = "extraction"

# Fields that must be present and confident for a document to skip human review.
# Deliberately broad: a missing field downstream (no vehicle number -> no cross-check)
# is a silent failure, and silent failures cost more than a human glance.
REQUIRED: dict[str, tuple[str, ...]] = {
    "gst_invoice": (
        "invoice_no",
        "invoice_date",
        "seller_gstin",
        "buyer_gstin",
        "eway_bill_no",
        "vehicle_no",
        "origin",
        "destination",
        "taxable_value",
        "total_amount",
    ),
    "eway_bill": (
        "eway_bill_no",
        "generated_date",
        "valid_upto",
        "from_gstin",
        "to_gstin",
        "origin",
        "destination",
        "distance_km",
        "vehicle_no",
        "invoice_no",
        "goods_value",
    ),
    "lorry_receipt": (
        "lr_number",
        "lr_date",
        "consignor",
        "consignee",
        "origin",
        "destination",
        "vehicle_no",
        "weight_kg",
        "freight_amount",
        "invoice_no",
    ),
}


def _score(doc_type: str, fields: dict, confidence: dict) -> tuple[list[dict], float]:
    issues = [i.to_dict() for i in validate(doc_type, fields)]
    required = REQUIRED.get(doc_type, ())
    min_conf = min((confidence.get(f, 0.0) for f in required), default=1.0)
    return issues, min_conf


def _accepted(issues: list[dict], min_conf: float) -> bool:
    return not any(i["severity"] == "error" for i in issues) and min_conf >= settings.review_threshold


def _attempt(method: str, fields: dict, confidence: dict, doc_type: str) -> dict:
    issues, min_conf = _score(doc_type, fields, confidence)
    return {"method": method, "fields": fields, "confidence": confidence, "issues": issues, "min_confidence": min_conf}


def _llm_extract(llm: ClaudeLLM, doc_type: str, text_for_llm: str, tr: TextResult, raw: bytes, tracer: Tracer) -> dict:
    if tr.is_image or not text_for_llm.strip():
        data = base64.standard_b64encode(raw).decode("utf-8")
        block_type = "document" if tr.media_type == "application/pdf" else "image"
        content = [
            {"type": block_type, "source": {"type": "base64", "media_type": tr.media_type, "data": data}},
            {"type": "text", "text": f"Extract the {doc_type} fields. {UNTRUSTED_DATA_RULE}"},
        ]
    else:
        content = f"Document type: {doc_type}\n{wrap_untrusted(text_for_llm)}"
    return llm.structured(
        agent=AGENT,
        name="run_llm_extractor",
        system=f"{EXTRACTION_SYSTEM} {UNTRUSTED_DATA_RULE}",
        content=content,
        schema=llm_schema(doc_type),
        tracer=tracer,
    )


def _normalise_for_grounding(value: object) -> str:
    text = str(value).upper()
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    return re.sub(r"[^A-Z0-9]", "", text)


def grounded(value: object, source_text: str) -> bool:
    """Does the proposed value literally occur in the document (ignoring spacing, commas, case)?"""
    if value in (None, ""):
        return True
    needle = _normalise_for_grounding(value)
    hay = re.sub(r"[^A-Z0-9]", "", source_text.upper())
    if isinstance(value, float):
        # Amounts print as 42,000.00 or 42000 - both normalise to a prefix of the digits.
        return needle in hay or _normalise_for_grounding(f"{value:.2f}") in hay
    return needle in hay


def _repair(llm: ClaudeLLM, doc_type: str, best: dict, source_text: str, llm_text: str, tracer: Tracer) -> dict | None:
    proposals: dict[str, object] = {}

    def validate_fields(fields_json: str) -> dict:
        candidate = {**best["fields"], **json.loads(fields_json)}
        return {"issues": [i.to_dict() for i in validate(doc_type, candidate)]}

    def request_human_review(corrections_json: str, reason: str) -> str:
        proposals.update(json.loads(corrections_json or "{}"))
        proposals["__reason__"] = reason
        return "recorded"

    tools = [
        Tool(
            "validate_fields",
            "Re-run the deterministic validators with corrected fields merged in. Pass a JSON object of only the fields you changed.",
            {
                "type": "object",
                "properties": {"fields_json": {"type": "string"}},
                "required": ["fields_json"],
                "additionalProperties": False,
            },
            validate_fields,
        ),
        Tool(
            "request_human_review",
            "Finish. Submit your corrections (JSON object, may be empty) and a one-line reason a human can act on, e.g. "
            "'OCR read S as 5 in seller GSTIN' or 'the invoice itself has an arithmetic error'.",
            {
                "type": "object",
                "properties": {"corrections_json": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["corrections_json", "reason"],
                "additionalProperties": False,
            },
            request_human_review,
        ),
    ]
    prompt = (
        f"An extraction of a {doc_type} failed validation.\n"
        f"Extracted fields: {json.dumps(best['fields'], default=str)}\n"
        f"Validation issues: {json.dumps(best['issues'])}\n\n"
        f"{wrap_untrusted(llm_text[:8000])}\n\n"
        "Decide whether each issue is an extraction mistake (fix it, using only values visible in the document) "
        "or a real error in the paper itself (leave it). Check your corrections with validate_fields, then call "
        "request_human_review exactly once."
    )
    try:
        llm.run_tools(
            agent=AGENT,
            system=f"You audit document extractions. {UNTRUSTED_DATA_RULE}",
            prompt=prompt,
            tools=tools,
            tracer=tracer,
        )
    except LLMRefusal as exc:
        decision(tracer, AGENT, "repair_refused", str(exc))
        return None
    if not proposals:
        return None

    reason = proposals.pop("__reason__", "")
    accepted, rejected = {}, {}
    for name, value in proposals.items():
        (accepted if grounded(value, source_text) else rejected)[name] = value
    decision(
        tracer, AGENT, "repair_grounding_check", {"accepted": accepted, "rejected_not_in_document": rejected, "reason": reason}
    )
    fields = {**best["fields"], **accepted}
    confidence = {**best["confidence"], **{k: 0.8 for k in accepted}}
    attempt = _attempt("llm_repair", fields, confidence, doc_type)
    attempt["reason"] = reason
    return attempt


def run_extraction(
    doc_type: str,
    text_result: TextResult,
    raw: bytes,
    llm_text: str,
    tracer: Tracer,
    llm: ClaudeLLM | None,
) -> ExtractionResult:
    if doc_type not in FIELD_MODELS:
        return ExtractionResult(
            doc_type=doc_type,
            fields={},
            method="none",
            needs_review=True,
            issues=[{"field": "doc_type", "code": "unsupported", "message": f"no extractor for {doc_type}", "severity": "error"}],
        )
    attempts: list[dict] = []
    text = text_result.text

    if text.strip():
        rx = call_tool(
            tracer, AGENT, "run_rule_extractor", lambda doc_type, text: rule_extract(doc_type, text), doc_type=doc_type, text=text
        )
        attempts.append(_attempt("rules", rx.fields, rx.confidence, doc_type))

    if (
        not (attempts and _accepted(attempts[-1]["issues"], attempts[-1]["min_confidence"]))
        and text_result.is_image
        and ml_available()
    ):
        fields = call_tool(tracer, AGENT, "run_ml_extractor", ml_extract, doc_type=doc_type, image_bytes=raw)
        attempts.append(_attempt("ml", fields, {k: 0.85 for k in fields}, doc_type))

    if not (attempts and _accepted(attempts[-1]["issues"], attempts[-1]["min_confidence"])) and llm is not None:
        try:
            fields = _llm_extract(llm, doc_type, llm_text, text_result, raw, tracer)
            attempts.append(_attempt("llm", fields, {k: 0.9 for k, v in fields.items() if v not in (None, [], "")}, doc_type))
        except LLMRefusal as exc:
            decision(tracer, AGENT, "llm_refused", str(exc))

    if not attempts:
        return ExtractionResult(
            doc_type=doc_type,
            fields={},
            method="none",
            needs_review=True,
            issues=[
                {"field": "*", "code": "no_text", "message": "no text layer and no OCR/LLM tier available", "severity": "error"}
            ],
        )

    def rank(a: dict) -> tuple:
        return (sum(i["severity"] == "error" for i in a["issues"]), -a["min_confidence"])

    best = min(attempts, key=rank)
    decision(
        tracer,
        AGENT,
        "best_attempt",
        {"method": best["method"], "errors": rank(best)[0], "min_confidence": best["min_confidence"]},
    )

    if not _accepted(best["issues"], best["min_confidence"]) and llm is not None and best["issues"]:
        repaired = _repair(llm, doc_type, best, text, llm_text, tracer)
        if repaired:
            attempts.append(repaired)
            if rank(repaired) < rank(best):
                best = repaired

    needs_review = not _accepted(best["issues"], best["min_confidence"])
    return ExtractionResult(
        doc_type=doc_type,
        fields=best["fields"],
        confidence=best["confidence"],
        method=best["method"],
        issues=best["issues"],
        attempts=[{k: v for k, v in a.items() if k != "fields"} for a in attempts],
        needs_review=needs_review,
    )
