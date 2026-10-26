"""Reconciliation agent: cross-check one document against its siblings and the contract.

The checks are deterministic (a vehicle number either matches or it does not).
Retrieval finds the contracted lane rate, and the model only writes the
explanation, citing the chunk ids it was given.
"""

from __future__ import annotations

from sqlalchemy import or_, select

from ..config import settings
from ..db import Document, session_scope
from ..llm import ClaudeLLM, LLMRefusal
from ..tools.retrieval import canonical_city, get_index, lookup_lane_rate
from ..tracing import Tracer
from .common import call_tool, decision

AGENT = "reconciliation"

LINK_KEYS = ("invoice_no", "eway_bill_no", "vehicle_no")


def get_related_documents(client_id: str, document_id: int | None, fields: dict) -> list[dict]:
    conds = []
    for key in LINK_KEYS:
        if fields.get(key):
            conds.append(Document.fields[key].as_string() == str(fields[key]))
    if not conds:
        return []
    with session_scope() as s:
        rows = s.scalars(
            select(Document).where(
                Document.client_id == client_id, Document.id != (document_id or -1), Document.fields.is_not(None), or_(*conds)
            )
        ).all()
        return [{"id": d.id, "doc_type": d.doc_type, "fields": d.fields} for d in rows]


def _same(a, b) -> bool | None:
    if a in (None, "") or b in (None, ""):
        return None
    if isinstance(a, str) and isinstance(b, str):
        return a.replace(" ", "").upper() == b.replace(" ", "").upper()
    return a == b


def compare_documents(doc_type: str, fields: dict, related: list[dict]) -> list[dict]:
    checks = []
    for other in related:
        o, label = other["fields"], f"{other['doc_type']}#{other['id']}"
        for key in ("vehicle_no", "invoice_no", "eway_bill_no"):
            result = _same(fields.get(key), o.get(key))
            if result is not None:
                checks.append({"check": f"{key} vs {label}", "ok": result, "ours": fields.get(key), "theirs": o.get(key)})
        for key in ("origin", "destination"):
            if fields.get(key) and o.get(key):
                ok = canonical_city(fields[key]) == canonical_city(o[key])
                checks.append({"check": f"{key} vs {label}", "ok": ok, "ours": fields[key], "theirs": o[key]})
        # E-way bill goods value must cover the invoice value.
        pair = {doc_type: fields, other["doc_type"]: o}
        inv, ewb = pair.get("gst_invoice"), pair.get("eway_bill")
        if inv and ewb and inv.get("total_amount") and ewb.get("goods_value"):
            ok = abs(inv["total_amount"] - ewb["goods_value"]) <= 1.0
            checks.append(
                {
                    "check": f"e-way bill goods value vs invoice total ({label})",
                    "ok": ok,
                    "ours": fields.get("total_amount") or fields.get("goods_value"),
                    "theirs": o.get("total_amount") or o.get("goods_value"),
                }
            )
        if fields.get("weight_kg") and o.get("weight_kg"):
            ok = abs(fields["weight_kg"] - o["weight_kg"]) / max(o["weight_kg"], 1) <= 0.02
            checks.append({"check": f"weight vs {label}", "ok": ok, "ours": fields["weight_kg"], "theirs": o["weight_kg"]})
    return checks


def _billed_freight(doc_type: str, fields: dict) -> float | None:
    if doc_type == "lorry_receipt":
        return fields.get("freight_amount")
    if doc_type == "gst_invoice":
        freight = [i["amount"] for i in fields.get("line_items") or [] if "freight" in i.get("description", "").lower()]
        return sum(freight) if freight else None
    return None


EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "2-3 sentences for an operations manager"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "cited_chunk_ids"],
    "additionalProperties": False,
}


def run_reconciliation(
    client_id: str, document_id: int | None, doc_type: str, fields: dict, tracer: Tracer, llm: ClaudeLLM | None
) -> dict:
    related = call_tool(
        tracer, AGENT, "get_related_documents", get_related_documents, client_id=client_id, document_id=document_id, fields=fields
    )
    checks = call_tool(tracer, AGENT, "compare_documents", compare_documents, doc_type=doc_type, fields=fields, related=related)

    rate_check = None
    billed = _billed_freight(doc_type, fields)
    if billed and fields.get("origin") and fields.get("destination"):
        lane = call_tool(
            tracer,
            AGENT,
            "search_rate_cards",
            lookup_lane_rate,
            client_id=client_id,
            origin=fields["origin"],
            destination=fields["destination"],
            weight_kg=fields.get("weight_kg"),
        )
        if lane:
            overcharge = (billed - lane["rate"]) / lane["rate"]
            rate_check = {
                "billed": billed,
                "contract_rate": lane["rate"],
                "vehicle_class_t": lane["capacity_t"],
                "difference_pct": round(overcharge * 100, 2),
                "ok": overcharge <= settings.rate_tolerance,
                "citation": lane["citation"],
            }
        else:
            rate_check = {"billed": billed, "ok": None, "note": "no contracted rate found for this lane"}

    mismatches = [c for c in checks if not c["ok"]]
    if rate_check and rate_check.get("ok") is False:
        mismatches.append(
            {"check": "freight vs contract rate", "ok": False, "ours": billed, "theirs": rate_check["contract_rate"]}
        )

    citation_ids = [rate_check["citation"]["id"]] if rate_check and rate_check.get("citation") else []
    summary = _template_summary(checks, rate_check, related)
    if llm is not None and (mismatches or rate_check):
        context = [rate_check["citation"]] if rate_check and rate_check.get("citation") else []
        context += [
            h.to_dict()
            for h in get_index().search(
                f"{fields.get('origin', '')} {fields.get('destination', '')} detention penalty dispute",
                k=2,
                source_prefix=client_id,
            )
        ]
        try:
            out = llm.structured(
                agent=AGENT,
                name="explain_findings",
                system="You explain freight billing reconciliation results. Cite only the chunk ids provided. Do not restate numbers that are not in the input.",
                content=f"Checks: {checks}\nRate check: {rate_check}\nContract excerpts: {context}",
                schema=EXPLAIN_SCHEMA,
                tracer=tracer,
            )
            known = {c["id"] for c in context}
            summary = out["summary"]
            citation_ids = [c for c in out["cited_chunk_ids"] if c in known] or citation_ids
        except LLMRefusal as exc:
            decision(tracer, AGENT, "llm_refused", str(exc))

    result = {
        "checks": checks,
        "rate_check": rate_check,
        "related_documents": [r["id"] for r in related],
        "mismatch_count": len(mismatches),
        "summary": summary,
        "citations": citation_ids,
    }
    decision(tracer, AGENT, "reconciled", {"mismatches": len(mismatches), "related": len(related)})
    return result


def _template_summary(checks: list[dict], rate_check: dict | None, related: list) -> str:
    parts = []
    failed = [c["check"] for c in checks if not c["ok"]]
    if related:
        parts.append(
            f"Compared with {len(related)} related document(s): "
            + (f"{len(failed)} mismatch(es) - " + "; ".join(failed) if failed else "all cross-checks passed")
            + "."
        )
    else:
        parts.append("No related documents uploaded yet for cross-checking.")
    if rate_check and rate_check.get("ok") is False:
        parts.append(
            f"Freight billed Rs {rate_check['billed']:,.0f} is {rate_check['difference_pct']}% above the contracted Rs {rate_check['contract_rate']:,.0f} ({rate_check['citation']['id']})."
        )
    elif rate_check and rate_check.get("ok"):
        parts.append(f"Freight is within the contracted rate ({rate_check['citation']['id']}).")
    elif rate_check:
        parts.append(rate_check["note"].capitalize() + ".")
    return " ".join(parts)
