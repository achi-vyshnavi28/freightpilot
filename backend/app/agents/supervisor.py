r"""Document workflow as a LangGraph state machine.

    intake -> extraction -> reconciliation -> comms -> finalize
          \-> finalize (unknown / unsupported doc)   \-> finalize (nothing to send)

Graph state is checkpointed per run (thread_id = "doc-<id>-<run_id>"), so a run can
be inspected or resumed step by step, and a reprocess never inherits stale state. Non-serialisable collaborators (tracer,
LLM client, raw bytes) travel in the run config, not in the state.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..db import Document, session_scope
from ..llm import ClaudeLLM
from ..tools.ocr import TextResult
from ..tracing import Tracer
from .comms import run_comms
from .extraction import run_extraction
from .intake import run_intake
from .reconciliation import run_reconciliation

RECONCILABLE = {"gst_invoice", "eway_bill", "lorry_receipt"}


class DocState(TypedDict, total=False):
    document_id: int
    client_id: str
    filename: str
    doc_type: str
    doc_confidence: float
    flags: dict
    llm_text: str
    extraction: dict
    reconciliation: dict
    approval_ids: list[int]
    status: str


def _ctx(config: RunnableConfig) -> tuple[Tracer, ClaudeLLM | None, TextResult, bytes]:
    c = config["configurable"]
    return c["tracer"], c["llm"], c["text_result"], c["raw"]


def _set_status(document_id: int, status: str, **fields: Any) -> None:
    with session_scope() as s:
        doc = s.get(Document, document_id)
        doc.status = status
        for k, v in fields.items():
            setattr(doc, k, v)


def intake_node(state: DocState, config: RunnableConfig) -> DocState:
    tracer, llm, tr, _ = _ctx(config)
    _set_status(state["document_id"], "classifying", raw_text=tr.text)
    with tracer.span("supervisor", "node", "intake"):
        result = run_intake(tr.text, tracer, llm)
    return {"doc_type": result.doc_type, "doc_confidence": result.confidence, "flags": result.flags, "llm_text": result.llm_text}


def extraction_node(state: DocState, config: RunnableConfig) -> DocState:
    tracer, llm, tr, raw = _ctx(config)
    _set_status(state["document_id"], "extracting", doc_type=state["doc_type"], guardrail_flags=state.get("flags"))
    with tracer.span("supervisor", "node", "extraction"):
        result = run_extraction(state["doc_type"], tr, raw, state.get("llm_text", tr.text), tracer, llm)
    return {"extraction": result.model_dump()}


def reconciliation_node(state: DocState, config: RunnableConfig) -> DocState:
    tracer, llm, _, _ = _ctx(config)
    _set_status(state["document_id"], "reconciling")
    with tracer.span("supervisor", "node", "reconciliation"):
        result = run_reconciliation(
            state["client_id"], state["document_id"], state["doc_type"], state["extraction"]["fields"], tracer, llm
        )
    return {"reconciliation": result}


def comms_node(state: DocState, config: RunnableConfig) -> DocState:
    tracer, llm, _, _ = _ctx(config)
    _set_status(state["document_id"], "drafting")
    ext, rec = state.get("extraction") or {}, state.get("reconciliation") or {}
    fields = ext.get("fields") or {}
    reference = next(
        (f"{k} {fields[k]}" for k in ("invoice_no", "lr_number", "eway_bill_no") if fields.get(k)), state["filename"]
    )
    ids: list[int] = []
    with tracer.span("supervisor", "node", "comms"):
        if _needs_review(state):
            issues = [i["message"] for i in ext.get("issues", [])]
            if state.get("flags", {}).get("prompt_injection"):
                issues.insert(0, f"possible prompt injection in document text: {state['flags']['prompt_injection']}")
            ids.append(
                run_comms(
                    state["client_id"],
                    "field_review",
                    {"reference": reference, "issues": issues},
                    tracer,
                    llm,
                    document_id=state["document_id"],
                )
            )
        elif rec.get("mismatch_count"):
            mismatches = [c["check"] for c in rec["checks"] if not c["ok"]]
            if rec.get("rate_check") and rec["rate_check"].get("ok") is False:
                mismatches.append(
                    f"freight Rs {rec['rate_check']['billed']:,.0f} vs contract Rs {rec['rate_check']['contract_rate']:,.0f}"
                )
            ids.append(
                run_comms(
                    state["client_id"],
                    "billing_dispute",
                    {"reference": reference, "mismatches": mismatches, "summary": rec.get("summary")},
                    tracer,
                    llm,
                    document_id=state["document_id"],
                )
            )
    return {"approval_ids": ids}


def finalize_node(state: DocState, config: RunnableConfig) -> DocState:
    tracer, _, _, _ = _ctx(config)
    ext = state.get("extraction") or {}
    status = "needs_review" if _needs_review(state) else "done"
    if state.get("doc_type") == "unknown":
        status = "needs_review"
    _set_status(
        state["document_id"],
        status,
        doc_type=state.get("doc_type"),
        fields=ext.get("fields"),
        confidence=ext.get("confidence"),
        issues=ext.get("issues"),
        extraction_method=ext.get("method"),
        reconciliation=state.get("reconciliation"),
        guardrail_flags=state.get("flags"),
        cost_usd=tracer.total_cost,
    )
    return {"status": status}


def _needs_review(state: DocState) -> bool:
    ext = state.get("extraction") or {}
    return bool(ext.get("needs_review") or state.get("flags", {}).get("prompt_injection"))


def route_after_intake(state: DocState) -> str:
    return "extraction" if state["doc_type"] in RECONCILABLE else "finalize"


def route_after_extraction(state: DocState) -> str:
    # Never reconcile (or dispute a bill) on fields we do not trust yet.
    return "comms" if _needs_review(state) else "reconciliation"


def route_after_reconciliation(state: DocState) -> str:
    return "comms" if (state.get("reconciliation") or {}).get("mismatch_count") else "finalize"


def build_graph():
    g = StateGraph(DocState)
    g.add_node("intake", intake_node)
    g.add_node("extraction", extraction_node)
    g.add_node("reconciliation", reconciliation_node)
    g.add_node("comms", comms_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, ["extraction", "finalize"])
    g.add_conditional_edges("extraction", route_after_extraction, ["comms", "reconciliation"])
    g.add_conditional_edges("reconciliation", route_after_reconciliation, ["comms", "finalize"])
    g.add_edge("comms", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=InMemorySaver())


GRAPH = build_graph()


def process_document(
    document_id: int, client_id: str, filename: str, text_result: TextResult, raw: bytes, tracer: Tracer, llm: ClaudeLLM | None
) -> DocState:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"doc-{document_id}-{tracer.run_id}",
            "tracer": tracer,
            "llm": llm,
            "text_result": text_result,
            "raw": raw,
        }
    }
    try:
        return GRAPH.invoke({"document_id": document_id, "client_id": client_id, "filename": filename}, config)
    except Exception as exc:
        _set_status(document_id, "failed", error=f"{type(exc).__name__}: {exc}", cost_usd=tracer.total_cost)
        raise
