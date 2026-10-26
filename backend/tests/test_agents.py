"""Agent behaviour, including the Claude tool loop driven by a scripted fake client."""

from types import SimpleNamespace

from app.agents.fleet_monitor import run_fleet_check
from app.agents.supervisor import process_document
from app.db import Approval, Document, Outbox, TraceSpan, session_scope
from app.llm import ClaudeLLM, Tool
from app.main import UPLOAD_DIR, ingest
from app.schemas import Trip
from app.tools.ocr import extract_text
from app.tools.vision import _DEMO_EVENTS, YardEvent, record_yard_events
from app.tracing import Tracer
from evals.dataset import build_fleet_scenarios, build_shipments


def _process(client, sample, llm=None):
    UPLOAD_DIR.mkdir(exist_ok=True)
    doc, _ = ingest(client, f"{sample.id}.txt", sample.text.encode())
    return process_document(
        doc.id,
        client,
        doc.filename,
        extract_text("x.txt", sample.text.encode()),
        sample.text.encode(),
        Tracer(document_id=doc.id),
        llm,
    ), doc.id


def test_clean_shipment_is_auto_verified():
    ship = build_shipments(1)[0]
    for sample in ship.docs:
        state, _ = _process("acme-steel", sample)
        assert state["status"] == "done", sample.id
    with session_scope() as s:
        assert s.query(Approval).count() == 0


def test_overcharge_drafts_dispute_but_nothing_is_sent():
    ship = next(s for s in build_shipments(9) if s.mismatch_kind == "overcharge")
    state, doc_id = _process("acme-steel", ship.docs[0])
    rate = state["reconciliation"]["rate_check"]
    assert rate["ok"] is False and rate["citation"]["id"].startswith("acme-steel#")
    with session_scope() as s:
        approval = s.query(Approval).filter_by(document_id=doc_id).one()
        assert approval.kind == "billing_dispute" and approval.status == "pending"
        assert s.query(Outbox).count() == 0


def test_injected_invoice_goes_to_review_and_skips_reconciliation():
    ship = build_shipments(11)[10]
    state, doc_id = _process("acme-steel", ship.docs[0])
    assert state["status"] == "needs_review"
    assert state["flags"]["prompt_injection"]
    assert "reconciliation" not in state
    with session_scope() as s:
        assert s.get(Document, doc_id).status == "needs_review"
        assert s.query(Approval).filter_by(kind="field_review").count() == 1


def test_every_step_is_traced():
    _, doc_id = _process("acme-steel", build_shipments(1)[0].docs[0])
    with session_scope() as s:
        names = [sp.name for sp in s.query(TraceSpan).filter_by(document_id=doc_id).order_by(TraceSpan.id)]
    for expected in ("screen_document", "classify_document", "run_rule_extractor", "search_rate_cards", "intake", "extraction"):
        assert expected in names


def _scenario(name):
    return next(s for s in build_fleet_scenarios() if s.name == name)


def test_fleet_playbook_actions():
    _DEMO_EVENTS.clear()
    stuck = run_fleet_check(Trip(**_scenario("stop_late_and_stuck").trip), Tracer(), None)
    assert any(a["action"] == "request_driver_call" for a in stuck["actions"])
    assert stuck["approval_id"] is not None  # delay notice drafted for approval


def test_yard_event_explains_long_stop():
    sc = _scenario("stop_at_warehouse")
    _DEMO_EVENTS.clear()
    record_yard_events([YardEvent(**e) for e in sc.yard_events])
    out = run_fleet_check(Trip(**sc.trip), Tracer(), None)
    assert not any(a["action"] == "request_driver_call" for a in out["actions"])


# ---------------------------------------------------------------- Claude loop with a fake client


def _block(**kw):
    return SimpleNamespace(**kw)


def _response(content, stop):
    return SimpleNamespace(
        content=content, stop_reason=stop, stop_details=None, usage=SimpleNamespace(input_tokens=100, output_tokens=20)
    )


class FakeMessages:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **params):
        self.requests.append(params)
        return self.script.pop(0)


def _fake_llm(script) -> ClaudeLLM:
    llm = ClaudeLLM.__new__(ClaudeLLM)
    messages = FakeMessages(script)
    llm.client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
    return llm


def test_tool_loop_denies_tools_outside_the_allowlist():
    calls = []
    script = [
        _response(
            [
                _block(type="tool_use", id="t1", name="raise_alert", input={"severity": "high", "detail": "x"}),
                _block(type="tool_use", id="t2", name="read_client_memory", input={}),
            ],
            "tool_use",
        ),
        _response([_block(type="text", text="done")], "end_turn"),
    ]
    llm = _fake_llm(script)
    tools = [
        Tool(
            "read_client_memory",
            "read",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            lambda: calls.append("read") or {"tone": "formal"},
        )
    ]
    tracer = Tracer(persist=False)
    text, results = llm.run_tools(agent="comms", system="s", prompt="p", tools=tools, tracer=tracer)

    assert text == "done" and calls == ["read"]
    denied = next(r for r in results if r.name == "raise_alert")
    assert denied.is_error and "not allowed" in denied.output
    # Both results returned together in one user turn, assistant content passed back unchanged.
    second = llm.client.messages.requests[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"] is script[0].content
    assert [r["tool_use_id"] for r in second[2]["content"]] == ["t1", "t2"]
    assert any(s.status == "denied" for s in tracer.spans)
    assert tracer.total_cost > 0


def test_repair_loop_rejects_corrections_not_in_document():
    ship = next(s for s in build_shipments(9) if "arithmetic" in s.docs[0].tags)
    invoice = ship.docs[0]
    script = [
        # The model tries to "fix" the total to make the arithmetic pass: that number is not on the paper.
        _response(
            [
                _block(
                    type="tool_use",
                    id="r1",
                    name="request_human_review",
                    input={"corrections_json": '{"total_amount": 1.5}', "reason": "total looks off"},
                )
            ],
            "tool_use",
        ),
        _response([_block(type="text", text="submitted")], "end_turn"),
    ]
    state, _ = _process("acme-steel", invoice, llm=_fake_llm_with_structured(script, invoice.expected_fields))
    assert state["status"] == "needs_review"
    assert state["extraction"]["fields"]["total_amount"] == invoice.expected_fields["total_amount"]


def _fake_llm_with_structured(script, fields):
    llm = _fake_llm(script)
    # LLM extraction tier returns the same (correct) values the rules found.
    llm.structured = lambda **kw: (
        {k: v for k, v in fields.items()}
        if kw["name"] == "run_llm_extractor"
        else {"summary": "", "cited_chunk_ids": [], "subject": "s", "body": "b"}
    )
    return llm
