"""Comms agent: drafts client messages in each client's remembered style.

It can only *submit drafts*. Releasing a message is a human action through the
approvals API, which is the only code path that writes to the outbox.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from ..db import Approval, ClientMemory, session_scope
from ..llm import ClaudeLLM, LLMRefusal
from ..tracing import Tracer
from .common import call_tool, decision

AGENT = "comms"

DEFAULT_MEMORY = {"tone": "formal", "language": "English", "contact_email": "", "cc": "", "salutation": "Dear Sir/Madam"}


def read_client_memory(client_id: str) -> dict:
    with session_scope() as s:
        rows = s.scalars(select(ClientMemory).where(ClientMemory.client_id == client_id)).all()
        return {**DEFAULT_MEMORY, **{r.key: r.value for r in rows}}


def write_client_memory(client_id: str, key: str, value: str) -> str:
    with session_scope() as s:
        row = s.scalar(select(ClientMemory).where(ClientMemory.client_id == client_id, ClientMemory.key == key))
        if row:
            row.value = value
        else:
            s.add(ClientMemory(client_id=client_id, key=key, value=value))
    return "saved"


def find_open_thread(client_id: str, kind: str, thread_key: str | None) -> dict | None:
    if not thread_key:
        return None
    with session_scope() as s:
        a = s.scalar(
            select(Approval).where(
                Approval.client_id == client_id,
                Approval.kind == kind,
                Approval.thread_key == thread_key,
                Approval.status == "pending",
            )
        )
        return {"id": a.id, "facts": a.facts or {}} if a else None


def submit_draft_for_approval(
    client_id: str,
    kind: str,
    to: str,
    cc: list[str],
    subject: str,
    body: str,
    reason: str,
    document_id: int | None = None,
    trip_id: str | None = None,
    thread_key: str | None = None,
    facts: dict | None = None,
) -> int:
    with session_scope() as s:
        existing = None
        if thread_key:
            existing = s.scalar(
                select(Approval).where(
                    Approval.client_id == client_id,
                    Approval.kind == kind,
                    Approval.thread_key == thread_key,
                    Approval.status == "pending",
                )
            )
        if existing:  # update the open draft instead of sending the client a second email
            existing.subject, existing.body, existing.reason, existing.facts = subject, body, reason, facts
            return existing.id
        approval = Approval(
            client_id=client_id,
            kind=kind,
            to=to,
            cc=cc,
            subject=subject,
            body=body,
            reason=reason,
            document_id=document_id,
            trip_id=trip_id,
            thread_key=thread_key,
            facts=facts,
        )
        s.add(approval)
        s.flush()
        return approval.id


def _merge_facts(old: dict, new: dict) -> dict:
    merged = {**old, **new}
    seen = list(old.get("mismatches", []))
    seen += [m for m in new.get("mismatches", []) if m not in seen]
    if seen:
        merged["mismatches"] = seen
    return merged


def _fmt_eta(value: str | None) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return str(value)


def _template(kind: str, facts: dict, memory: dict) -> tuple[str, str]:
    greet = memory.get("salutation") or "Dear Sir/Madam"
    sign = "Regards,\nOperations Desk"
    if kind == "billing_dispute":
        subject = f"Billing query on {facts.get('reference', 'your recent document')}"
        lines = "\n".join(f"- {m}" for m in facts.get("mismatches", []))
        body = f"{greet},\n\nWhile reconciling {facts.get('reference', 'the document')} we found the following:\n{lines}\n\nPlease review and share a corrected copy or your clarification.\n\n{sign}"
    elif kind == "delay_notice":
        subject = f"Delivery update: {facts.get('vehicle_no')} ({facts.get('origin')} to {facts.get('destination')})"
        body = f"{greet},\n\nYour consignment on vehicle {facts.get('vehicle_no')} is running behind schedule. Reason: {facts.get('reason')}.\nRevised ETA: {_fmt_eta(facts.get('eta'))}.\n\nWe are tracking it closely and will update you if anything changes.\n\n{sign}"
    else:  # field_review
        subject = f"Document needs review: {facts.get('reference')}"
        body = f"Internal: {facts.get('reference')} could not be auto-verified.\nIssues:\n" + "\n".join(
            f"- {i}" for i in facts.get("issues", [])
        )
    return subject, body


DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
    "additionalProperties": False,
}


def run_comms(
    client_id: str,
    kind: str,
    facts: dict,
    tracer: Tracer,
    llm: ClaudeLLM | None,
    document_id: int | None = None,
    trip_id: str | None = None,
    thread_key: str | None = None,
) -> int:
    memory = call_tool(tracer, AGENT, "read_client_memory", read_client_memory, client_id=client_id)
    open_thread = find_open_thread(client_id, kind, thread_key)
    if open_thread:
        facts = _merge_facts(open_thread["facts"], facts)
        decision(tracer, AGENT, "merged_into_open_draft", {"approval_id": open_thread["id"], "thread_key": thread_key})
    subject, body = _template(kind, facts, memory)
    if llm is not None and kind != "field_review":
        try:
            out = llm.structured(
                agent=AGENT,
                name="draft_message",
                system=(
                    f"You write short logistics emails for an Indian transporter. Tone: {memory['tone']}. "
                    f"Language: {memory['language']}. Salutation: {memory['salutation']}. Use only the facts given; "
                    "never promise refunds, credits or dates that are not in the facts."
                ),
                content=f"Message type: {kind}\nFacts: {facts}\nPast context: {memory.get('last_' + kind, 'none')}",
                schema=DRAFT_SCHEMA,
                tracer=tracer,
            )
            subject, body = out["subject"], out["body"]
        except LLMRefusal as exc:
            decision(tracer, AGENT, "llm_refused", str(exc))

    to = memory.get("contact_email", "") if kind != "field_review" else "ops-review@internal"
    cc = [c.strip() for c in memory.get("cc", "").split(",") if c.strip()] if kind != "field_review" else []
    approval_id = call_tool(
        tracer,
        AGENT,
        "submit_draft_for_approval",
        submit_draft_for_approval,
        client_id=client_id,
        kind=kind,
        to=to,
        cc=cc,
        subject=subject,
        body=body,
        reason=facts.get("summary") or facts.get("reason", ""),
        document_id=document_id,
        trip_id=trip_id,
        thread_key=thread_key,
        facts=facts,
    )
    call_tool(
        tracer,
        AGENT,
        "write_client_memory",
        write_client_memory,
        client_id=client_id,
        key=f"last_{kind}",
        value=f"{datetime.now(UTC).date().isoformat()}: {subject}",
    )
    return approval_id
