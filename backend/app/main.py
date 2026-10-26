"""HTTP API: documents, live agent traces (SSE), approvals inbox, fleet, memory, search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select

from .agents.comms import read_client_memory, write_client_memory
from .agents.fleet_monitor import run_fleet_check
from .agents.supervisor import process_document
from .config import BASE_DIR, settings
from .db import Approval, Document, FleetAlert, Outbox, TraceSpan, init_db, session_scope
from .llm import get_llm
from .schemas import Trip
from .tools.ocr import extract_text
from .tools.retrieval import get_index
from .tools.vision import YardEvent, record_yard_events
from .tracing import Tracer

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("freightpilot")

UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
TERMINAL = {"done", "needs_review", "failed"}
ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    UPLOAD_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(title="FreightPilot", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- tenancy


def _api_keys() -> dict[str, str]:
    """FP_API_KEYS="acme-steel:secret1,deccan-pharma:secret2". Unset = dev mode, no auth."""
    pairs = (p.split(":", 1) for p in os.getenv("FP_API_KEYS", "").split(",") if ":" in p)
    return {key.strip(): client.strip() for client, key in pairs}


def tenant(client_id: str = Query("acme-steel"), x_api_key: str | None = Header(None)) -> str:
    keys = _api_keys()
    if not keys:
        return client_id
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(401, "missing or invalid X-API-Key")
    return keys[x_api_key]  # the key decides the tenant; the query param cannot override it


# ---------------------------------------------------------------- serialisers


def _doc(d: Document, spans: list[TraceSpan] | None = None) -> dict:
    out = {
        "id": d.id,
        "client_id": d.client_id,
        "filename": d.filename,
        "status": d.status,
        "doc_type": d.doc_type,
        "fields": d.fields,
        "confidence": d.confidence,
        "issues": d.issues or [],
        "reconciliation": d.reconciliation,
        "guardrail_flags": d.guardrail_flags or {},
        "extraction_method": d.extraction_method,
        "cost_usd": round(d.cost_usd or 0.0, 6),
        "error": d.error,
        "created_at": d.created_at.isoformat(),
    }
    if spans is not None:
        out["trace"] = [_span(s) for s in spans]
    return out


def _span(s: TraceSpan) -> dict:
    return {
        "id": s.id,
        "run_id": s.run_id,
        "agent": s.agent,
        "kind": s.kind,
        "name": s.name,
        "status": s.status,
        "input": s.input_summary,
        "output": s.output_summary,
        "model": s.model,
        "input_tokens": s.input_tokens,
        "output_tokens": s.output_tokens,
        "cost_usd": s.cost_usd,
        "latency_ms": round(s.latency_ms, 1),
        "at": s.created_at.isoformat(),
    }


def _approval(a: Approval) -> dict:
    return {
        "id": a.id,
        "client_id": a.client_id,
        "document_id": a.document_id,
        "trip_id": a.trip_id,
        "kind": a.kind,
        "status": a.status,
        "to": a.to,
        "cc": a.cc,
        "subject": a.subject,
        "body": a.body,
        "reason": a.reason,
        "decided_by": a.decided_by,
        "created_at": a.created_at.isoformat(),
        "decided_at": a.decided_at.isoformat() if a.decided_at else None,
    }


# ---------------------------------------------------------------- documents


@app.get("/api/health")
def health() -> dict:
    llm = get_llm()
    return {"status": "ok", "llm": "claude" if llm else "offline", "model": settings.model if llm else None}


def _run_pipeline(document_id: int, client_id: str, filename: str, path: Path) -> None:
    raw = path.read_bytes()
    tracer = Tracer(document_id=document_id)
    try:
        text_result = extract_text(filename, raw)
        process_document(document_id, client_id, filename, text_result, raw, tracer, get_llm())
    except Exception:
        log.exception("pipeline failed for document %s", document_id)
        with session_scope() as s:
            doc = s.get(Document, document_id)
            if doc.status not in TERMINAL:
                doc.status, doc.error = "failed", "unexpected error; see server logs"


def ingest(client_id: str, filename: str, data: bytes) -> tuple[Document, bool]:
    """Store an upload. Returns (document, created). Same bytes for the same client = same document."""
    digest = hashlib.sha256(data).hexdigest()
    with session_scope() as s:
        existing = s.scalar(select(Document).where(Document.client_id == client_id, Document.content_hash == digest))
        if existing:
            return existing, False
        suffix = Path(filename).suffix.lower()
        (UPLOAD_DIR / f"{digest}{suffix}").write_bytes(data)
        doc = Document(client_id=client_id, filename=Path(filename).name, content_hash=digest)
        s.add(doc)
        s.flush()
        return doc, True


@app.post("/api/documents", status_code=201)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...), client: str = Depends(tenant)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file larger than 15 MB")
    filename = file.filename or "upload.txt"
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"supported types: {', '.join(sorted(ALLOWED_SUFFIXES))}")
    doc, created = ingest(client, filename, data)
    if created:
        path = UPLOAD_DIR / f"{doc.content_hash}{Path(filename).suffix.lower()}"
        background.add_task(_run_pipeline, doc.id, client, doc.filename, path)
    return {**_doc(doc), "duplicate": not created}


@app.get("/api/documents")
def list_documents(client: str = Depends(tenant), status: str | None = None, limit: int = Query(100, le=500)) -> list[dict]:
    with session_scope() as s:
        q = select(Document).where(Document.client_id == client).order_by(Document.id.desc()).limit(limit)
        if status:
            q = q.where(Document.status == status)
        return [_doc(d) for d in s.scalars(q)]


def _get_doc(s, document_id: int, client: str) -> Document:
    doc = s.get(Document, document_id)
    if doc is None or doc.client_id != client:  # 404, not 403: don't reveal other tenants' ids
        raise HTTPException(404, "document not found")
    return doc


@app.get("/api/documents/{document_id}")
def get_document(document_id: int, client: str = Depends(tenant)) -> dict:
    with session_scope() as s:
        doc = _get_doc(s, document_id, client)
        spans = s.scalars(select(TraceSpan).where(TraceSpan.document_id == document_id).order_by(TraceSpan.id)).all()
        return {**_doc(doc, spans), "raw_text": doc.raw_text}


@app.post("/api/documents/{document_id}/reprocess")
def reprocess(document_id: int, background: BackgroundTasks, client: str = Depends(tenant)) -> dict:
    with session_scope() as s:
        doc = _get_doc(s, document_id, client)
        if doc.status not in TERMINAL:
            raise HTTPException(409, "document is still processing")
        doc.status, doc.error = "queued", None
        path = UPLOAD_DIR / f"{doc.content_hash}{Path(doc.filename).suffix.lower()}"
        background.add_task(_run_pipeline, doc.id, client, doc.filename, path)
        return _doc(doc)


@app.get("/api/documents/{document_id}/events")
async def document_events(document_id: int, client: str = Depends(tenant)) -> StreamingResponse:
    """Server-sent events: each trace span as it is written, then a final status event."""
    with session_scope() as s:
        _get_doc(s, document_id, client)

    async def stream():
        last_id, idle = 0, 0
        while idle < 1200:  # ~10 minutes max
            with session_scope() as s:
                spans = s.scalars(
                    select(TraceSpan).where(TraceSpan.document_id == document_id, TraceSpan.id > last_id).order_by(TraceSpan.id)
                ).all()
                status = s.get(Document, document_id).status
            for span in spans:
                last_id = span.id
                yield f"event: span\ndata: {json.dumps(_span(span))}\n\n"
            if status in TERMINAL and not spans:
                yield f"event: status\ndata: {json.dumps({'status': status})}\n\n"
                return
            idle = 0 if spans else idle + 1
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- approvals (human in the loop)


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    decided_by: str = "ops"
    subject: str | None = None
    body: str | None = None


@app.get("/api/approvals")
def list_approvals(client: str = Depends(tenant), status: str | None = "pending") -> list[dict]:
    with session_scope() as s:
        q = select(Approval).where(Approval.client_id == client).order_by(Approval.id.desc())
        if status:
            q = q.where(Approval.status == status)
        return [_approval(a) for a in s.scalars(q)]


@app.post("/api/approvals/{approval_id}/decision")
def decide(approval_id: int, body: Decision, client: str = Depends(tenant)) -> dict:
    with session_scope() as s:
        a = s.get(Approval, approval_id)
        if a is None or a.client_id != client:
            raise HTTPException(404, "approval not found")
        if a.status != "pending":
            raise HTTPException(409, f"already {a.status}")
        if body.subject is not None:
            a.subject = body.subject
        if body.body is not None:
            a.body = body.body
        a.status = "approved" if body.decision == "approve" else "rejected"
        a.decided_by, a.decided_at = body.decided_by, datetime.now(UTC)
        # The only path in the codebase that releases a message.
        if a.status == "approved" and a.kind != "field_review":
            s.add(Outbox(approval_id=a.id, to=a.to or "unknown@client", subject=a.subject, body=a.body))
        return _approval(a)


@app.get("/api/outbox")
def outbox(client: str = Depends(tenant)) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(
            select(Outbox, Approval)
            .join(Approval, Outbox.approval_id == Approval.id)
            .where(Approval.client_id == client)
            .order_by(Outbox.id.desc())
        ).all()
        return [
            {"id": o.id, "approval_id": o.approval_id, "to": o.to, "subject": o.subject, "sent_at": o.sent_at.isoformat()}
            for o, _ in rows
        ]


# ---------------------------------------------------------------- client memory


class MemoryValue(BaseModel):
    value: str


@app.get("/api/memory")
def get_memory(client: str = Depends(tenant)) -> dict:
    return read_client_memory(client)


@app.put("/api/memory/{key}")
def put_memory(key: str, body: MemoryValue, client: str = Depends(tenant)) -> dict:
    write_client_memory(client, key, body.value)
    return read_client_memory(client)


# ---------------------------------------------------------------- fleet


@app.post("/api/fleet/check")
def fleet_check(trip: Trip, client: str = Depends(tenant)) -> dict:
    if trip.client_id != client:
        raise HTTPException(403, "trip belongs to another client")
    return run_fleet_check(trip, Tracer(), get_llm())


@app.get("/api/fleet/trace/{run_id}")
def fleet_trace(run_id: str, _: str = Depends(tenant)) -> list[dict]:
    with session_scope() as s:
        return [_span(x) for x in s.scalars(select(TraceSpan).where(TraceSpan.run_id == run_id).order_by(TraceSpan.id))]


@app.get("/api/fleet/alerts")
def fleet_alerts(client: str = Depends(tenant), limit: int = Query(100, le=500)) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(select(FleetAlert).where(FleetAlert.client_id == client).order_by(FleetAlert.id.desc()).limit(limit))
        return [
            {
                "id": r.id,
                "trip_id": r.trip_id,
                "vehicle_no": r.vehicle_no,
                "kind": r.kind,
                "severity": r.severity,
                "detail": r.detail,
                "actions": r.actions,
                "delay_risk": r.delay_risk,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


@app.post("/api/yard/events")
def yard_events(events: list[dict], _: str = Depends(tenant)) -> dict:
    record_yard_events([YardEvent(**e) for e in events])
    return {"recorded": len(events)}


# ---------------------------------------------------------------- search + metrics


@app.get("/api/rate-cards/search")
def search_rate_cards(q: str, client: str = Depends(tenant), k: int = Query(5, le=20)) -> list[dict]:
    return [h.to_dict() for h in get_index().search(q, k=k, source_prefix=client)]


@app.get("/api/metrics")
def metrics(client: str = Depends(tenant)) -> dict:
    with session_scope() as s:
        by_status = dict(
            s.execute(select(Document.status, func.count()).where(Document.client_id == client).group_by(Document.status)).all()
        )
        by_method = dict(
            s.execute(
                select(Document.extraction_method, func.count())
                .where(Document.client_id == client, Document.extraction_method.is_not(None))
                .group_by(Document.extraction_method)
            ).all()
        )
        cost = s.scalar(select(func.coalesce(func.sum(Document.cost_usd), 0.0)).where(Document.client_id == client))
        pending = s.scalar(
            select(func.count()).select_from(Approval).where(Approval.client_id == client, Approval.status == "pending")
        )
    finished = by_status.get("done", 0) + by_status.get("needs_review", 0)
    return {
        "documents": by_status,
        "extraction_methods": by_method,
        "pending_approvals": pending,
        "auto_verified_rate": round(by_status.get("done", 0) / finished, 3) if finished else None,
        "total_cost_usd": round(cost, 6),
        "cost_per_document_usd": round(cost / finished, 6) if finished else None,
    }


# ---------------------------------------------------------------- demo data


@app.post("/api/demo/seed")
def seed_demo(background: BackgroundTasks, client: str = Depends(tenant), shipments: int = Query(9, le=40)) -> dict:
    """Load synthetic shipments and fleet scenarios so the UI has something to show."""
    from evals.dataset import build_fleet_scenarios, build_shipments

    created = 0
    for ship in build_shipments(shipments):
        for sample in ship.docs:
            doc, new = ingest(client, f"{sample.id}.txt", sample.text.encode())
            if new:
                created += 1
                # Sequential, so later documents in a shipment can reconcile against earlier ones.
                background.add_task(_run_pipeline, doc.id, client, doc.filename, UPLOAD_DIR / f"{doc.content_hash}.txt")
    trips = []
    for sc in build_fleet_scenarios():
        if sc.yard_events:
            record_yard_events([YardEvent(**e) for e in sc.yard_events])
        trips.append(run_fleet_check(Trip(**{**sc.trip, "client_id": client}), Tracer(), get_llm())["trip_id"])
    return {"documents_queued": created, "trips_checked": trips}
