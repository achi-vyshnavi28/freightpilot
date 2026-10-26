from fastapi.testclient import TestClient

from app.main import app
from evals.dataset import build_shipments


def _client():
    return TestClient(app)


def test_upload_process_and_idempotency():
    sample = build_shipments(1)[0].docs[0]
    with _client() as c:
        r = c.post("/api/documents?client_id=acme-steel", files={"file": ("inv.txt", sample.text.encode(), "text/plain")})
        assert r.status_code == 201
        doc_id = r.json()["id"]
        detail = c.get(f"/api/documents/{doc_id}?client_id=acme-steel").json()  # background task ran before TestClient returned
        assert detail["status"] == "done" and detail["doc_type"] == "gst_invoice"
        assert detail["trace"]
        again = c.post(
            "/api/documents?client_id=acme-steel", files={"file": ("copy.txt", sample.text.encode(), "text/plain")}
        ).json()
        assert again["duplicate"] and again["id"] == doc_id


def test_rejects_unsupported_type():
    with _client() as c:
        r = c.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
        assert r.status_code == 415


def test_sse_stream_ends_with_status():
    sample = build_shipments(1)[0].docs[1]
    with _client() as c:
        doc_id = c.post("/api/documents", files={"file": ("e.txt", sample.text.encode(), "text/plain")}).json()["id"]
        body = c.get(f"/api/documents/{doc_id}/events").text
        assert "event: span" in body and 'event: status\ndata: {"status": "done"}' in body


def test_approval_flow_writes_outbox_only_on_approve():
    ship = next(s for s in build_shipments(9) if s.mismatch_kind == "overcharge")
    with _client() as c:
        c.post("/api/documents", files={"file": ("inv.txt", ship.docs[0].text.encode(), "text/plain")})
        (pending,) = c.get("/api/approvals").json()
        assert c.get("/api/outbox").json() == []
        r = c.post(f"/api/approvals/{pending['id']}/decision", json={"decision": "approve", "body": "edited body"})
        assert r.json()["status"] == "approved"
        assert len(c.get("/api/outbox").json()) == 1
        assert c.post(f"/api/approvals/{pending['id']}/decision", json={"decision": "reject"}).status_code == 409


def test_tenant_isolation_with_api_keys(monkeypatch):
    monkeypatch.setenv("FP_API_KEYS", "acme-steel:k-acme,deccan-pharma:k-deccan")
    sample = build_shipments(1)[0].docs[0]
    with _client() as c:
        assert c.get("/api/documents").status_code == 401
        doc_id = c.post(
            "/api/documents", headers={"X-API-Key": "k-acme"}, files={"file": ("i.txt", sample.text.encode(), "text/plain")}
        ).json()["id"]
        # Deccan's key cannot read Acme's document, even by asking for client_id=acme-steel.
        r = c.get(f"/api/documents/{doc_id}?client_id=acme-steel", headers={"X-API-Key": "k-deccan"})
        assert r.status_code == 404


def test_metrics_and_search():
    with _client() as c:
        hits = c.get("/api/rate-cards/search", params={"q": "Hyderabad Chennai 10 T"}).json()
        assert hits and "42,000" in hits[0]["text"]
        assert c.get("/api/metrics").json()["pending_approvals"] == 0


def test_fleet_check_endpoint():
    from evals.dataset import build_fleet_scenarios

    trip = build_fleet_scenarios()[1].trip
    with _client() as c:
        out = c.post("/api/fleet/check", json=trip).json()
        assert out["actions"]
        assert c.get("/api/fleet/alerts").json()
        assert c.get(f"/api/fleet/trace/{out['run_id']}").json()
