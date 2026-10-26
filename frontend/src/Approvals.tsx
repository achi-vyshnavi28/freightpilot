import { useCallback, useEffect, useState } from "react";
import { api, type Approval } from "./api";

const KIND: Record<string, string> = {
  billing_dispute: "Billing dispute",
  delay_notice: "Delay notice",
  field_review: "Needs human review",
  client_update: "Client update",
};

export default function Approvals({ onChange }: { onChange: () => void }) {
  const [items, setItems] = useState<Approval[]>([]);
  const [status, setStatus] = useState("pending");
  const load = useCallback(() => api.approvals(status).then(setItems), [status]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <section className="panel">
      <div className="row between">
        <p className="muted">Agents only draft. Nothing reaches a client until a person approves it here.</p>
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Filter">
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="">All</option>
        </select>
      </div>
      {!items.length && <p className="muted empty">Inbox zero.</p>}
      <div className="cards">
        {items.map((a) => (
          <ApprovalCard
            key={a.id}
            a={a}
            onDone={() => {
              load();
              onChange();
            }}
          />
        ))}
      </div>
    </section>
  );
}

function ApprovalCard({ a, onDone }: { a: Approval; onDone: () => void }) {
  const [subject, setSubject] = useState(a.subject);
  const [body, setBody] = useState(a.body);
  const [busy, setBusy] = useState(false);
  const pending = a.status === "pending";

  const decide = async (d: "approve" | "reject") => {
    setBusy(true);
    try {
      await api.decide(a.id, d, { subject, body });
      onDone();
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className={`card card-${a.kind}`}>
      <header className="row between">
        <span className="tag">{KIND[a.kind] ?? a.kind}</span>
        <span className="muted small">
          {a.document_id ? `document #${a.document_id}` : a.trip_id ? `trip ${a.trip_id}` : ""} · {new Date(a.created_at).toLocaleString()}
        </span>
      </header>
      {a.reason && <p className="small">{a.reason}</p>}
      <label>
        To <input value={a.to || "(no contact saved)"} readOnly />
      </label>
      <label>
        Subject <input value={subject} onChange={(e) => setSubject(e.target.value)} readOnly={!pending} />
      </label>
      <label>
        Message <textarea rows={8} value={body} onChange={(e) => setBody(e.target.value)} readOnly={!pending} />
      </label>
      {pending ? (
        <div className="row">
          <button className="primary" disabled={busy} onClick={() => decide("approve")}>
            {a.kind === "field_review" ? "Mark reviewed" : "Approve & send"}
          </button>
          <button disabled={busy} onClick={() => decide("reject")}>
            Reject
          </button>
        </div>
      ) : (
        <p className="muted small">{a.status}</p>
      )}
    </article>
  );
}
