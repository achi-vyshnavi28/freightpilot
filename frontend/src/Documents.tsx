import { useCallback, useEffect, useRef, useState } from "react";
import { api, money, type Doc, type Span } from "./api";
import Trace from "./Trace";

const TERMINAL = new Set(["done", "needs_review", "failed"]);
const LABEL: Record<string, string> = {
  gst_invoice: "GST invoice",
  eway_bill: "E-way bill",
  lorry_receipt: "Lorry receipt",
  proof_of_delivery: "POD",
  unknown: "Unknown",
};

export default function Documents({ onChange }: { onChange: () => void }) {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(() => api.documents().then(setDocs).catch((e) => setError(e.message)), []);

  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [load]);

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    try {
      let last: Doc | null = null;
      for (const f of Array.from(files)) last = await api.upload(f);
      if (last) setSelected(last.id);
      await load();
      onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const seed = async () => {
    setBusy(true);
    try {
      await api.seed();
      await load();
      onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="split">
      <section className="panel list-panel">
        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            upload(e.dataTransfer.files);
          }}
        >
          <p>Drop invoices, e-way bills or lorry receipts</p>
          <p className="muted">.txt · .pdf · .png · .jpg</p>
          <div className="row">
            <button className="primary" disabled={busy} onClick={() => fileInput.current?.click()}>
              Upload
            </button>
            <button disabled={busy} onClick={seed}>
              Load demo data
            </button>
          </div>
          <input ref={fileInput} type="file" multiple hidden accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.webp" onChange={(e) => upload(e.target.files)} />
        </div>
        {error && <p className="error">{error}</p>}
        <ul className="doc-list">
          {docs.map((d) => (
            <li key={d.id}>
              <button className={selected === d.id ? "doc selected" : "doc"} onClick={() => setSelected(d.id)}>
                <span className="doc-name">{d.filename}</span>
                <span className="doc-meta">
                  {d.doc_type ? LABEL[d.doc_type] ?? d.doc_type : "…"}
                  {d.reconciliation?.mismatch_count ? <span className="tag tag-warn">{d.reconciliation.mismatch_count} mismatch</span> : null}
                  {d.guardrail_flags?.prompt_injection ? <span className="tag tag-bad">injection</span> : null}
                </span>
                <Status status={d.status} />
              </button>
            </li>
          ))}
          {!docs.length && <li className="muted empty">No documents yet. Upload one or load the demo data.</li>}
        </ul>
      </section>
      <section className="panel detail-panel">
        {selected ? <DocumentDetail id={selected} onChange={onChange} /> : <p className="muted empty">Select a document to see what each agent did.</p>}
      </section>
    </div>
  );
}

export function Status({ status }: { status: string }) {
  const cls = status === "done" ? "ok" : status === "needs_review" ? "warn" : status === "failed" ? "bad" : "run";
  return <span className={`status status-${cls}`}>{status.replace("_", " ")}</span>;
}

function DocumentDetail({ id, onChange }: { id: number; onChange: () => void }) {
  const [doc, setDoc] = useState<Doc | null>(null);
  const [live, setLive] = useState<Span[]>([]);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    let source: EventSource | null = null;
    setLive([]);
    api.document(id).then((d) => {
      setDoc(d);
      if (!TERMINAL.has(d.status)) {
        source = api.events(id);
        source.addEventListener("span", (e) => setLive((s) => [...s, JSON.parse((e as MessageEvent).data)]));
        source.addEventListener("status", () => {
          source?.close();
          api.document(id).then(setDoc);
          onChange();
        });
      }
    });
    return () => source?.close();
  }, [id, onChange]);

  if (!doc) return <p className="muted">Loading…</p>;
  const spans = doc.trace?.length && TERMINAL.has(doc.status) ? doc.trace : live;
  const issues = doc.issues ?? [];
  const rec = doc.reconciliation;

  return (
    <div className="detail">
      <div className="detail-head">
        <div>
          <h2>{doc.filename}</h2>
          <p className="muted">
            {doc.doc_type ? LABEL[doc.doc_type] ?? doc.doc_type : "classifying"} · extracted by <b>{doc.extraction_method ?? "—"}</b> · cost ${doc.cost_usd.toFixed(5)}
          </p>
        </div>
        <div className="row">
          <Status status={doc.status} />
          {TERMINAL.has(doc.status) && (
            <button onClick={() => api.reprocess(doc.id).then(() => setDoc({ ...doc, status: "queued", trace: [] }))}>Reprocess</button>
          )}
        </div>
      </div>

      {doc.guardrail_flags?.prompt_injection && (
        <div className="callout callout-bad">
          <b>Prompt injection blocked.</b> The document contains instructions aimed at an AI: “{doc.guardrail_flags.prompt_injection.join("”, “")}”. It was treated as data and routed to a human.
        </div>
      )}
      {doc.guardrail_flags?.pii_redacted && (
        <p className="muted small">
          PII redacted before any model call: {Object.entries(doc.guardrail_flags.pii_redacted).map(([k, v]) => `${v} ${k}`).join(", ")}
        </p>
      )}

      {issues.length > 0 && (
        <div className="callout callout-warn">
          <b>Validation</b>
          <ul>
            {issues.map((i, n) => (
              <li key={n}>
                <code>{i.field}</code> {i.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {doc.fields && (
        <>
          <h3>Extracted fields</h3>
          <table className="fields">
            <tbody>
              {Object.entries(doc.fields)
                .filter(([k]) => k !== "line_items")
                .map(([k, v]) => {
                  const c = doc.confidence?.[k] ?? 0;
                  return (
                    <tr key={k}>
                      <th>{k.replace(/_/g, " ")}</th>
                      <td>{typeof v === "number" && /amount|value|gst|total/.test(k) ? money(v) : String(v ?? "—")}</td>
                      <td className="conf">
                        <span className="bar" title={`confidence ${c}`}>
                          <span style={{ width: `${c * 100}%` }} className={c >= 0.75 ? "hi" : "lo"} />
                        </span>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </>
      )}

      {rec && (
        <>
          <h3>Reconciliation</h3>
          <p>{rec.summary}</p>
          {rec.checks.length > 0 && (
            <ul className="checks">
              {rec.checks.map((c, n) => (
                <li key={n} className={c.ok ? "pass" : "fail"}>
                  {c.ok ? "✓" : "✗"} {c.check}
                  {!c.ok && (
                    <span className="muted">
                      {" "}
                      ({String(c.ours)} vs {String(c.theirs)})
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {rec.rate_check?.citation && (
            <details className="citation">
              <summary>
                Contract rate {money(rec.rate_check.contract_rate)} for {rec.rate_check.vehicle_class_t} T vs billed {money(rec.rate_check.billed)} · source{" "}
                <code>{rec.rate_check.citation.id}</code>
              </summary>
              <pre>{rec.rate_check.citation.text}</pre>
            </details>
          )}
        </>
      )}

      <h3>Agent trace</h3>
      <Trace spans={spans} />

      {doc.raw_text && (
        <>
          <button className="link" onClick={() => setShowRaw(!showRaw)}>
            {showRaw ? "Hide" : "Show"} document text
          </button>
          {showRaw && <pre className="raw">{doc.raw_text}</pre>}
        </>
      )}
    </div>
  );
}
