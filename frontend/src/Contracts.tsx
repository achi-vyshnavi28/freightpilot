import { useEffect, useState, type FormEvent } from "react";
import { api, type Hit } from "./api";

const MEMORY_KEYS = ["contact_email", "cc", "tone", "language", "salutation"];

export default function Contracts() {
  const [q, setQ] = useState("detention charges");
  const [hits, setHits] = useState<Hit[]>([]);
  const [memory, setMemory] = useState<Record<string, string>>({});

  const search = (e?: FormEvent) => {
    e?.preventDefault();
    api.search(q).then(setHits);
  };

  useEffect(() => {
    search();
    api.memory().then(setMemory);
  }, []);

  return (
    <div className="split">
      <section className="panel">
        <h3>Contract search</h3>
        <p className="muted small">Hybrid retrieval (BM25 + TF-IDF, rank fusion). The reconciliation agent cites these chunk ids.</p>
        <form onSubmit={search} className="row">
          <input value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search the contract" />
          <button className="primary">Search</button>
        </form>
        <ol className="hits">
          {hits.map((h) => (
            <li key={h.id}>
              <code>{h.id}</code> <span className="muted small">score {h.score}</span>
              <pre>{h.text}</pre>
            </li>
          ))}
        </ol>
      </section>
      <section className="panel">
        <h3>Client memory</h3>
        <p className="muted small">Long-term memory the comms agent reads before drafting. Edit it to change how drafts are written.</p>
        {MEMORY_KEYS.map((k) => (
          <MemoryField key={k} name={k} value={memory[k] ?? ""} onSave={(v) => api.setMemory(k, v).then(setMemory)} />
        ))}
        {Object.entries(memory)
          .filter(([k]) => k.startsWith("last_"))
          .map(([k, v]) => (
            <p key={k} className="small">
              <code>{k}</code> {v}
            </p>
          ))}
      </section>
    </div>
  );
}

function MemoryField({ name, value, onSave }: { name: string; value: string; onSave: (v: string) => void }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <label className="memory">
      {name.replace(/_/g, " ")}
      <span className="row">
        <input value={v} onChange={(e) => setV(e.target.value)} />
        <button disabled={v === value} onClick={() => onSave(v)}>
          Save
        </button>
      </span>
    </label>
  );
}
