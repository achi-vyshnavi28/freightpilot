import { useCallback, useEffect, useState } from "react";
import { api, setClient, type Metrics } from "./api";
import Documents from "./Documents";
import Approvals from "./Approvals";
import Fleet from "./Fleet";
import Contracts from "./Contracts";

const TABS = ["Documents", "Approvals", "Fleet", "Contracts"] as const;
type Tab = (typeof TABS)[number];
const CLIENTS = ["acme-steel", "deccan-pharma"];

export default function App() {
  const [tab, setTab] = useState<Tab>("Documents");
  const [client, setClientState] = useState(CLIENTS[0]);
  const [engine, setEngine] = useState<string>("…");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [tick, setTick] = useState(0);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    api.health().then((h) => setEngine(h.llm === "claude" ? `Claude · ${h.model}` : "Offline engine")).catch(() => setEngine("API down"));
  }, []);

  useEffect(() => {
    api.metrics().then(setMetrics).catch(() => setMetrics(null));
    const t = setInterval(() => api.metrics().then(setMetrics).catch(() => undefined), 4000);
    return () => clearInterval(t);
  }, [client, tick]);

  const switchClient = (id: string) => {
    setClient(id);
    setClientState(id);
    refresh();
  };

  const docs = metrics?.documents ?? {};
  const finished = (docs.done ?? 0) + (docs.needs_review ?? 0);

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden>▲</span>
          <div>
            <h1>FreightPilot</h1>
            <p>AI back office for transporters</p>
          </div>
        </div>
        <div className="top-right">
          <span className={`pill ${engine.startsWith("Claude") ? "pill-ok" : ""}`}>{engine}</span>
          <label className="client">
            Client
            <select value={client} onChange={(e) => switchClient(e.target.value)}>
              {CLIENTS.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <section className="kpis" aria-label="Key metrics">
        <Kpi label="Documents processed" value={String(finished)} />
        <Kpi label="Auto-verified" value={metrics?.auto_verified_rate != null ? `${Math.round(metrics.auto_verified_rate * 100)}%` : "—"} />
        <Kpi label="Needs review" value={String(docs.needs_review ?? 0)} />
        <Kpi label="Pending approvals" value={String(metrics?.pending_approvals ?? 0)} />
        <Kpi label="Cost / document" value={metrics?.cost_per_document_usd != null ? `$${metrics.cost_per_document_usd.toFixed(4)}` : "—"} />
      </section>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t}
            {t === "Approvals" && metrics?.pending_approvals ? <span className="count">{metrics.pending_approvals}</span> : null}
          </button>
        ))}
      </nav>

      <main key={client}>
        {tab === "Documents" && <Documents onChange={refresh} />}
        {tab === "Approvals" && <Approvals onChange={refresh} />}
        {tab === "Fleet" && <Fleet />}
        {tab === "Contracts" && <Contracts />}
      </main>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi">
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
    </div>
  );
}
