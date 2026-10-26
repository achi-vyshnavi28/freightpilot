import { useEffect, useState } from "react";
import { api, type Alert } from "./api";

const ACTION: Record<string, string> = {
  update_eta: "Updated ETA",
  raise_alert: "Raised alert",
  request_driver_call: "Driver call requested",
};

export default function Fleet() {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    api.alerts().then(setAlerts);
  }, []);

  const trips = Object.values(
    alerts.reduce<Record<string, Alert[]>>((acc, a) => {
      (acc[a.trip_id] ??= []).push(a);
      return acc;
    }, {}),
  );

  return (
    <section className="panel">
      <p className="muted">Code detects the anomalies. The fleet agent decides what to do about them; every action is listed.</p>
      {!trips.length && <p className="muted empty">No trips checked yet. Load the demo data from the Documents tab.</p>}
      <div className="cards">
        {trips.map((group) => {
          const first = group[0];
          const risk = first.delay_risk ?? 0;
          return (
            <article key={first.trip_id} className="card">
              <header className="row between">
                <b>{first.trip_id.replace(/_/g, " ")}</b>
                <span className="muted small">{first.vehicle_no}</span>
              </header>
              <div className="risk" title="Model-predicted probability of missing the deadline">
                <span>Delay risk</span>
                <span className="bar wide">
                  <span style={{ width: `${risk * 100}%` }} className={risk >= 0.6 ? "lo" : "hi"} />
                </span>
                <b>{Math.round(risk * 100)}%</b>
              </div>
              <ul className="checks">
                {group.map((a) => (
                  <li key={a.id} className={a.kind === "on_track" ? "pass" : a.severity === "high" ? "fail" : "warn"}>
                    {a.kind.replace(/_/g, " ")}: {a.detail}
                  </li>
                ))}
              </ul>
              <div className="actions">
                {first.actions.length ? (
                  first.actions.map((x, i) => (
                    <span key={i} className="tag">
                      {ACTION[x.action] ?? x.action}
                    </span>
                  ))
                ) : (
                  <span className="muted small">No action needed</span>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
