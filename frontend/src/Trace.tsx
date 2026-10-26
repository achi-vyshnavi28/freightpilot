import type { Span } from "./api";

const ICON: Record<Span["kind"], string> = { node: "◆", llm: "✦", tool: "⚙", guardrail: "⛨", decision: "→" };

/** Spans are written when they finish, so a node arrives after its children. Put each node first. */
function nodeFirst(spans: Span[]): (Span & { child: boolean })[] {
  const out: (Span & { child: boolean })[] = [];
  let pending: Span[] = [];
  for (const s of spans) {
    if (s.kind === "node") {
      out.push({ ...s, child: false }, ...pending.map((p) => ({ ...p, child: true })));
      pending = [];
    } else pending.push(s);
  }
  return [...out, ...pending.map((p) => ({ ...p, child: false }))];
}

export default function Trace({ spans }: { spans: Span[] }) {
  if (!spans.length) return <p className="muted small">Waiting for the first step…</p>;
  const tokens = spans.reduce((a, s) => a + s.input_tokens + s.output_tokens, 0);
  const cost = spans.reduce((a, s) => a + s.cost_usd, 0);
  return (
    <>
      <p className="muted small">
        {spans.length} steps · {tokens.toLocaleString()} tokens · ${cost.toFixed(5)}
      </p>
      <ol className="trace">
        {nodeFirst(spans).map((s) => (
          <li key={s.id} className={`span span-${s.kind} span-${s.status}${s.child ? " span-child" : ""}`}>
            <details>
              <summary>
                <span className="span-icon" aria-hidden>
                  {ICON[s.kind] ?? "·"}
                </span>
                <span className="span-agent">{s.agent}</span>
                <span className="span-name">{s.name}</span>
                {s.status !== "ok" && <span className={`tag ${s.status === "error" || s.status === "denied" ? "tag-bad" : "tag-warn"}`}>{s.status}</span>}
                <span className="span-meta">
                  {s.model ? `${s.model} · ${s.input_tokens}+${s.output_tokens} tok · ` : ""}
                  {s.latency_ms.toFixed(0)} ms
                </span>
              </summary>
              {s.input && (
                <>
                  <h4>input</h4>
                  <pre>{s.input}</pre>
                </>
              )}
              {s.output && (
                <>
                  <h4>output</h4>
                  <pre>{s.output}</pre>
                </>
              )}
            </details>
          </li>
        ))}
      </ol>
    </>
  );
}
