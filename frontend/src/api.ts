export type Issue = { field: string; code: string; message: string; severity: "error" | "warning" };

export type Span = {
  id: number;
  run_id: string;
  agent: string;
  kind: "node" | "llm" | "tool" | "guardrail" | "decision";
  name: string;
  status: string;
  input: string | null;
  output: string | null;
  model: string | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number;
  at: string;
};

export type Check = { check: string; ok: boolean; ours: unknown; theirs: unknown };

export type Reconciliation = {
  checks: Check[];
  rate_check: {
    billed: number;
    contract_rate?: number;
    difference_pct?: number;
    vehicle_class_t?: number;
    ok: boolean | null;
    note?: string;
    citation?: { id: string; text: string };
  } | null;
  related_documents: number[];
  mismatch_count: number;
  summary: string;
  citations: string[];
};

export type Doc = {
  id: number;
  client_id: string;
  filename: string;
  status: string;
  doc_type: string | null;
  fields: Record<string, unknown> | null;
  confidence: Record<string, number> | null;
  issues: Issue[];
  reconciliation: Reconciliation | null;
  guardrail_flags: { prompt_injection?: string[]; pii_redacted?: Record<string, number> };
  extraction_method: string | null;
  cost_usd: number;
  error: string | null;
  created_at: string;
  trace?: Span[];
  raw_text?: string | null;
  duplicate?: boolean;
};

export type Approval = {
  id: number;
  kind: string;
  status: string;
  to: string;
  cc: string[];
  subject: string;
  body: string;
  reason: string;
  document_id: number | null;
  trip_id: string | null;
  created_at: string;
};

export type Alert = {
  id: number;
  trip_id: string;
  vehicle_no: string;
  kind: string;
  severity: string;
  detail: string;
  actions: { action: string; [k: string]: unknown }[];
  delay_risk: number | null;
  created_at: string;
};

export type Metrics = {
  documents: Record<string, number>;
  extraction_methods: Record<string, number>;
  pending_approvals: number;
  auto_verified_rate: number | null;
  total_cost_usd: number;
  cost_per_document_usd: number | null;
};

export type Hit = { id: string; source: string; text: string; score: number };

let clientId = "acme-steel";
export const setClient = (id: string) => (clientId = id);

function url(path: string, params: Record<string, string> = {}) {
  const qs = new URLSearchParams({ client_id: clientId, ...params });
  return `${path}${path.includes("?") ? "&" : "?"}${qs}`;
}

async function call<T>(path: string, init?: RequestInit, params?: Record<string, string>): Promise<T> {
  const res = await fetch(url(path, params), init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => call<{ llm: string; model: string | null }>("/api/health"),
  metrics: () => call<Metrics>("/api/metrics"),
  documents: () => call<Doc[]>("/api/documents"),
  document: (id: number) => call<Doc>(`/api/documents/${id}`),
  reprocess: (id: number) => call<Doc>(`/api/documents/${id}/reprocess`, { method: "POST" }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return call<Doc>("/api/documents", { method: "POST", body: form });
  },
  seed: () => call<{ documents_queued: number; trips_checked: string[] }>("/api/demo/seed", { method: "POST" }),
  approvals: (status = "pending") => call<Approval[]>("/api/approvals", undefined, { status }),
  decide: (id: number, decision: "approve" | "reject", edits: { subject?: string; body?: string }) =>
    call<Approval>(`/api/approvals/${id}/decision`, json({ decision, ...edits })),
  alerts: () => call<Alert[]>("/api/fleet/alerts"),
  fleetTrace: (runId: string) => call<Span[]>(`/api/fleet/trace/${runId}`),
  search: (q: string) => call<Hit[]>("/api/rate-cards/search", undefined, { q }),
  memory: () => call<Record<string, string>>("/api/memory"),
  setMemory: (key: string, value: string) =>
    call<Record<string, string>>(`/api/memory/${key}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ value }) }),
  events: (id: number) => new EventSource(url(`/api/documents/${id}/events`)),
};

export const money = (v: unknown) =>
  typeof v === "number" ? `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : String(v ?? "—");
