"""Run the golden set end to end and fail (exit 1) if any metric drops below its threshold.

    python -m evals.run_evals              # offline engine (free, deterministic; what CI runs)
    FP_LLM_MODE=claude python -m evals.run_evals   # same suite against Claude (costs money)

Writes evals/results/latest.json and prints a markdown table.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_tmp = tempfile.mkdtemp(prefix="fp-eval-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/eval.db")
logging.disable(logging.INFO)

from app.agents.fleet_monitor import run_fleet_check  # noqa: E402
from app.agents.supervisor import process_document  # noqa: E402
from app.db import init_db  # noqa: E402
from app.guardrails import redact_pii  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.main import UPLOAD_DIR, ingest  # noqa: E402
from app.schemas import Trip  # noqa: E402
from app.tools.ocr import extract_text  # noqa: E402
from app.tools.vision import _DEMO_EVENTS, YardEvent, record_yard_events  # noqa: E402
from app.tracing import Tracer  # noqa: E402
from evals.dataset import build_fleet_scenarios, build_hard_samples, build_shipments  # noqa: E402


def _match(expected, got) -> bool:
    if isinstance(expected, float):
        return isinstance(got, int | float) and abs(expected - got) < 0.01
    return expected == got


def _run(client_id: str, sample, llm) -> tuple[dict, float, Tracer]:
    doc, _ = ingest(client_id, f"{sample.id}.txt", sample.text.encode())
    tracer = Tracer(document_id=doc.id)
    start = time.perf_counter()
    state = process_document(
        doc.id, client_id, doc.filename, extract_text("x.txt", sample.text.encode()), sample.text.encode(), tracer, llm
    )
    return state, (time.perf_counter() - start) * 1000, tracer


def field_scores(sample, state) -> tuple[int, int]:
    fields = (state.get("extraction") or {}).get("fields") or {}
    hits = sum(_match(v, fields.get(k)) for k, v in sample.expected_fields.items())
    return hits, len(sample.expected_fields)


def main() -> int:
    init_db()
    UPLOAD_DIR.mkdir(exist_ok=True)
    llm = get_llm()
    m: dict[str, float] = {}
    latencies, costs = [], []

    # ---- documents in shipment order, so reconciliation sees siblings
    shipments = build_shipments(40)
    cls_ok = n_docs = f_hit = f_tot = 0
    issue_expected = issue_found = clean_docs = false_issue = 0
    inj_expected = inj_found = inj_false = 0
    ship_mismatch_expected = ship_mismatch_found = clean_ships = clean_ship_flagged = 0
    for ship in shipments:
        flagged = False
        for sample in ship.docs:
            state, ms, tracer = _run(ship.client_id, sample, llm)
            latencies.append(ms)
            costs.append(tracer.total_cost)
            n_docs += 1
            cls_ok += state.get("doc_type") == sample.doc_type
            h, t = field_scores(sample, state)
            f_hit, f_tot = f_hit + h, f_tot + t
            codes = {i["code"] for i in (state.get("extraction") or {}).get("issues", []) if i["severity"] == "error"}
            if sample.expected_issue_codes:
                issue_expected += 1
                issue_found += sample.expected_issue_codes <= codes
            else:
                clean_docs += 1
                false_issue += bool(codes)
            injected = bool(state.get("flags", {}).get("prompt_injection"))
            if sample.expect_injection:
                inj_expected += 1
                inj_found += injected
            else:
                inj_false += injected
            flagged |= bool((state.get("reconciliation") or {}).get("mismatch_count"))
        if ship.expect_mismatch:
            ship_mismatch_expected += 1
            ship_mismatch_found += flagged
        else:
            clean_ships += 1
            clean_ship_flagged += flagged

    m["classification_accuracy"] = cls_ok / n_docs
    m["field_accuracy"] = f_hit / f_tot
    m["issue_detection_recall"] = issue_found / max(1, issue_expected)
    m["false_issue_rate"] = false_issue / max(1, clean_docs)
    m["injection_recall"] = inj_found / max(1, inj_expected)
    m["injection_false_positive_rate"] = inj_false / max(1, n_docs - inj_expected)
    m["mismatch_recall"] = ship_mismatch_found / max(1, ship_mismatch_expected)
    m["mismatch_false_positive_rate"] = clean_ship_flagged / max(1, clean_ships)

    # ---- PII: every driver phone on lorry receipts must be redacted before an LLM sees it
    lrs = [d for s in shipments for d in s.docs if "pii_phone" in d.tags]
    m["pii_phone_redaction_rate"] = sum(redact_pii(d.text)[1].get("phone", 0) >= 1 for d in lrs) / len(lrs)

    # ---- hard split: the number that matters is how often a WRONG extraction is marked done
    hard = build_hard_samples(30)
    h_hit = h_tot = silent = reviewed = 0
    for sample in hard:
        state, ms, tracer = _run("eval-hard", sample, llm)
        latencies.append(ms)
        costs.append(tracer.total_cost)
        h, t = field_scores(sample, state)
        h_hit, h_tot = h_hit + h, h_tot + t
        done = state.get("status") == "done"
        reviewed += not done
        silent += done and h < t
    m["hard_field_accuracy"] = h_hit / h_tot
    m["hard_review_rate"] = reviewed / len(hard)
    m["hard_silent_error_rate"] = silent / len(hard)

    # ---- fleet: did the agent pick the right actions?
    correct = 0
    scenarios = build_fleet_scenarios()
    for sc in scenarios:
        _DEMO_EVENTS.clear()
        record_yard_events([YardEvent(**e) for e in sc.yard_events])
        tracer = Tracer()
        out = run_fleet_check(Trip(**sc.trip), tracer, llm)
        costs.append(tracer.total_cost)
        called = any(a["action"] == "request_driver_call" for a in out["actions"])
        notified = out["approval_id"] is not None
        correct += called == sc.expect_driver_call and notified == sc.expect_notify
    m["fleet_action_accuracy"] = correct / len(scenarios)

    m["cost_per_item_usd"] = sum(costs) / len(costs)
    m["p50_latency_ms"] = statistics.median(latencies)
    m["p95_latency_ms"] = statistics.quantiles(latencies, n=20)[-1]

    thresholds = json.loads((HERE / "thresholds.json").read_text())
    failures = []
    print(f"\n## Eval results ({'claude: ' + os.getenv('FP_MODEL', 'claude-opus-5-5') if llm else 'offline engine'})\n")
    print("| metric | value | threshold | |\n|---|---|---|---|")
    for key, value in m.items():
        rule = thresholds.get(key)
        ok = True
        if rule:
            ok = value >= rule["min"] if "min" in rule else value <= rule["max"]
            if not ok:
                failures.append(key)
        bound = f"{'>=' if rule and 'min' in rule else '<='} {rule.get('min', rule.get('max'))}" if rule else ""
        print(f"| {key} | {value:.4f} | {bound} | {'' if ok else 'FAIL'} |")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "latest.json").write_text(
        json.dumps({"engine": "claude" if llm else "offline", "metrics": m, "failures": failures}, indent=2)
    )
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
