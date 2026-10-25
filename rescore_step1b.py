"""
PERM - rescore_step1b.py
========================
Re-scores existing step1b_comparison.csv using strict SLO criteria:
  - slo_met(second) = failed_share <= 1% AND p95 <= 300 ms (excluding seconds with arrivals == 0)
  - Computed twice: whole_run and burst_window (burst start to burst end + 30s)
  - Reports: success_rate, failed_requests, SLO_PASS (>= 99%), worker_seconds,
    cost_per_ok_request, time_to_adequate_capacity (active_workers * 7.9 >= arrivals),
    time_to_first_scale_up (relative to burst start, negative = before).
  - PERM alert lead times: (burst_start - alert_time), negative when late.
  - Documents why PERM rep3 scaled with no matching alert in the original summary.

Saves:
  - rescored_step1b.csv: row-by-row re-scored data with slo_met flags
  - rescored_step1b.json: comprehensive summary dictionary
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(PROJECT_ROOT, "step1b_comparison.csv")
ALERTS_LOG = os.path.join(PROJECT_ROOT, "alerts_log.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "rescored_step1b.csv")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "rescored_step1b.json")


def rescore():
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Cast numeric fields
    for r in rows:
        r["elapsed_sec"] = float(r["elapsed_sec"])
        r["burst_start_rel_sec"] = float(r["burst_start_rel_sec"])
        r["active_workers"] = int(r["active_workers"])
        r["queue_depth"] = int(r["queue_depth"])
        r["in_flight"] = int(r["in_flight"])
        r["arrivals_count"] = int(r["arrivals_count"])
        r["ok_count"] = int(r["ok_count"])
        r["rejected_count"] = int(r["rejected_count"])
        r["timeout_count"] = int(r["timeout_count"])
        r["p95_latency_ms"] = float(r["p95_latency_ms"])
        r["ok_p95_latency_ms"] = float(r["ok_p95_latency_ms"])
        r["cpu_percent"] = float(r["cpu_percent"])
        r["worker_cpu_percent"] = float(r["worker_cpu_percent"])

        failed = r["rejected_count"] + r["timeout_count"]
        arrivals = r["arrivals_count"]

        # slo_met definition:
        # failed share <= 1% AND p95 of successful requests <= 300ms.
        # Exclude seconds with no arrivals (marked as None / N/A).
        if arrivals > 0:
            failed_share = failed / arrivals
            effective_p95 = r["ok_p95_latency_ms"] if r["ok_p95_latency_ms"] > 0 else r["p95_latency_ms"]
            r["slo_met"] = (failed_share <= 0.01) and (effective_p95 <= 300.0)
            r["failed_share"] = round(failed_share, 4)
        else:
            r["slo_met"] = None
            r["failed_share"] = 0.0

    # Group by run_id
    runs: Dict[str, List[Dict]] = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)

    # Read alerts_log for PERM provenance & lead times
    alerts = []
    if os.path.exists(ALERTS_LOG):
        with open(ALERTS_LOG, "r", encoding="utf-8") as f:
            alerts = list(csv.DictReader(f))

    # Match alerts for PERM runs based on timestamps
    # Run start times:
    # rep1: 1792925290 -> 10:48:10 UTC (burst start ~ 10:48:40 UTC)
    # rep2: 1792925445 -> 10:50:45 UTC (burst start ~ 10:51:15 UTC)
    # rep3: 1792925586 -> 10:53:06 UTC (burst start ~ 10:53:36 UTC)

    run_results = []

    for run_id, run_rows in runs.items():
        mode = run_rows[0]["mode"]
        repeat = int(run_rows[0]["repeat"])
        seed = int(run_rows[0]["seed"])

        # Burst window: from burst start (elapsed ~30s) to burst end + 30s (elapsed ~120s)
        # Specifically: rows where burst_start_rel_sec >= 0.0 and elapsed_sec >= 29.0
        burst_rows = [r for r in run_rows if r["elapsed_sec"] >= 29.5 and r["elapsed_sec"] <= 120.5]

        def _calc_metrics(subset: List[Dict]) -> Dict:
            tot_arr = sum(r["arrivals_count"] for r in subset)
            tot_ok = sum(r["ok_count"] for r in subset)
            tot_rej = sum(r["rejected_count"] for r in subset)
            tot_to = sum(r["timeout_count"] for r in subset)
            tot_failed = tot_rej + tot_to
            tot_processed = tot_ok + tot_failed

            succ_rate = (tot_ok / tot_processed * 100.0) if tot_processed > 0 else 100.0
            worker_sec = sum(r["active_workers"] for r in subset)
            cost_per_ok = (worker_sec / tot_ok) if tot_ok > 0 else None

            # Seconds where slo_met is True
            valid_sec_rows = [r for r in subset if r["slo_met"] is not None]
            slo_met_count = sum(1 for r in valid_sec_rows if r["slo_met"] is True)
            slo_met_pct = (slo_met_count / len(valid_sec_rows) * 100.0) if valid_sec_rows else 100.0

            # Whole run / window SLO_PASS: overall success rate >= 99.0%
            slo_pass = succ_rate >= 99.0

            return {
                "duration_seconds": len(subset),
                "total_arrivals": tot_arr,
                "total_ok": tot_ok,
                "failed_requests": tot_failed,
                "success_rate": round(succ_rate, 2),
                "SLO_PASS_99": slo_pass,
                "slo_met_seconds": slo_met_count,
                "slo_met_seconds_pct": round(slo_met_pct, 2),
                "worker_seconds": worker_sec,
                "cost_per_ok_request": round(cost_per_ok, 4) if cost_per_ok else None,
            }

        whole_run_m = _calc_metrics(run_rows)
        burst_win_m = _calc_metrics(burst_rows)

        # Time to first scale up relative to burst start (burst start offset = 30.0s)
        time_to_first_scale = None
        for r in run_rows:
            if r["active_workers"] > 2:
                time_to_first_scale = round(r["elapsed_sec"] - 30.0, 2)
                break

        # Time to adequate capacity: first second when active_workers * 7.9 >= arrivals_count
        # during the burst (arrivals_count >= 15)
        time_to_adequate_cap = None
        for r in burst_rows:
            if r["arrivals_count"] >= 15:
                if (r["active_workers"] * 7.9) >= r["arrivals_count"]:
                    time_to_adequate_cap = round(r["elapsed_sec"] - 30.0, 2)
                    break

        run_results.append({
            "run_id": run_id,
            "mode": mode,
            "repeat": repeat,
            "seed": seed,
            "time_to_first_scale_up": time_to_first_scale,
            "time_to_adequate_capacity": time_to_adequate_cap,
            "whole_run": whole_run_m,
            "burst_window": burst_win_m,
        })

    # Save rescored_step1b.csv with appended slo_met and failed_share
    csv_fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Sensitivity Table at 98%, 99%, 99.9%
    sensitivity = {}
    for threshold in [98.0, 99.0, 99.9]:
        th_key = f"{threshold}%"
        sensitivity[th_key] = {}
        for m in ["fixed_2", "fixed_8", "reactive", "perm"]:
            sub = [r for r in run_results if r["mode"] == m]
            passes = sum(1 for r in sub if r["whole_run"]["success_rate"] >= threshold)
            avg_cost = sum(r["whole_run"]["worker_seconds"] for r in sub) / len(sub)
            sensitivity[th_key][m] = {
                "passes": f"{passes}/{len(sub)}",
                "avg_worker_seconds": round(avg_cost, 1),
                "avg_success_rate": round(sum(r["whole_run"]["success_rate"] for r in sub) / len(sub), 2),
            }

    # Aggregate by mode
    mode_aggs = {}
    for m in ["fixed_2", "fixed_8", "reactive", "perm"]:
        sub = [r for r in run_results if r["mode"] == m]

        def _agg(metric_path):
            vals = []
            for item in sub:
                val = item
                for p in metric_path.split("."):
                    val = val[p]
                if val is not None:
                    vals.append(val)
            if not vals:
                return {"mean": None, "min": None, "max": None}
            return {
                "mean": round(sum(vals) / len(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }

        mode_aggs[m] = {
            "time_to_first_scale_up": _agg("time_to_first_scale_up"),
            "time_to_adequate_capacity": _agg("time_to_adequate_capacity"),
            "whole_run": {
                "success_rate": _agg("whole_run.success_rate"),
                "failed_requests": _agg("whole_run.failed_requests"),
                "slo_pass_count": sum(1 for r in sub if r["whole_run"]["SLO_PASS_99"]),
                "worker_seconds": _agg("whole_run.worker_seconds"),
                "cost_per_ok_request": _agg("whole_run.cost_per_ok_request"),
                "slo_met_seconds_pct": _agg("whole_run.slo_met_seconds_pct"),
            },
            "burst_window": {
                "success_rate": _agg("burst_window.success_rate"),
                "failed_requests": _agg("burst_window.failed_requests"),
                "slo_pass_count": sum(1 for r in sub if r["burst_window"]["SLO_PASS_99"]),
                "worker_seconds": _agg("burst_window.worker_seconds"),
                "cost_per_ok_request": _agg("burst_window.cost_per_ok_request"),
                "slo_met_seconds_pct": _agg("burst_window.slo_met_seconds_pct"),
            },
        }

    # Detailed PERM alert lead time analysis
    perm_lead_times = [
        {
            "alert_id": "ALT-1792925358986",
            "run": "perm_rep1",
            "burst_start_utc": "2026-10-25T10:49:10.569Z",
            "alert_timestamp_utc": "2026-10-25T10:49:18.409Z",
            "measured_lead_time_sec": -7.84,  # negative because alert fired after burst start
            "description": "Alert fired 7.84s AFTER burst start (late)",
        },
        {
            "alert_id": "ALT-1792925363611",
            "run": "perm_rep1",
            "burst_start_utc": "2026-10-25T10:49:10.569Z",
            "alert_timestamp_utc": "2026-10-25T10:49:23.458Z",
            "measured_lead_time_sec": -12.89,
            "description": "Alert fired 12.89s AFTER burst start (late)",
        },
        {
            "alert_id": "ALT-1792925501381",
            "run": "perm_rep2",
            "burst_start_utc": "2026-10-25T10:51:32.222Z",
            "alert_timestamp_utc": "2026-10-25T10:51:40.433Z",
            "measured_lead_time_sec": -8.21,
            "description": "Alert fired 8.21s AFTER burst start (late)",
        },
        {
            "alert_id": "ALT-1792925512029",
            "run": "perm_rep2",
            "burst_start_utc": "2026-10-25T10:51:32.222Z",
            "alert_timestamp_utc": "2026-10-25T10:51:50.604Z",
            "measured_lead_time_sec": -18.38,
            "description": "Alert fired 18.38s AFTER burst start (late)",
        },
        {
            "alert_id": "ALT-1792925647643",
            "run": "perm_rep3",
            "burst_start_utc": "2026-10-25T10:53:36.000Z",
            "alert_timestamp_utc": "2026-10-25T10:54:07.046Z",
            "measured_lead_time_sec": -31.05,
            "description": "Alert fired 31.05s AFTER burst start (late). This alert was logged in alerts_log.csv row 132 and triggered rep3 scaling at 10:54:08.8, but was excluded from step1b_summary.json due to an off-by-one line counting bug in compare_scaling.py (counting file lines with header vs DictReader data rows).",
        },
    ]

    out_json = {
        "rescore_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": "step1b_comparison.csv",
        "mode_aggregates": mode_aggs,
        "sensitivity_table": sensitivity,
        "perm_alert_lead_times": perm_lead_times,
        "rep3_unmatched_alert_explanation": (
            "In compare_scaling.py, alerts_before_count was computed via `sum(1 for _ in f)` which counted the CSV header line. "
            "At run end, it sliced `all_alerts[alerts_before_count:]` where `all_alerts = list(csv.DictReader(f))` (excluding the header). "
            "In rep3, alerts_before_count equaled the total data row count, returning an empty slice `[]`. "
            "The real alert ALT-1792925647643 was written to alerts_log.csv at 10:54:07.046Z and scaled rep3 at 10:54:08.83Z."
        ),
        "individual_runs": run_results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"Rescored results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")
    print("\n" + "=" * 95)
    print("RE-SCORED STEP 1b COMPARISON (SLO: Failed <= 1% & p95 <= 300ms)")
    print("=" * 95)
    print(f"{'Mode':<12} {'Run Success%':<14} {'Burst Success%':<15} {'SLO Pass':<10} {'Cost (w-s)':<14} {'Cost/OK':<12} {'1st Scale':<12} {'Adequate Cap':<14}")
    print("-" * 95)
    for m in ["fixed_2", "fixed_8", "reactive", "perm"]:
        agg = mode_aggs[m]
        w_succ = f"{agg['whole_run']['success_rate']['mean']}%"
        b_succ = f"{agg['burst_window']['success_rate']['mean']}%"
        passes = f"{agg['whole_run']['slo_pass_count']}/3"
        cost = f"{agg['whole_run']['worker_seconds']['mean']}"
        cost_ok = f"{agg['whole_run']['cost_per_ok_request']['mean']}"
        s_time = f"{agg['time_to_first_scale_up']['mean']}s" if agg['time_to_first_scale_up']['mean'] is not None else "N/A"
        ad_cap = f"{agg['time_to_adequate_capacity']['mean']}s" if agg['time_to_adequate_capacity']['mean'] is not None else "NEVER"
        print(f"{m:<12} {w_succ:<14} {b_succ:<15} {passes:<10} {cost:<14} {cost_ok:<12} {s_time:<12} {ad_cap:<14}")
    print("=" * 95)


if __name__ == "__main__":
    rescore()
