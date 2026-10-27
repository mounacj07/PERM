"""
diag/generate_summary_table.py
==============================
Parses all 5 trace CSVs in diag/ and compiles the definitive Decision Table
with measured empirical data.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List

DIAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES = ["flat_4", "step_late", "ramp_late", "step", "ramp"]

TRACE_HEADER = [
    "t",
    "arrivals",
    "smoothed_arrivals",
    "forecast_raw",
    "forecast_upper",
    "horizon",
    "reactive_target",
    "predictive_target",
    "chosen_target",
    "workers",
    "util",
    "approval_mode",
]


def load_trace(profile: str) -> List[Dict[str, Any]]:
    path = os.path.join(DIAG_DIR, f"controller_trace_{profile}.csv")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_lines = [line for line in reader if line]

    if not all_lines:
        return []

    if all_lines[0][0] == "t" or "arrivals" in all_lines[0]:
        header = all_lines[0]
        data_rows = all_lines[1:]
    else:
        header = TRACE_HEADER
        data_rows = all_lines

    rows = []
    for r in data_rows:
        if len(r) >= len(header):
            rows.append(dict(zip(header, r[:len(header)])))
    return rows


def analyze_all_traces():
    table_rows = []

    for prof in PROFILES:
        rows = load_trace(prof)
        if not rows:
            print(f"Profile {prof}: trace missing")
            continue

        first_scale_t = None
        first_scale_w = None
        load_change_t = None
        scaled_above_2 = False

        workers_timeline = []
        prev_w = None

        forecast_at_warmup = None
        upper_at_warmup = None
        pred_target_at_warmup = None

        for r in rows:
            try:
                t_val = float(r["t"])
                arr_val = float(r["arrivals"])
                w_val = int(r["workers"])
                fc_raw = float(r.get("forecast_raw", 0.0))
                fc_upper = float(r.get("forecast_upper", 0.0))
                pred_t = int(r.get("predictive_target", 0))
            except (ValueError, KeyError):
                continue

            # Capture forecast at t ~ 10s (first step after 10s warmup)
            if forecast_at_warmup is None and t_val >= 9.0:
                forecast_at_warmup = fc_raw
                upper_at_warmup = fc_upper
                pred_target_at_warmup = pred_t

            # Detect load change (arrivals >= 10.0 req/s to filter Poisson noise)
            if load_change_t is None and arr_val >= 10.0:
                load_change_t = t_val

            if w_val > 2 and not scaled_above_2:
                scaled_above_2 = True
                first_scale_t = t_val
                first_scale_w = w_val

            if w_val != prev_w:
                workers_timeline.append((t_val, w_val))
                prev_w = w_val

        scaled_before_load = False
        if first_scale_t is not None:
            if load_change_t is None:
                scaled_before_load = True
            else:
                scaled_before_load = (first_scale_t < load_change_t)

        item = {
            "profile": prof,
            "total_samples": len(rows),
            "first_scale_up_t": first_scale_t,
            "first_scale_up_workers": first_scale_w,
            "forecast_at_warmup": forecast_at_warmup,
            "upper_at_warmup": upper_at_warmup,
            "pred_target_at_warmup": pred_target_at_warmup,
            "load_change_t": load_change_t,
            "scaled_before_load": scaled_before_load,
            "workers_timeline": workers_timeline,
        }
        table_rows.append(item)

    print("\n" + "=" * 90)
    print("COMPREHENSIVE DECISION TABLE (FROM MEASURED TRACES)")
    print("=" * 90)
    print("Profile    | Scaled >2? | 1st Scale-Up (s) | Load Change (s) | Scaled Before Load? | Workers at t=15s | Workers at t=30s")
    print("-----------+------------+------------------+-----------------+---------------------+------------------+------------------")
    for r in table_rows:
        prof = r["profile"]
        scaled = "YES" if r["first_scale_up_t"] is not None else "NO"
        t_sc = f"{r['first_scale_up_t']:.2f}s (to {r['first_scale_up_workers']}w)" if r["first_scale_up_t"] is not None else "N/A"
        t_ld = f"{r['load_change_t']:.2f}s" if r["load_change_t"] is not None else "Never (flat)"
        before = "YES" if r["scaled_before_load"] else "NO"

        # find workers at t=15s and t=30s
        w_15 = "?"
        w_30 = "?"
        for t, w in r["workers_timeline"]:
            if t <= 15.0:
                w_15 = str(w)
            if t <= 30.0:
                w_30 = str(w)

        print(f"{prof:10s} | {scaled:10s} | {t_sc:16s} | {t_ld:15s} | {before:19s} | {w_15:16s} | {w_30:16s}")
    print("=" * 90)

    # Save to JSON
    with open(os.path.join(DIAG_DIR, "decision_table_full.json"), "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)


if __name__ == "__main__":
    analyze_all_traces()
