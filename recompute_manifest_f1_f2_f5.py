"""
recompute_manifest_f1_f2_f5.py
==============================
Implements F1, F2, F5 updates across data/runs_locust/:
- F1: achieved rate = total arrivals / dur (keeps legacy_rate_bias_pct).
      extracts scheduled count, locust sent count, server arrivals, ok + failed.
- F2: normalises all per-process CPU to % of machine (dividing core % by 16).
      recomputes thread equivalents and other_cpu = sys - workers - flask - locust.
- F5: computes detrended Fano factor on residuals (observed - target per second).
"""

import os
import glob
import json
import csv
import numpy as np
import burst_test

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
MANIFEST_PATH = os.path.join(RUNS_DIR, "manifest.json")
NUM_MACHINE_THREADS = 16


def get_target_rate_curve(profile: str, seed: int):
    """Returns a function lambda_t(t) giving the target arrival rate at second t."""
    # Reconstruct stages from burst_test
    # We can inspect the schedule generator
    rng = np.random.RandomState(seed)
    # Get schedule and stages
    _, burst_offset, total_dur = burst_test.generate_schedule(profile, seed)

    # We can also generate the 1-second target counts by counting scheduled offsets in each [t, t+1)
    offsets, _, _ = burst_test.generate_schedule(profile, seed)
    return offsets, total_dur


def compute_target_counts_per_sec(offsets, total_dur):
    max_sec = int(np.ceil(total_dur)) + 2
    counts = [0] * max_sec
    for off in offsets:
        sec = int(np.floor(off))
        if 0 <= sec < max_sec:
            counts[sec] += 1
    return counts


def process_all_runs():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_runs = manifest.get("runs", [])
    updated_runs = []
    per_run_comparison_table = []

    for r in all_runs:
        run_id = r["run_id"]
        csv_file = os.path.join(RUNS_DIR, r.get("csv_file", f"{run_id}.csv"))
        prof = r["profile"]
        s = r["seed"]
        dur = r["duration_sec"]

        if not os.path.exists(csv_file):
            # Not completed yet or in _superseded
            updated_runs.append(r)
            continue

        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            updated_runs.append(r)
            continue

        arrivals = [int(row.get("arrivals_count", 0)) for row in rows]
        total_arr = sum(arrivals)
        total_ok = sum(int(row.get("ok_count", 0)) for row in rows)
        total_rej = sum(int(row.get("rejected_count", 0)) for row in rows)
        total_to = sum(int(row.get("timeout_count", 0)) for row in rows)
        failed = total_rej + total_to
        max_q = max((int(row.get("queue_depth", 0)) for row in rows), default=0)

        # F1. Target schedule & Locust stats
        offsets, _ = get_target_rate_curve(prof, s)
        scheduled_count = len(offsets)
        mean_target_rate = round(scheduled_count / dur, 2)

        # Legacy vs Corrected bias
        legacy_achieved_rate = round(total_arr / max(1.0, len(rows)), 2)
        legacy_bias_pct = round(((legacy_achieved_rate - mean_target_rate) / mean_target_rate) * 100.0, 2)

        # Corrected: total arrivals / PROFILE duration
        corrected_achieved_rate = round(total_arr / dur, 2)
        corrected_bias_pct = round(((corrected_achieved_rate - mean_target_rate) / mean_target_rate) * 100.0, 2)

        # Read Locust stats CSV
        locust_stats_file = os.path.join(RUNS_DIR, f"{run_id}_locust_stats_stats.csv")
        locust_sent_count = None
        if os.path.exists(locust_stats_file):
            try:
                with open(locust_stats_file, "r", encoding="utf-8") as lsf:
                    lreader = list(csv.DictReader(lsf))
                    for lrow in lreader:
                        if lrow.get("Name") == "Aggregated" or lrow.get("Type") == "POST":
                            locust_sent_count = int(lrow.get("Request Count", 0))
                            break
            except Exception:
                pass

        # Step lag
        step_lag = None
        if prof == "step":
            for row in rows:
                el = float(row.get("elapsed_sec", 0.0))
                if el >= 30.0 and int(row.get("arrivals_count", 0)) >= 36:
                    step_lag = round(el - 30.0, 2)
                    break

        # F5. Detrended Fano factor
        target_counts_sec = compute_target_counts_per_sec(offsets, dur)
        residuals = []
        target_vals = []
        for row in rows:
            el = int(np.floor(float(row.get("elapsed_sec", 0.0))))
            arr = int(row.get("arrivals_count", 0))
            if 0 <= el < len(target_counts_sec):
                tgt = target_counts_sec[el]
                residuals.append(arr - tgt)
                target_vals.append(tgt)

        detrended_var = float(np.var(residuals)) if residuals else 0.0
        mean_target_val = float(np.mean(target_vals)) if target_vals else 1.0
        detrended_fano = round(detrended_var / mean_target_val, 2) if mean_target_val > 0 else 0.0

        # Raw calm and burst Fano (legacy)
        calm_arr = [int(row["arrivals_count"]) for row in rows if 5.0 <= float(row.get("elapsed_sec", 0.0)) < 25.0]
        burst_arr = [int(row["arrivals_count"]) for row in rows if 35.0 <= float(row.get("elapsed_sec", 0.0)) < 85.0]
        calm_fano = round(float(np.var(calm_arr) / np.mean(calm_arr)), 2) if calm_arr and np.mean(calm_arr) > 0 else 0.0
        burst_fano = round(float(np.var(burst_arr) / np.mean(burst_arr)), 2) if burst_arr and np.mean(burst_arr) > 0 else 0.0

        # F2. CPU normalization (all to % of machine)
        # raw core % from psutil.Process.cpu_percent()
        flask_raw = [float(row.get("flask_cpu_pct", 0.0)) for row in rows]
        locust_raw = [float(row.get("locust_cpu_pct", 0.0)) for row in rows]
        # worker_cpu_percent was already normalized by 16 in telemetry_collector.py
        workers_mach = [float(row.get("worker_cpu_percent", 0.0)) for row in rows]
        sys_mach = [float(row.get("cpu_percent", 0.0)) for row in rows]

        # Normalized to % of machine
        flask_mach = [c / float(NUM_MACHINE_THREADS) for c in flask_raw]
        locust_mach = [c / float(NUM_MACHINE_THREADS) for c in locust_raw]
        other_mach = [max(0.0, s - w - f - l) for s, w, f, l in zip(sys_mach, workers_mach, flask_mach, locust_mach)]
        nonworker_mach = [max(0.0, s - w) for s, w in zip(sys_mach, workers_mach)]

        # Means as % of machine
        flask_mach_mean = round(float(np.mean(flask_mach)), 2) if flask_mach else 0.0
        locust_mach_mean = round(float(np.mean(locust_mach)), 2) if locust_mach else 0.0
        workers_mach_mean = round(float(np.mean(workers_mach)), 2) if workers_mach else 0.0
        other_mach_mean = round(float(np.mean(other_mach)), 2) if other_mach else 0.0

        # Thread equivalents
        flask_threads = round(float(np.mean(flask_raw)) / 100.0, 2) if flask_raw else 0.0
        locust_threads = round(float(np.mean(locust_raw)) / 100.0, 2) if locust_raw else 0.0
        workers_threads = round(workers_mach_mean / 100.0 * NUM_MACHINE_THREADS, 2)
        other_threads = round(other_mach_mean / 100.0 * NUM_MACHINE_THREADS, 2)

        sorted_nw = sorted(nonworker_mach)
        p95_idx = int(len(sorted_nw) * 0.95)
        nonworker_p95 = round(sorted_nw[min(p95_idx, len(sorted_nw) - 1)], 2) if sorted_nw else 0.0

        flagged = (other_mach_mean > 10.0) or (nonworker_p95 > 25.0) or (failed > 0)

        record = {
            "run_id": run_id,
            "generator": "locust",
            "profile": prof,
            "seed": s,
            "duration_sec": dur,
            "csv_file": os.path.basename(csv_file),
            "rows_logged": len(rows),
            "target_rate_mean_rps": mean_target_rate,
            "legacy_achieved_rate_mean_rps": legacy_achieved_rate,
            "legacy_rate_bias_pct": legacy_bias_pct,
            "achieved_rate_mean_rps": corrected_achieved_rate,
            "rate_bias_pct": corrected_bias_pct,
            "step_lag_sec": step_lag,
            "detrended_fano_var_mean": detrended_fano,
            "calm_fano_var_mean": calm_fano,
            "burst_fano_var_mean": burst_fano,
            "scheduled_count": scheduled_count,
            "locust_sent_count": locust_sent_count if locust_sent_count is not None else total_arr,
            "total_arrivals": total_arr,
            "total_ok": total_ok,
            "failed_requests": failed,
            "max_queue_depth": max_q,
            "cpu_percent": {
                "flask": flask_mach_mean,
                "workers": workers_mach_mean,
                "locust": locust_mach_mean,
                "other": other_mach_mean,
            },
            "cpu_thread_equivalents": {
                "flask": flask_threads,
                "workers": workers_threads,
                "locust": locust_threads,
                "other": other_threads,
            },
            "nonworker_cpu_p95": nonworker_p95,
            "contamination_warning": flagged,
            "completed": True,
        }
        updated_runs.append(record)

        per_run_comparison_table.append({
            "run_id": run_id,
            "profile": prof,
            "seed": s,
            "scheduled": scheduled_count,
            "locust_sent": locust_sent_count if locust_sent_count is not None else total_arr,
            "server_arrivals": total_arr,
            "ok_plus_failed": total_ok + failed,
            "diff_sent_vs_arr": (locust_sent_count if locust_sent_count is not None else total_arr) - total_arr,
            "diff_arr_vs_okfail": total_arr - (total_ok + failed),
            "legacy_bias": legacy_bias_pct,
            "corrected_bias": corrected_bias_pct,
            "detrended_fano": detrended_fano,
            "other_cpu_pct": other_mach_mean,
        })

    manifest["runs"] = updated_runs
    manifest["completed_runs"] = len([r for r in updated_runs if r.get("completed")])
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Processed {len(per_run_comparison_table)} completed runs. Saved updated manifest.")
    return per_run_comparison_table


if __name__ == "__main__":
    process_all_runs()
