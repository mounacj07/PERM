"""
PERM - collect_dataset_v2.py
============================
Collects randomized dataset v2 across 30 seeds (0 to 29) of 'random_mix'
profile with capacity FIXED at 8 workers.
Outputs:
  - data/runs_v2/random_mix_seed<k>.csv (per-second telemetry)
  - data/runs_v2/manifest.json (and data/manifest_v2.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import burst_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_dataset_v2")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs_v2")
MANIFEST_PATH = os.path.join(RUNS_DIR, "manifest.json")
MANIFEST_ROOT_PATH = os.path.join(DATA_DIR, "manifest_v2.json")

COLLECTOR_URL = "http://127.0.0.1:5000"
PROFILES = ["random_mix"]
DEFAULT_SEEDS = list(range(30))

CSV_HEADER = [
    "run_id",
    "profile",
    "seed",
    "burst_start_sec",
    "timestamp",
    "elapsed_sec",
    "burst_start_rel_sec",
    "active_workers",
    "queue_depth",
    "in_flight",
    "queue_wait_ms",
    "arrivals_count",
    "ok_count",
    "rejected_count",
    "timeout_count",
    "rate_req_per_sec",
    "target_rate",
    "cpu_percent",
    "worker_cpu_percent",
    "p95_latency_ms",
    "max_latency_ms",
    "mean_latency_ms",
    "median_latency_ms",
    "p90_latency_ms",
    "p99_latency_ms",
]


def _kill_proc(proc: Optional[subprocess.Popen]):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/metrics/latest", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def run_dataset_collection(seeds: List[int] = DEFAULT_SEEDS):
    os.makedirs(RUNS_DIR, exist_ok=True)
    all_runs = []
    for s in seeds:
        all_runs.append({
            "profile": "random_mix",
            "seed": s,
            "duration": 150.0,
            "csv_path": os.path.join(RUNS_DIR, f"random_mix_seed{s}.csv"),
            "completed": False,
        })

    pending_runs = []
    for r in all_runs:
        if os.path.exists(r["csv_path"]) and os.path.getsize(r["csv_path"]) > 500:
            r["completed"] = True
        else:
            pending_runs.append(r)

    log.info("Total planned v2 runs: %d", len(all_runs))
    log.info("Already completed:     %d", len(all_runs) - len(pending_runs))
    log.info("Pending runs to run:   %d", len(pending_runs))

    if not pending_runs:
        log.info("All v2 runs are already collected! Updating manifest and exiting.")
        update_manifest(all_runs)
        return

    collector_proc = None
    try:
        # Check if collector is already running
        is_running = False
        try:
            r = requests.get(f"{COLLECTOR_URL}/metrics/latest", timeout=1.0)
            if r.status_code == 200:
                is_running = True
        except Exception:
            pass

        if not is_running:
            log.info("Starting telemetry collector with 8 workers...")
            env = os.environ.copy()
            env["INITIAL_WORKERS"] = "8"
            env["MIN_WORKERS"] = "8"
            env["MAX_WORKERS"] = "8"
            env["QUEUE_CAPACITY"] = "20"
            collector_proc = subprocess.Popen(
                [sys.executable, "-u", "telemetry_collector.py"],
                cwd=PROJECT_ROOT,
                env=env,
            )
            if not _wait_for_server(COLLECTOR_URL, timeout=15.0):
                raise RuntimeError("Telemetry collector failed to start within 15 seconds")

        time.sleep(2.0)

        for idx, run_info in enumerate(pending_runs, 1):
            seed = run_info["seed"]
            csv_path = run_info["csv_path"]
            run_id = f"random_mix_seed{seed}"

            log.info(
                "[%d/%d] Starting v2 run %s (dur=%.0fs)...",
                idx,
                len(pending_runs),
                run_id,
                run_info["duration"],
            )

            # Ensure capacity is strictly 8
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=3.0)

            # Generate schedule
            offsets, burst_start_offset, total_dur = burst_test.generate_schedule("random_mix", seed)
            run_info["duration"] = total_dur

            collected_rows: List[Dict] = []
            stop_sampling = threading.Event()
            t0_mono = time.monotonic()
            burst_start_mono = t0_mono + burst_start_offset
            burst_start_utc = datetime.now(timezone.utc)

            def _sample_telemetry():
                last_seen_ts = None
                while not stop_sampling.is_set():
                    time.sleep(0.5)
                    try:
                        r = requests.get(f"{COLLECTOR_URL}/metrics/latest", timeout=1.5)
                        if r.status_code != 200:
                            continue
                        m = r.json()
                    except Exception:
                        continue

                    ts = m.get("timestamp")
                    if ts and ts != last_seen_ts:
                        last_seen_ts = ts
                        now_mono = time.monotonic()
                        elapsed = round(now_mono - t0_mono, 2)
                        rel_burst = round(now_mono - burst_start_mono, 2)

                        row = {
                            "run_id": run_id,
                            "profile": "random_mix",
                            "seed": seed,
                            "burst_start_sec": round(burst_start_offset, 2),
                            "timestamp": ts,
                            "elapsed_sec": elapsed,
                            "burst_start_rel_sec": rel_burst,
                            "active_workers": m.get("active_workers", 8),
                            "queue_depth": m.get("queue_depth", 0),
                            "in_flight": m.get("in_flight", 0),
                            "queue_wait_ms": m.get("queue_wait_ms", 0.0),
                            "arrivals_count": m.get("arrivals_count", 0),
                            "ok_count": m.get("ok_count", 0),
                            "rejected_count": m.get("rejected_count", 0),
                            "timeout_count": m.get("timeout_count", 0),
                            "rate_req_per_sec": m.get("rate_req_per_sec", 0.0),
                            "target_rate": m.get("target_rate", 0.0),
                            "cpu_percent": m.get("cpu_percent", 0.0),
                            "worker_cpu_percent": m.get("worker_cpu_percent", 0.0),
                            "p95_latency_ms": m.get("p95_latency_ms", 0.0),
                            "max_latency_ms": m.get("max_latency_ms", 0.0),
                            "mean_latency_ms": m.get("mean_latency_ms", 0.0),
                            "median_latency_ms": m.get("median_latency_ms", 0.0),
                            "p90_latency_ms": m.get("p90_latency_ms", 0.0),
                            "p99_latency_ms": m.get("p99_latency_ms", 0.0),
                        }
                        collected_rows.append(row)

            sampler_th = threading.Thread(target=_sample_telemetry, daemon=True)
            sampler_th.start()

            burst_start_event = threading.Event()

            def _on_burst(utc_dt: datetime, mono_t: float):
                burst_start_event.set()

            traffic_summary = burst_test.run_burst_load(
                target_url=f"{COLLECTOR_URL}/api/process",
                load=7,
                profile="random_mix",
                seed=seed,
                burst_start_cb=_on_burst,
            )

            time.sleep(2.0)
            stop_sampling.set()
            sampler_th.join(timeout=3.0)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                for row in collected_rows:
                    writer.writerow(row)

            log.info(
                "Completed v2 run %s: wrote %d rows to %s (traffic ok=%d, rej=%d)",
                run_id,
                len(collected_rows),
                csv_path,
                traffic_summary.get("ok_count", 0),
                traffic_summary.get("rejected_count", 0),
            )

            run_info["completed"] = True
            update_manifest(all_runs)
            time.sleep(3.0)

    finally:
        _kill_proc(collector_proc)
        log.info("Dataset v2 collection session finished.")


def update_manifest(all_runs: List[Dict]):
    manifest_records = []
    for r in all_runs:
        csv_path = r["csv_path"]
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                total_arr = sum(int(row.get("arrivals_count", 0)) for row in rows)
                total_ok = sum(int(row.get("ok_count", 0)) for row in rows)
                total_rej = sum(int(row.get("rejected_count", 0)) for row in rows)
                total_to = sum(int(row.get("timeout_count", 0)) for row in rows)
                failed_requests = total_rej + total_to

                max_queue_depth = max(
                    (int(row.get("queue_depth", 0)) for row in rows), default=0
                )

                nonworker_vals = []
                for row in rows:
                    sys_cpu = float(row.get("cpu_percent", 0.0))
                    wk_cpu = float(row.get("worker_cpu_percent", 0.0))
                    nonworker_vals.append(max(0.0, sys_cpu - wk_cpu))
                mean_nonworker_cpu = (
                    round(sum(nonworker_vals) / len(nonworker_vals), 2)
                    if nonworker_vals else 0.0
                )
                sorted_nwk = sorted(nonworker_vals)
                p95_idx = int(len(sorted_nwk) * 0.95)
                p95_nonworker_cpu = (
                    round(sorted_nwk[min(p95_idx, len(sorted_nwk) - 1)], 2)
                    if sorted_nwk else 0.0
                )

                manifest_records.append({
                    "run_id": f"random_mix_seed{r['seed']}",
                    "profile": "random_mix",
                    "seed": r["seed"],
                    "duration_sec": r["duration"],
                    "csv_file": os.path.basename(csv_path),
                    "rows_logged": len(rows),
                    "total_arrivals": total_arr,
                    "total_ok": total_ok,
                    "failed_requests": failed_requests,
                    "max_queue_depth": max_queue_depth,
                    "mean_nonworker_cpu_pct": mean_nonworker_cpu,
                    "p95_nonworker_cpu_pct": p95_nonworker_cpu,
                    "contamination_warning": mean_nonworker_cpu > 10.0,
                    "completed": True,
                })
            except Exception:
                pass
        else:
            manifest_records.append({
                "run_id": f"random_mix_seed{r['seed']}",
                "profile": "random_mix",
                "seed": r["seed"],
                "duration_sec": r["duration"],
                "csv_file": os.path.basename(csv_path),
                "rows_logged": 0,
                "completed": False,
            })

    manifest_obj = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_capacity_workers": 8,
        "profile": "random_mix",
        "seeds": [r["seed"] for r in all_runs],
        "total_runs": len(all_runs),
        "completed_runs": sum(1 for m in manifest_records if m["completed"]),
        "runs": manifest_records,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)
    with open(MANIFEST_ROOT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect PERM dataset v2 (random_mix, 30 seeds)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds to run")
    args = parser.parse_args()
    run_dataset_collection(seeds=args.seeds)
