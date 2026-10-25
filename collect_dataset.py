"""
PERM - collect_dataset.py
=========================
Collects a benchmark dataset across all traffic profiles with capacity FIXED at 8 workers.
Because 8 workers do not saturate under any profile (capacity ~61.8 req/s vs peak 40-45 req/s),
these runs show true uncontaminated demand signals.

Profiles (5 seeds each: 42, 43, 44, 45, 46):
  1. step:         120s (burst at 30s)
  2. ramp:         135s (ramp start at 30s)
  3. spike:         75s (spike at 30s)
  4. double_burst: 130s (bursts at 20s, 80s)
  5. sawtooth:     120s (ramps up/down from 20s)
  6. random_walk:  120s (seeded rate changes)

Total: 30 runs, ~3,500 seconds (~58.3 minutes).
Resumes automatically if stopped (skips existing valid run CSVs).
Outputs:
  - data/runs/<profile>_seed<k>.csv (per-second telemetry with run metadata)
  - data/manifest.json (complete manifest of all runs)
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
log = logging.getLogger("collect_dataset")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

COLLECTOR_URL = "http://127.0.0.1:5000"
PROFILES = ["step", "ramp", "spike", "double_burst", "sawtooth", "random_walk"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

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
    "arrivals_count",
    "ok_count",
    "rejected_count",
    "timeout_count",
    "p95_latency_ms",
    "ok_p95_latency_ms",
    "cpu_percent",
    "worker_cpu_percent",
    "busy_workers",
    "utilization",
    "worker_cpu_per_active_pct",
    "queue_wait_ms_avg",
    "cpu_freq_mhz",
]


def _wait_healthy(url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _kill_proc(proc: Optional[subprocess.Popen]):
    if proc is None:
        return
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        parent.kill()
    except Exception:
        if proc.poll() is None:
            proc.kill()


def get_profile_duration(profile: str) -> float:
    _, _, dur = burst_test.generate_schedule(profile, 42)
    return dur


def is_run_completed(csv_path: str, expected_dur: float) -> bool:
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Check that rows cover at least 90% of expected duration
        if len(rows) >= int(expected_dur * 0.9):
            return True
    except Exception:
        pass
    return False


def run_dataset_collection(
    profiles: List[str] = PROFILES,
    seeds: List[int] = DEFAULT_SEEDS,
    load: int = 7,
):
    os.makedirs(RUNS_DIR, exist_ok=True)

    # 1. Calculate remaining and total expected runtime
    all_runs = []
    total_expected_sec = 0.0
    pending_expected_sec = 0.0

    for prof in profiles:
        dur = get_profile_duration(prof)
        for s in seeds:
            csv_file = os.path.join(RUNS_DIR, f"{prof}_seed{s}.csv")
            completed = is_run_completed(csv_file, dur)
            all_runs.append({
                "profile": prof,
                "seed": s,
                "duration": dur,
                "csv_path": csv_file,
                "completed": completed,
            })
            total_expected_sec += dur
            if not completed:
                pending_expected_sec += dur

    pending_runs = [r for r in all_runs if not r["completed"]]

    print("=" * 70)
    print("PERM DATASET COLLECTION (Capacity FIXED at 8 Workers)")
    print("=" * 70)
    print(f"Total planned runs:    {len(all_runs)} (6 profiles x {len(seeds)} seeds)")
    print(f"Already completed:     {len(all_runs) - len(pending_runs)}")
    print(f"Pending runs to run:   {len(pending_runs)}")
    print(f"Total dataset runtime: {total_expected_sec:.0f}s ({total_expected_sec / 60.0:.1f} mins)")
    print(f"Expected runtime now:  {pending_expected_sec:.0f}s ({pending_expected_sec / 60.0:.1f} mins)")
    print("=" * 70)

    if not pending_runs:
        log.info("All runs already completed. Updating manifest.json...")
        update_manifest(all_runs)
        return

    # 2. Launch telemetry collector with FIXED 8 workers
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "2.0"

    log.info("Starting telemetry collector with 8 workers...")
    collector_proc = subprocess.Popen(
        [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "8"],
        env=env,
        cwd=PROJECT_ROOT,
    )

    try:
        if not _wait_healthy(f"{COLLECTOR_URL}/health", timeout=30.0):
            raise RuntimeError("Telemetry collector failed to start on :5000")

        # Guarantee capacity fixed at 8
        requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=5.0)
        time.sleep(2.0)

        for idx, run_info in enumerate(pending_runs, 1):
            prof = run_info["profile"]
            seed = run_info["seed"]
            csv_path = run_info["csv_path"]
            run_id = f"{prof}_seed{seed}"

            log.info(
                "[%d/%d] Starting run %s (profile=%s, seed=%d, dur=%.0fs)...",
                idx,
                len(pending_runs),
                run_id,
                prof,
                seed,
                run_info["duration"],
            )

            # Ensure capacity is strictly 8
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=3.0)

            # Generate schedule
            offsets, burst_start_offset, total_dur = burst_test.generate_schedule(prof, seed)

            collected_rows: List[Dict] = []
            stop_sampling = threading.Event()
            t0_mono = time.monotonic()
            burst_start_mono = t0_mono + burst_start_offset
            burst_start_utc = datetime.now(timezone.utc)

            # Sampler thread
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
                            "profile": prof,
                            "seed": seed,
                            "burst_start_sec": round(burst_start_offset, 2),
                            "timestamp": ts,
                            "elapsed_sec": elapsed,
                            "burst_start_rel_sec": rel_burst,
                            "active_workers": m.get("active_workers", 8),
                            "queue_depth": m.get("queue_depth", 0),
                            "in_flight": m.get("in_flight", 0),
                            "arrivals_count": m.get("arrivals_count", 0),
                            "ok_count": m.get("ok_count", 0),
                            "rejected_count": m.get("rejected_count", 0),
                            "timeout_count": m.get("timeout_count", 0),
                            "p95_latency_ms": m.get("p95_latency_ms", 0.0),
                            "ok_p95_latency_ms": m.get("ok_p95_latency_ms", 0.0),
                            "cpu_percent": m.get("cpu_percent", 0.0),
                            "worker_cpu_percent": m.get("worker_cpu_percent", 0.0),
                            "busy_workers": m.get("busy_workers", 0),
                            "utilization": m.get("utilization", 0.0),
                            "worker_cpu_per_active_pct": m.get("worker_cpu_per_active_pct", 0.0),
                            "queue_wait_ms_avg": m.get("queue_wait_ms_avg", 0.0),
                            "cpu_freq_mhz": m.get("cpu_freq_mhz", 0.0),
                        }
                        collected_rows.append(row)

            sampler_th = threading.Thread(target=_sample_telemetry, daemon=True)
            sampler_th.start()

            # Execute traffic
            traffic_summary = burst_test.run_burst(
                target_url=f"{COLLECTOR_URL}/api/process",
                profile=prof,
                seed=seed,
                load=load,
                timeout_sec=3.0,
            )

            # Allow trailing samples to register
            time.sleep(2.0)
            stop_sampling.set()
            sampler_th.join(timeout=3.0)

            # Write run CSV
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                for r in collected_rows:
                    writer.writerow(r)

            log.info(
                "Completed run %s: wrote %d rows to %s (traffic ok=%d, rej=%d)",
                run_id,
                len(collected_rows),
                csv_path,
                traffic_summary.get("ok_count", 0),
                traffic_summary.get("rejected_count", 0),
            )

            run_info["completed"] = True
            update_manifest(all_runs)

            # Brief calm pause between runs
            time.sleep(3.0)

    finally:
        _kill_proc(collector_proc)
        log.info("Dataset collection session finished. Telemetry collector stopped.")


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
                manifest_records.append({
                    "run_id": f"{r['profile']}_seed{r['seed']}",
                    "profile": r["profile"],
                    "seed": r["seed"],
                    "duration_sec": r["duration"],
                    "csv_file": os.path.basename(csv_path),
                    "rows_logged": len(rows),
                    "total_arrivals": total_arr,
                    "total_ok": total_ok,
                    "completed": True,
                })
            except Exception:
                pass
        else:
            manifest_records.append({
                "run_id": f"{r['profile']}_seed{r['seed']}",
                "profile": r["profile"],
                "seed": r["seed"],
                "duration_sec": r["duration"],
                "csv_file": os.path.basename(csv_path),
                "rows_logged": 0,
                "completed": False,
            })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_capacity_workers": 8,
            "profiles": PROFILES,
            "seeds": DEFAULT_SEEDS,
            "total_runs": len(all_runs),
            "completed_runs": sum(1 for m in manifest_records if m["completed"]),
            "runs": manifest_records,
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect PERM dataset with fixed 8 workers")
    parser.add_argument("--profiles", nargs="+", default=PROFILES, help="Subset of profiles to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seeds to run")
    args = parser.parse_args()
    run_dataset_collection(profiles=args.profiles, seeds=args.seeds)
