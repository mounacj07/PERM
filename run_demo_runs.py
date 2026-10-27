"""
PERM - run_demo_runs.py
=======================
Automated runner for Demo Controller Evaluation Runs:
Runs step (seed 44) and ramp (seed 44) for:
  1. reactive
  2. capacity_aware
  3. predictive
(6 runs total, starting at 2 workers).

Metrics collected per run:
  1. First scale-up time relative to burst/ramp start (negative = before)
  2. Workers during the first 25s (should be 2)
  3. Workers timeline
  4. Failed requests
  5. Worker-seconds
  6. Peak pool utilization (%)
  7. Seconds above 70%

All data is strictly real measured telemetry from the live testbed runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo_runner")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLLECTOR_URL = "http://127.0.0.1:5000"
INFERENCE_URL = "http://127.0.0.1:5001"


def wait_for_url(url: str, timeout: float = 35.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def terminate_proc(proc: Optional[subprocess.Popen], name: str = "proc") -> None:
    if proc is None:
        return
    if proc.poll() is None:
        log.info("Terminating %s (PID %d)...", name, proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3.0)
            except Exception:
                pass


def run_single_demo(mode: str, profile: str = "step", seed: int = 44) -> Dict[str, Any]:
    run_key = f"{mode}_{profile}"
    log.info("=" * 70)
    log.info("STARTING DEMO RUN: MODE = %s | PROFILE = %s | SEED = %d", mode.upper(), profile.upper(), seed)
    log.info("=" * 70)

    collector_proc = None
    inference_proc = None
    controller_proc = None
    locust_proc = None

    run_events_csv = os.path.join(RESULTS_DIR, f"controller_events_{run_key}.csv")
    if os.path.exists(run_events_csv):
        os.remove(run_events_csv)

    locust_csv_prefix = os.path.join(RESULTS_DIR, f"locust_demo_{run_key}")

    telemetry_timeline: List[Dict[str, Any]] = []

    # Profile timing parameters
    # step: 30s calm, 60s burst, 30s calm -> 120s total, run locust 125s
    # ramp: 30s calm, 30s ramp, 45s hold, 30s calm -> 135s total, run locust 140s
    locust_duration = 125 if profile == "step" else 140
    controller_duration = locust_duration + 15

    try:
        # 1. Start Telemetry Collector
        log.info("Starting telemetry collector on port 5000 with 2 workers...")
        collector_proc = subprocess.Popen(
            [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "2"],
            cwd=PROJECT_ROOT,
        )
        if not wait_for_url(f"{COLLECTOR_URL}/health", timeout=15.0):
            raise RuntimeError("Telemetry collector failed to respond on /health within 15s")
        # Ensure initial capacity is 2
        requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=2.0)
        log.info("Telemetry collector ready with 2 workers.")

        # 2. If predictive mode, start Inference Service with PERM_FORECASTER=stagbdt
        if mode == "predictive":
            log.info("Starting inference service with PERM_FORECASTER=stagbdt...")
            inf_env = os.environ.copy()
            inf_env["PERM_FORECASTER"] = "stagbdt"
            inference_proc = subprocess.Popen(
                [
                    sys.executable,
                    "inference_service.py",
                    "--port",
                    "5001",
                    "--collector-url",
                    COLLECTOR_URL,
                ],
                cwd=PROJECT_ROOT,
                env=inf_env,
            )
            if not wait_for_url(f"{INFERENCE_URL}/health", timeout=15.0):
                raise RuntimeError("Inference service failed to respond on /health within 15s")
            log.info("Inference service ready on port 5001.")

        # 3. Start Demo Controller
        log.info("Starting demo controller (mode=%s)...", mode)
        controller_proc = subprocess.Popen(
            [
                sys.executable,
                "demo_controller.py",
                "--mode",
                mode,
                "--initial-workers",
                "2",
                "--collector-url",
                COLLECTOR_URL,
                "--inference-url",
                INFERENCE_URL,
                "--csv",
                run_events_csv,
                "--duration",
                str(controller_duration),
            ],
            cwd=PROJECT_ROOT,
        )
        time.sleep(2.0)  # Controller stabilization

        # 4. Start Locust Profile with Seed 44
        log.info("Launching Locust %s profile (seed %d, duration %ds)...", profile, seed, locust_duration)
        loc_env = os.environ.copy()
        loc_env["PERM_PROFILE"] = profile
        loc_env["PERM_SEED"] = str(seed)

        locust_cmd = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            "locustfile_perm.py",
            "--headless",
            "-u",
            "1",
            "-r",
            "1",
            "-t",
            f"{locust_duration}s",
            "--host",
            COLLECTOR_URL,
            "--csv",
            locust_csv_prefix,
        ]

        t_locust_start = time.monotonic()
        locust_proc = subprocess.Popen(locust_cmd, cwd=PROJECT_ROOT, env=loc_env)

        # Burst/ramp start is exactly offset 30.0s after Locust starts
        burst_start_time = t_locust_start + 30.0
        first_scale_up_time: Optional[float] = None
        first_scale_up_target: Optional[int] = None

        # Sample metrics every 1 second while Locust is running
        raw_util_window: List[float] = []

        while locust_proc.poll() is None:
            t_now = time.monotonic()
            elapsed_from_start = t_now - t_locust_start

            try:
                r = requests.get(f"{COLLECTOR_URL}/metrics/latest", timeout=1.0)
                if r.status_code == 200:
                    data = r.json()
                    worker_cpu = float(data.get("worker_cpu_percent", 0.0))
                    workers = int(data.get("active_workers", 2))
                    raw_util = float(min(100.0, max(0.0, (worker_cpu * 16.0) / max(1, workers))))
                    raw_util_window.append(raw_util)
                    if len(raw_util_window) > 3:
                        raw_util_window.pop(0)
                    smoothed_util = sum(raw_util_window) / len(raw_util_window)

                    sample = {
                        "elapsed_sec": round(elapsed_from_start, 2),
                        "timestamp": data.get("timestamp"),
                        "workers": workers,
                        "worker_cpu_percent": worker_cpu,
                        "raw_pool_util": round(raw_util, 2),
                        "smoothed_pool_util": round(smoothed_util, 2),
                        "arrivals_count": float(data.get("arrivals_count", 0.0)),
                        "queue_depth": int(data.get("queue_depth", 0)),
                        "error_count": int(data.get("error_count", 0)),
                        "in_flight": int(data.get("in_flight", 0)),
                        "avg_latency_ms": float(data.get("avg_latency_ms", 0.0)),
                    }
                    telemetry_timeline.append(sample)

                    # Detect first scale-up (workers > 2)
                    if workers > 2 and first_scale_up_time is None:
                        first_scale_up_time = t_now
                        first_scale_up_target = workers
                        rel_time = round(t_now - burst_start_time, 2)
                        log.info(
                            "FIRST SCALE-UP DETECTED: 2 -> %d workers at elapsed=%.2fs (rel to burst start = %.2fs)",
                            workers,
                            elapsed_from_start,
                            rel_time,
                        )

            except Exception:
                pass

            time.sleep(1.0)

        log.info("Locust process exited with returncode %d", locust_proc.returncode)

    finally:
        # Cleanly terminate all child processes
        terminate_proc(locust_proc, "Locust")
        terminate_proc(controller_proc, "DemoController")
        terminate_proc(inference_proc, "InferenceService")
        terminate_proc(collector_proc, "TelemetryCollector")
        time.sleep(5.0)  # Wait for sockets to close

    # 5. Compute Run Metrics from Real Telemetry
    if not telemetry_timeline:
        raise RuntimeError(f"No telemetry was collected during demo run {run_key}!")

    peak_pool_util = max(s["smoothed_pool_util"] for s in telemetry_timeline)
    seconds_above_70 = sum(1 for s in telemetry_timeline if s["smoothed_pool_util"] > 70.0)

    # First scale-up relative to burst/ramp start: negative = before, positive = after
    if first_scale_up_time is not None:
        rel_scale_time = round(first_scale_up_time - burst_start_time, 2)
    else:
        rel_scale_time = None

    # Check workers during the first 25 seconds
    workers_25s = [s["workers"] for s in telemetry_timeline if s["elapsed_sec"] <= 25.0]
    unique_workers_25s = sorted(list(set(workers_25s))) if workers_25s else [2]
    workers_first_25s_str = ", ".join(map(str, unique_workers_25s)) if len(unique_workers_25s) > 1 else str(unique_workers_25s[0])

    final_util = telemetry_timeline[-1]["smoothed_pool_util"]
    total_failures = telemetry_timeline[-1]["error_count"]

    # Compute worker-seconds: integral of workers over elapsed time steps
    worker_seconds = 0.0
    for i in range(len(telemetry_timeline)):
        dt = (
            (telemetry_timeline[i]["elapsed_sec"] - telemetry_timeline[i - 1]["elapsed_sec"])
            if i > 0
            else 1.0
        )
        worker_seconds += telemetry_timeline[i]["workers"] * max(0.5, min(2.0, dt))
    worker_seconds = round(worker_seconds, 1)

    # Worker timeline intervals
    worker_transitions = []
    current_w = telemetry_timeline[0]["workers"]
    start_t = telemetry_timeline[0]["elapsed_sec"]
    for s in telemetry_timeline:
        if s["workers"] != current_w:
            worker_transitions.append({
                "workers": current_w,
                "start_sec": round(start_t, 1),
                "end_sec": round(s["elapsed_sec"], 1),
                "duration_sec": round(s["elapsed_sec"] - start_t, 1),
            })
            current_w = s["workers"]
            start_t = s["elapsed_sec"]
    worker_transitions.append({
        "workers": current_w,
        "start_sec": round(start_t, 1),
        "end_sec": round(telemetry_timeline[-1]["elapsed_sec"], 1),
        "duration_sec": round(telemetry_timeline[-1]["elapsed_sec"] - start_t, 1),
    })

    # Read controller events from CSV
    events_from_csv = []
    if os.path.exists(run_events_csv):
        with open(run_events_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            events_from_csv = list(reader)

    # Read Locust summary CSV if available
    locust_stats_file = f"{locust_csv_prefix}_stats.csv"
    locust_requests = 0
    locust_failures = 0
    if os.path.exists(locust_stats_file):
        with open(locust_stats_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Name") == "Aggregated":
                    try:
                        locust_requests = int(row.get("Request Count", 0))
                        locust_failures = int(row.get("Failure Count", 0))
                    except Exception:
                        pass

    run_summary = {
        "mode": mode,
        "profile": profile,
        "seed": seed,
        "burst_start_sec": 30.0,
        "first_scale_up_rel_sec": rel_scale_time,
        "first_scale_up_target": first_scale_up_target,
        "workers_first_25s": workers_first_25s_str,
        "peak_pool_utilization_pct": round(peak_pool_util, 2),
        "seconds_above_70": seconds_above_70,
        "final_utilization_pct": round(final_util, 2),
        "failed_requests": max(total_failures, locust_failures),
        "locust_requests": locust_requests,
        "locust_failures": locust_failures,
        "worker_seconds": worker_seconds,
        "num_telemetry_samples": len(telemetry_timeline),
        "workers_over_time": worker_transitions,
        "controller_events": events_from_csv,
    }

    # Save summary and timeline
    summary_path = os.path.join(RESULTS_DIR, f"demo_run_{run_key}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    timeline_path = os.path.join(RESULTS_DIR, f"telemetry_timeline_{run_key}.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_timeline, f, indent=2)

    log.info("Finished demo run [%s]. Summary saved to %s", run_key, summary_path)
    return run_summary


def format_timeline(transitions: List[Dict[str, Any]]) -> str:
    parts = []
    for t in transitions:
        parts.append(f"{t['start_sec']}s-{t['end_sec']}s: {t['workers']}w ({t['duration_sec']}s)")
    return " -> ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Run PERM Demo Controller Honest Runs")
    parser.add_argument(
        "--mode",
        choices=["all", "reactive", "capacity_aware", "predictive"],
        default="all",
        help="Controller mode to run: all, reactive, capacity_aware, predictive",
    )
    parser.add_argument(
        "--profile",
        choices=["both", "step", "ramp"],
        default="both",
        help="Profile to run: both, step, ramp",
    )
    args = parser.parse_args()

    modes = ["reactive", "capacity_aware", "predictive"] if args.mode == "all" else [args.mode]
    profiles = ["step", "ramp"] if args.profile == "both" else [args.profile]

    results: Dict[str, Any] = {}
    comparison_file = os.path.join(RESULTS_DIR, "demo_runs_comparison_all6.json")

    # Load existing if available
    if os.path.exists(comparison_file):
        try:
            with open(comparison_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    run_list = []
    for prof in profiles:
        for m in modes:
            run_list.append((m, prof))

    log.info("Execution Plan: %d runs scheduled: %s", len(run_list), run_list)

    for idx, (m, prof) in enumerate(run_list):
        run_key = f"{m}_{prof}"
        log.info("\n>>> STARTING RUN %d/%d: %s <<<", idx + 1, len(run_list), run_key)
        res = run_single_demo(mode=m, profile=prof, seed=44)
        results[run_key] = res

        # Save cumulative results
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if idx < len(run_list) - 1:
            log.info("Pausing 10s between runs for socket closure and cooling...")
            time.sleep(10.0)

    # Print summary table
    print("\n" + "=" * 110)
    print("PERM DEMO CONTROLLER EVALUATION RESULTS (Seed 44, Initial 2 Workers)")
    print("NOTE: These are illustrative single runs, not multi-seed experimental results.")
    print("=" * 110)
    print(f"{'Run':<24} | {'1st Scale Rel Burst':<20} | {'W @ <=25s':<10} | {'Peak Util':<10} | {'Sec >70%':<10} | {'Failures':<9} | {'Worker-s':<9}")
    print("-" * 110)

    for run_key, r in results.items():
        rel_scale = f"{r.get('first_scale_up_rel_sec', 'N/A')}s"
        w_25s = str(r.get("workers_first_25s", "N/A"))
        peak_u = f"{r.get('peak_pool_utilization_pct', 0.0):.1f}%"
        sec_70 = str(r.get("seconds_above_70", 0))
        fails = str(r.get("failed_requests", 0))
        w_s = f"{r.get('worker_seconds', 0.0):.1f}"
        print(f"{run_key:<24} | {rel_scale:<20} | {w_25s:<10} | {peak_u:<10} | {sec_70:<10} | {fails:<9} | {w_s:<9}")

    print("=" * 110 + "\n")

    # Detailed timelines
    print("DETAILED WORKER TIMELINES:")
    for run_key, r in results.items():
        print(f"[{run_key}]")
        print(f"  Timeline: {format_timeline(r.get('workers_over_time', []))}")


if __name__ == "__main__":
    main()
