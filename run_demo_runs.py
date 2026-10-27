"""
PERM - run_demo_runs.py
=======================
Automated runner for D3 Demo Runs (Step Profile, Seed 44, 2 initial workers).
Runs one at a time, alone:
  Run 1: --mode reactive
  Run 2: --mode predictive

Metrics collected per run:
  1. Peak pool utilization (%)
  2. Seconds above 70%
  3. Time from burst start to first scale-up (s)
  4. Workers over time (timeline / intervals)
  5. Final pool utilization (%)
  6. Failed requests (count)
  7. Worker-seconds (integrated active workers over time)

All data is strictly real measured telemetry from the live testbed runs.
"""

from __future__ import annotations

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


def run_single_demo(mode: str) -> Dict[str, Any]:
    log.info("=" * 70)
    log.info("STARTING DEMO RUN: MODE = %s (Step Profile, Seed 44, Initial 2 Workers)", mode.upper())
    log.info("=" * 70)

    collector_proc = None
    inference_proc = None
    controller_proc = None
    locust_proc = None

    run_events_csv = os.path.join(RESULTS_DIR, f"controller_events_{mode}.csv")
    if os.path.exists(run_events_csv):
        os.remove(run_events_csv)

    locust_csv_prefix = os.path.join(RESULTS_DIR, f"locust_demo_{mode}")

    telemetry_timeline: List[Dict[str, Any]] = []

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
                "140",
            ],
            cwd=PROJECT_ROOT,
        )
        time.sleep(2.0)  # Controller stabilization

        # 4. Start Locust Step Profile with Seed 44
        log.info("Launching Locust step profile (seed 44, duration 125s)...")
        loc_env = os.environ.copy()
        loc_env["PERM_PROFILE"] = "step"
        loc_env["PERM_SEED"] = "44"

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
            "125s",
            "--host",
            COLLECTOR_URL,
            "--csv",
            locust_csv_prefix,
        ]

        t_locust_start = time.monotonic()
        locust_proc = subprocess.Popen(locust_cmd, cwd=PROJECT_ROOT, env=loc_env)

        # Burst start is exactly offset 30.0s after Locust start
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

                    # Detect first scale-up
                    if workers > 2 and first_scale_up_time is None:
                        first_scale_up_time = t_now
                        first_scale_up_target = workers
                        log.info(
                            "FIRST SCALE-UP DETECTED: 2 -> %d workers at elapsed=%.2fs (t - burst_start = %.2fs)",
                            workers,
                            elapsed_from_start,
                            t_now - burst_start_time,
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
        time.sleep(4.0)  # Wait for sockets to close

    # 5. Compute Run Metrics from Real Telemetry
    if not telemetry_timeline:
        raise RuntimeError("No telemetry was collected during demo run!")

    peak_pool_util = max(s["smoothed_pool_util"] for s in telemetry_timeline)
    seconds_above_70 = sum(1 for s in telemetry_timeline if s["smoothed_pool_util"] > 70.0)

    if first_scale_up_time is not None:
        time_to_first_scale = round(max(0.0, first_scale_up_time - burst_start_time), 2)
    else:
        time_to_first_scale = None

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
        "profile": "step",
        "seed": 44,
        "initial_workers": 2,
        "peak_pool_utilization_pct": round(peak_pool_util, 2),
        "seconds_above_70": seconds_above_70,
        "time_to_first_scale_sec": time_to_first_scale,
        "first_scale_up_target": first_scale_up_target,
        "workers_over_time": worker_transitions,
        "final_utilization_pct": round(final_util, 2),
        "failed_requests": max(total_failures, locust_failures),
        "locust_requests": locust_requests,
        "locust_failures": locust_failures,
        "worker_seconds": worker_seconds,
        "num_telemetry_samples": len(telemetry_timeline),
        "controller_events": events_from_csv,
    }

    # Save summary and timeline
    summary_path = os.path.join(RESULTS_DIR, f"demo_run_{mode}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    timeline_path = os.path.join(RESULTS_DIR, f"telemetry_timeline_{mode}.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_timeline, f, indent=2)

    log.info("Finished demo run [%s]. Summary saved to %s", mode, summary_path)
    return run_summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run D3 Demo Runs")
    parser.add_argument(
        "--mode",
        choices=["both", "reactive", "predictive"],
        default="both",
        help="Which demo runs to execute: 'both', 'reactive', or 'predictive'",
    )
    args = parser.parse_args()

    log.info("Starting PERM D3 Demo Runs execution (target: %s)...", args.mode)
    results = {}

    reactive_summary_path = os.path.join(RESULTS_DIR, "demo_run_reactive.json")
    predictive_summary_path = os.path.join(RESULTS_DIR, "demo_run_predictive.json")

    # Run Reactive
    if args.mode in ("both", "reactive"):
        results["reactive"] = run_single_demo("reactive")
    elif os.path.exists(reactive_summary_path):
        with open(reactive_summary_path, "r", encoding="utf-8") as f:
            results["reactive"] = json.load(f)

    if args.mode == "both":
        log.info("Pausing 10s between demo runs to allow cooling and socket recycling...")
        time.sleep(10.0)

    # Run Predictive
    if args.mode in ("both", "predictive"):
        results["predictive"] = run_single_demo("predictive")
    elif os.path.exists(predictive_summary_path):
        with open(predictive_summary_path, "r", encoding="utf-8") as f:
            results["predictive"] = json.load(f)

    combined_summary_path = os.path.join(RESULTS_DIR, "demo_runs_comparison.json")
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print comparative table
    print("\n" + "=" * 80)
    print("PERM DEMO RUNS COMPARISON (Step Profile, Seed 44, 2 Initial Workers)")
    print("NOTE: These are two illustrative demonstration runs, not experimental results.")
    print("=" * 80)
    print(f"{'Metric':<40} | {'Reactive':<16} | {'Predictive':<16}")
    print("-" * 80)
    r = results["reactive"]
    p = results["predictive"]

    print(f"{'Peak Pool Utilization (%)':<40} | {r['peak_pool_utilization_pct']:<16.1f} | {p['peak_pool_utilization_pct']:<16.1f}")
    print(f"{'Seconds Above 70%':<40} | {r['seconds_above_70']:<16d} | {p['seconds_above_70']:<16d}")
    print(f"{'Time from Burst Start to 1st Scale (s)':<40} | {str(r['time_to_first_scale_sec']) + 's':<16} | {str(p['time_to_first_scale_sec']) + 's':<16}")
    print(f"{'Final Utilization (%)':<40} | {r['final_utilization_pct']:<16.1f} | {p['final_utilization_pct']:<16.1f}")
    print(f"{'Failed Requests':<40} | {r['failed_requests']:<16d} | {p['failed_requests']:<16d}")
    print(f"{'Worker-Seconds':<40} | {r['worker_seconds']:<16.1f} | {p['worker_seconds']:<16.1f}")
    print("-" * 80)
    print(f"Reactive Workers Timeline:   {r['workers_over_time']}")
    print(f"Predictive Workers Timeline: {p['workers_over_time']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
