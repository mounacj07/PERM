"""
smoke_test_l0.py
================
L0 Smoke Test Gate:
Compares Locust open-loop traffic generator (locustfile_perm.py) vs burst_test.py
on the identical 'step' profile (seed 42, load 7, fixed 8 workers).

Evaluates:
  - Per-second achieved arrivals vs target: MAE, mean bias %
  - Step-response lag (seconds to >= 90% of the step, i.e., 36.4 req/s)
  - Poisson variance / mean (Fano factor) in calm and burst steady segments
  - Generator CPU (% of machine)
  - Flask CPU (% of machine)
  - Failed requests (rejected + timeout)

PASS Criteria:
  |bias| < 5.0% AND lag < 2.0s AND loadgen_cpu_pct > 0.0%
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import requests

import burst_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test_l0")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_URL = "http://127.0.0.1:5000"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "smoke_test")


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


def get_step_target(elapsed_sec: float) -> float:
    # 30s calm @ 4, 60s burst @ 40, 30s calm @ 4
    if elapsed_sec < 30.0:
        return 4.0
    elif elapsed_sec < 90.0:
        return 40.0
    else:
        return 4.0


def run_single_smoke_run(
    generator_type: str,
    profile: str = "step",
    seed: int = 42,
    load: int = 7,
) -> Tuple[List[Dict], float]:
    """
    Run one testbed run using either 'locust' or 'burst_test'.
    Returns (collected_rows, mean_loadgen_cpu_pct).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Start telemetry collector with 8 workers pinned to cores [4-15]
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "2.0"

    log.info("[%s] Launching telemetry collector with 8 workers...", generator_type)
    collector_proc = subprocess.Popen(
        [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "8"],
        env=env,
        cwd=PROJECT_ROOT,
    )

    locust_proc: Optional[subprocess.Popen] = None
    collected_rows: List[Dict] = []
    loadgen_cpu_samples: List[float] = []
    flask_cpu_samples: List[float] = []

    try:
        if not _wait_healthy(f"{COLLECTOR_URL}/health", timeout=30.0):
            raise RuntimeError("Telemetry collector failed to start")

        # Guarantee capacity fixed at 8
        requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=5.0)
        time.sleep(2.0)

        # Attach psutil to collector and its worker children
        flask_psutil = psutil.Process(collector_proc.pid)
        try:
            flask_psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # Prepare generator
        offsets, burst_offset, total_dur = burst_test.generate_schedule(profile, seed)

        stop_sampling = threading.Event()
        t0_mono = time.monotonic()
        traffic_start_mono: Optional[float] = None

        # Sampler thread
        def _sample_loop():
            nonlocal traffic_start_mono
            last_seen_ts = None
            locust_ps: Optional[psutil.Process] = None

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

                    # Track traffic start when first arrival is observed
                    arr_cnt = int(m.get("arrivals_count", 0))
                    if traffic_start_mono is None and arr_cnt > 0:
                        traffic_start_mono = now_mono

                    elapsed = round(now_mono - (traffic_start_mono or t0_mono), 2)

                    # Measure process CPU
                    try:
                        f_cpu = flask_psutil.cpu_percent(interval=None)
                    except Exception:
                        f_cpu = 0.0

                    lg_cpu = 0.0
                    if generator_type == "locust":
                        if locust_ps is None and locust_proc is not None:
                            try:
                                locust_ps = psutil.Process(locust_proc.pid)
                                locust_ps.cpu_percent(interval=None)
                            except Exception:
                                pass
                        elif locust_ps is not None:
                            try:
                                procs = [locust_ps] + locust_ps.children(recursive=True)
                                lg_cpu = sum(p.cpu_percent(interval=None) for p in procs)
                            except Exception:
                                lg_cpu = 0.0

                    flask_cpu_samples.append(f_cpu)
                    if generator_type == "locust" and lg_cpu > 0.0:
                        loadgen_cpu_samples.append(lg_cpu)

                    row = {
                        "elapsed_sec": elapsed,
                        "timestamp": ts,
                        "arrivals_count": arr_cnt,
                        "ok_count": m.get("ok_count", 0),
                        "rejected_count": m.get("rejected_count", 0),
                        "timeout_count": m.get("timeout_count", 0),
                        "p95_latency_ms": m.get("p95_latency_ms", 0.0),
                        "ok_p95_latency_ms": m.get("ok_p95_latency_ms", 0.0),
                        "cpu_percent": m.get("cpu_percent", 0.0),
                        "worker_cpu_percent": m.get("worker_cpu_percent", 0.0),
                        "flask_cpu_pct": round(f_cpu, 2),
                        "loadgen_cpu_pct": round(lg_cpu, 2),
                    }
                    collected_rows.append(row)

        sampler_th = threading.Thread(target=_sample_loop, daemon=True)
        sampler_th.start()

        # Run traffic generator
        if generator_type == "locust":
            loc_env = os.environ.copy()
            loc_env["PERM_PROFILE"] = profile
            loc_env["PERM_SEED"] = str(seed)
            loc_env["PERM_LOAD"] = str(load)
            loc_env["PERM_TIMEOUT"] = "3.0"

            csv_prefix = os.path.join(OUTPUT_DIR, f"locust_{profile}_seed{seed}")

            log.info("Starting Locust headless open-loop dispatcher...")
            locust_proc = subprocess.Popen(
                [
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
                    f"{int(total_dur + 5)}s",
                    "--host",
                    COLLECTOR_URL,
                    "--csv",
                    csv_prefix,
                ],
                env=loc_env,
                cwd=PROJECT_ROOT,
            )

            # Pin Locust process to cores [0, 1, 2, 3] and init cpu_percent baseline
            try:
                p_loc = psutil.Process(locust_proc.pid)
                p_loc.cpu_affinity([0, 1, 2, 3])
                p_loc.cpu_percent(interval=None)
            except Exception:
                pass

            locust_proc.wait(timeout=total_dur + 20)
            log.info("Locust process completed.")

        elif generator_type == "burst_test":
            log.info("Starting burst_test.py...")
            burst_test.run_burst(
                target_url=f"{COLLECTOR_URL}/api/process",
                profile=profile,
                seed=seed,
                load=load,
                timeout_sec=3.0,
            )
            log.info("burst_test run completed.")

        # Allow trailing metrics
        time.sleep(2.0)
        stop_sampling.set()
        sampler_th.join(timeout=3.0)

    finally:
        _kill_proc(locust_proc)
        _kill_proc(collector_proc)
        time.sleep(2.0)

    mean_lg_cpu = float(np.mean(loadgen_cpu_samples)) if loadgen_cpu_samples else 0.0
    return collected_rows, mean_lg_cpu


def analyze_run(rows: List[Dict], mean_lg_cpu: float) -> Dict:
    """
    Compute MAE, bias %, step lag, Poisson Fano factor (var/mean), CPU, and failures.
    """
    # Extract arrivals per second aligned to target
    achieved = []
    targets = []
    flasks = []
    failed_total = 0

    calm_arrivals = []
    burst_arrivals = []

    for r in rows:
        el = r["elapsed_sec"]
        if el < 0.0 or el > 120.0:
            continue
        arr = int(r["arrivals_count"])
        tgt = get_step_target(el)

        achieved.append(arr)
        targets.append(tgt)
        flasks.append(r.get("flask_cpu_pct", 0.0))
        failed_total += int(r["rejected_count"]) + int(r["timeout_count"])

        # Calm segment: seconds 5 to 25
        if 5.0 <= el < 25.0:
            calm_arrivals.append(arr)
        # Burst steady segment: seconds 35 to 85
        elif 35.0 <= el < 85.0:
            burst_arrivals.append(arr)

    achieved_arr = np.array(achieved, dtype=np.float64)
    target_arr = np.array(targets, dtype=np.float64)

    # 1. Per-second MAE
    mae = float(np.mean(np.abs(achieved_arr - target_arr)))

    # 2. Mean bias %: 100 * (mean_achieved - mean_target) / mean_target
    mean_achieved = float(np.mean(achieved_arr))
    mean_target = float(np.mean(target_arr))
    bias_pct = ((mean_achieved - mean_target) / mean_target) * 100.0

    # 3. Step-response lag (seconds from t=30 to reach >= 90% of step = 36.4 req/s)
    lag = 0.0
    for r in rows:
        el = r["elapsed_sec"]
        if el >= 30.0 and int(r["arrivals_count"]) >= 36:
            lag = el - 30.0
            break

    # 4. Poisson Fano factor (variance / mean)
    # Calm segment (target = 4.0)
    calm_fano = float(np.var(calm_arrivals) / np.mean(calm_arrivals)) if len(calm_arrivals) > 2 and np.mean(calm_arrivals) > 0 else 0.0
    # Burst segment (target = 40.0)
    burst_fano = float(np.var(burst_arrivals) / np.mean(burst_arrivals)) if len(burst_arrivals) > 2 and np.mean(burst_arrivals) > 0 else 0.0

    mean_flask = float(np.mean(flasks)) if flasks else 0.0

    return {
        "mae": round(mae, 2),
        "mean_achieved": round(mean_achieved, 2),
        "mean_target": round(mean_target, 2),
        "bias_pct": round(bias_pct, 2),
        "step_lag_sec": round(lag, 2),
        "calm_fano_var_mean": round(calm_fano, 2),
        "burst_fano_var_mean": round(burst_fano, 2),
        "loadgen_cpu_pct": round(mean_lg_cpu, 2),
        "flask_cpu_pct": round(mean_flask, 2),
        "failed_requests": failed_total,
    }


def main():
    print("=" * 70)
    print("L0 SMOKE TEST GATE: LOCUST vs BURST_TEST COMPARISON")
    print("=" * 70)

    # 1. Run Locust
    log.info("Executing Pass 1: Locust Open-Loop Dispatcher...")
    locust_rows, locust_lg_cpu = run_single_smoke_run(generator_type="locust", profile="step", seed=42)
    locust_metrics = analyze_run(locust_rows, locust_lg_cpu)

    # Save CSV
    locust_csv = os.path.join(OUTPUT_DIR, "smoke_locust_step_seed42.csv")
    with open(locust_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(locust_rows[0].keys()))
        writer.writeheader()
        writer.writerows(locust_rows)
    log.info("Wrote Locust smoke run telemetry to %s", locust_csv)

    time.sleep(3.0)

    # 2. Run burst_test.py
    log.info("Executing Pass 2: burst_test.py Reference Generator...")
    bt_rows, bt_lg_cpu = run_single_smoke_run(generator_type="burst_test", profile="step", seed=42)
    bt_metrics = analyze_run(bt_rows, bt_lg_cpu)

    # Save CSV
    bt_csv = os.path.join(OUTPUT_DIR, "smoke_burst_test_step_seed42.csv")
    with open(bt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bt_rows)
    log.info("Wrote burst_test smoke run telemetry to %s", bt_csv)

    # Summary Table
    print("\n" + "=" * 80)
    print("SMOKE TEST RESULTS TABLE")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Locust (Canonical)':<20} | {'burst_test.py (Ref)':<20}")
    print("-" * 80)
    print(f"{'Per-second Arrivals MAE':<35} | {locust_metrics['mae']:<20} | {bt_metrics['mae']:<20}")
    print(f"{'Mean Bias %':<35} | {locust_metrics['bias_pct']:<20}% | {bt_metrics['bias_pct']:<20}%")
    print(f"{'Step-response Lag (to 90% step)':<35} | {locust_metrics['step_lag_sec']:<20}s | {bt_metrics['step_lag_sec']:<20}s")
    print(f"{'Calm Fano Factor (Var/Mean, ~1)':<35} | {locust_metrics['calm_fano_var_mean']:<20} | {bt_metrics['calm_fano_var_mean']:<20}")
    print(f"{'Burst Fano Factor (Var/Mean, ~1)':<35} | {locust_metrics['burst_fano_var_mean']:<20} | {bt_metrics['burst_fano_var_mean']:<20}")
    print(f"{'Generator CPU % (% of machine)':<35} | {locust_metrics['loadgen_cpu_pct']:<20}% | {bt_metrics['loadgen_cpu_pct']:<20}%")
    print(f"{'Flask CPU % (% of machine)':<35} | {locust_metrics['flask_cpu_pct']:<20}% | {bt_metrics['flask_cpu_pct']:<20}%")
    print(f"{'Failed Requests':<35} | {locust_metrics['failed_requests']:<20} | {bt_metrics['failed_requests']:<20}")
    print("=" * 80)

    # Check Gate
    bias_pass = abs(locust_metrics["bias_pct"]) < 5.0
    lag_pass = locust_metrics["step_lag_sec"] < 2.0
    cpu_pass = locust_metrics["loadgen_cpu_pct"] > 0.0

    print("\nGATE CRITERIA EVALUATION:")
    print(f"1. |bias| < 5.0%:          {locust_metrics['bias_pct']}% -> {'PASS' if bias_pass else 'FAIL'}")
    print(f"2. Step Lag < 2.0s:        {locust_metrics['step_lag_sec']}s -> {'PASS' if lag_pass else 'FAIL'}")
    print(f"3. loadgen_cpu_pct > 0.0%: {locust_metrics['loadgen_cpu_pct']}% -> {'PASS' if cpu_pass else 'FAIL'}")

    overall_pass = bias_pass and lag_pass and cpu_pass
    print(f"\nOVERALL SMOKE TEST GATE: {'PASSED' if overall_pass else 'FAILED'}")
    print("=" * 80)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "locust": locust_metrics,
        "burst_test": bt_metrics,
        "gate_passed": overall_pass,
    }
    with open(os.path.join(OUTPUT_DIR, "smoke_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not overall_pass:
        log.error("SMOKE TEST GATE FAILED! Halting per L0 task instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
