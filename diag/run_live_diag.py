"""
diag/run_live_diag.py
=====================
Automated test harness for Part 2 live diagnostic runs.
Executes 5 live runs with Locust headless, seed 44:
  1. flat_4 (120s)
  2. step_late (120s)
  3. ramp_late (120s)
  4. step (120s)
  5. ramp (135s)

For each run:
  - Spawns telemetry_collector.py (--workers 2, port 5000)
  - Spawns inference_service.py (--port 5001, PERM_FORECASTER=stagbdt)
  - Spawns demo_controller.py (--mode predictive --approval auto)
  - Runs Locust headless with diag/locustfile_diag.py (seed 44)
  - Copies results/controller_trace.csv to diag/controller_trace_<profile>.csv
  - Analyzes trace for scale-up events, timings, and worker history
  - Terminates child processes cleanly
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TRACE_SRC = os.path.join(RESULTS_DIR, "controller_trace.csv")

PYTHON_EXE = sys.executable

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

PROFILES = [
    ("flat_4", 125),
    ("step_late", 125),
    ("ramp_late", 125),
    ("step", 125),
    ("ramp", 140),
]


def kill_proc(p: Optional[subprocess.Popen]):
    if p is None:
        return
    try:
        p.terminate()
        p.wait(timeout=2.0)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def wait_healthy(url: str, proc: Optional[subprocess.Popen] = None, timeout: float = 25.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc and proc.poll() is not None:
            print(f"Process died with exit code {proc.poll()}")
            return False
        try:
            r = requests.get(url, timeout=0.8)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_single_profile(profile: str, duration: int, seed: int = 44) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STARTING LIVE RUN: profile={profile}, duration={duration}s, seed={seed}")
    print("=" * 70)

    cleanup_stale_processes()
    time.sleep(1.0)

    # 1. Clean previous results/controller_trace.csv
    if os.path.exists(TRACE_SRC):
        try:
            os.remove(TRACE_SRC)
        except Exception:
            pass

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    collector_proc = None
    inference_proc = None
    controller_proc = None
    col_log = None
    inf_log = None
    ctrl_log = None

    try:
        # Start Telemetry Collector
        col_log = open(os.path.join(DIAG_DIR, f"collector_{profile}.log"), "w", encoding="utf-8")
        col_cmd = [PYTHON_EXE, os.path.join(PROJECT_ROOT, "telemetry_collector.py"), "--port", "5000", "--workers", "2"]
        collector_proc = subprocess.Popen(col_cmd, cwd=PROJECT_ROOT, stdout=col_log, stderr=col_log)
        if not wait_healthy("http://127.0.0.1:5000/health", proc=collector_proc, timeout=20.0):
            raise RuntimeError("Telemetry collector failed to become healthy on port 5000")
        print("  [OK] Telemetry Collector healthy on :5000 (initial workers: 2)")

        # Start Inference Service with PERM_FORECASTER=stagbdt
        inf_log = open(os.path.join(DIAG_DIR, f"inference_{profile}.log"), "w", encoding="utf-8")
        inf_env = os.environ.copy()
        inf_env["PERM_FORECASTER"] = "stagbdt"
        inf_cmd = [PYTHON_EXE, os.path.join(PROJECT_ROOT, "inference_service.py"), "--port", "5001"]
        inference_proc = subprocess.Popen(inf_cmd, cwd=PROJECT_ROOT, env=inf_env, stdout=inf_log, stderr=inf_log)
        if not wait_healthy("http://127.0.0.1:5001/health", proc=inference_proc, timeout=25.0):
            raise RuntimeError("Inference service failed to become healthy on port 5001")
        print("  [OK] Inference Service healthy on :5001 (ST-AGBDT enabled)")

        # Start Demo Controller
        ctrl_log = open(os.path.join(DIAG_DIR, f"controller_{profile}.log"), "w", encoding="utf-8")
        ctrl_cmd = [
            PYTHON_EXE,
            os.path.join(PROJECT_ROOT, "demo_controller.py"),
            "--mode", "predictive",
            "--approval", "auto",
            "--initial-workers", "2",
            "--collector-url", "http://127.0.0.1:5000",
            "--inference-url", "http://127.0.0.1:5001",
        ]
        controller_proc = subprocess.Popen(ctrl_cmd, cwd=PROJECT_ROOT, stdout=ctrl_log, stderr=ctrl_log)
        time.sleep(2.0)
        print("  [OK] Demo Controller started (predictive mode, auto approval)")

        # Run Locust headless
        loc_env = os.environ.copy()
        loc_env["PERM_PROFILE"] = profile
        loc_env["PERM_SEED"] = str(seed)
        loc_env["PERM_LOAD"] = "7"
        loc_env["PERM_TIMEOUT"] = "3.0"

        locust_file = os.path.join(DIAG_DIR, "locustfile_diag.py")
        loc_cmd = [
            "locust",
            "-f", locust_file,
            "--headless",
            "-u", "1",
            "-r", "1",
            "-t", f"{duration}s",
            "--host", "http://127.0.0.1:5000",
        ]

        print(f"  [RUNNING] Locust headless ({profile}, seed={seed}, dur={duration}s)...")
        t_loc_start = time.time()
        loc_proc = subprocess.Popen(loc_cmd, cwd=PROJECT_ROOT, env=loc_env)
        loc_proc.wait()
        loc_elapsed = time.time() - t_loc_start
        print(f"  [FINISHED] Locust completed in {loc_elapsed:.1f}s (exit code {loc_proc.returncode})")

        # Give controller 3 seconds to flush final trace rows
        time.sleep(3.0)

    finally:
        kill_proc(controller_proc)
        kill_proc(inference_proc)
        kill_proc(collector_proc)
        for fh in [ctrl_log, inf_log, col_log]:
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        time.sleep(2.0)

    # Copy trace CSV
    trace_dest = os.path.join(DIAG_DIR, f"controller_trace_{profile}.csv")
    if os.path.exists(TRACE_SRC):
        shutil.copy2(TRACE_SRC, trace_dest)
        print(f"  [COPIED] Trace saved to {trace_dest}")
    else:
        print(f"  [WARNING] Trace file {TRACE_SRC} was not generated!")
        return {"profile": profile, "error": "Trace file missing"}

    # Analyze trace data
    return analyze_trace(trace_dest, profile)


def cleanup_stale_processes():
    """Kill any lingering perm service processes."""
    try:
        cmd = 'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | Select-Object ProcessId, CommandLine'
        out = subprocess.check_output(["powershell", "-Command", cmd], universal_newlines=True)
        for line in out.splitlines():
            if any(k in line for k in ["demo_controller.py", "telemetry_collector.py", "inference_service.py"]):
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    pid = int(parts[0])
                    try:
                        subprocess.run(["powershell", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"], timeout=3)
                    except Exception:
                        pass
    except Exception:
        pass


def analyze_trace(trace_path: str, profile: str) -> Dict[str, Any]:
    with open(trace_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_lines = [line for line in reader if line]

    if not all_lines:
        return {"profile": profile, "rows": 0, "first_scale_up_t": None}

    # Determine if line 0 is a header
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

    if not rows:
        return {"profile": profile, "rows": 0, "first_scale_up_t": None}

    # Find first scale up above 2 workers
    first_scale_up_t = None
    first_scale_up_workers = None
    first_scale_up_rule = None
    scaled_above_2 = False

    load_change_t = None
    workers_timeline = []
    prev_w = None

    for r in rows:
        try:
            t_val = float(r["t"])
            arr_val = float(r["arrivals"])
            w_val = int(r["workers"])
        except (ValueError, KeyError):
            continue

        if load_change_t is None and arr_val >= 8.0:
            load_change_t = t_val

        if w_val > 2 and not scaled_above_2:
            scaled_above_2 = True
            first_scale_up_t = t_val
            first_scale_up_workers = w_val

        if w_val != prev_w:
            workers_timeline.append((t_val, w_val))
            prev_w = w_val

    # Scaled up before load changed?
    scaled_before_load_change = False
    if first_scale_up_t is not None:
        if load_change_t is None:
            # Load never changed (e.g. flat_4)
            scaled_before_load_change = True
        else:
            scaled_before_load_change = (first_scale_up_t < load_change_t)

    summary = {
        "profile": profile,
        "total_rows": len(rows),
        "first_scale_up_t": first_scale_up_t,
        "first_scale_up_workers": first_scale_up_workers,
        "load_change_t": load_change_t,
        "scaled_before_load_change": scaled_before_load_change,
        "workers_timeline": workers_timeline,
    }

    print("\n--- Run Analysis Summary ---")
    print(f"Profile: {profile}")
    print(f"First scale-up time           : {first_scale_up_t}s (to {first_scale_up_workers} workers)")
    print(f"Load change arrival time (>8) : {load_change_t}s")
    print(f"Scaled before load change?    : {scaled_before_load_change}")
    print(f"Workers timeline (t, workers) : {workers_timeline}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Live Diagnostic Runner")
    parser.add_argument("--profile", type=str, default="all", help="Specific profile to run or 'all'")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    to_run = PROFILES if args.profile == "all" else [(args.profile, 125)]

    results = []
    for prof, dur in to_run:
        res = run_single_profile(prof, dur, seed=args.seed)
        results.append(res)

    print("\n" + "=" * 80)
    print("DECISION TABLE (MEASURED DATA ONLY)")
    print("=" * 80)
    print("Profile    | Scaled >2? | 1st Scale-Up (s) | Load Change (s) | Scaled Before Load? | Timeline")
    print("-----------+------------+------------------+-----------------+---------------------+-------------------------")
    for r in results:
        prof = r.get("profile", "unknown")
        scaled = "YES" if r.get("first_scale_up_t") is not None else "NO"
        t_sc = f"{r.get('first_scale_up_t'):.1f}s" if r.get("first_scale_up_t") is not None else "N/A"
        t_ld = f"{r.get('load_change_t'):.1f}s" if r.get("load_change_t") is not None else "Never"
        before = "YES" if r.get("scaled_before_load_change") else "NO"
        tl = " -> ".join([f"{w}w@{t:.0f}s" for t, w in r.get("workers_timeline", [])])
        print(f"{prof:10s} | {scaled:10s} | {t_sc:16s} | {t_ld:15s} | {before:19s} | {tl}")
    print("=" * 80)

    # Save summary JSON
    sum_path = os.path.join(DIAG_DIR, "live_decision_table.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {sum_path}")


if __name__ == "__main__":
    main()
