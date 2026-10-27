"""
run_part_d_diagnostic.py
========================
Step 3b-L Part D: CPU diagnostic executed ALONE (120 s).
Runs one step profile at 8 workers and, every second, logs the top 10 processes by CPU
(including the collection runner, sampler, IDE, Defender).
Reports the summary table and unattributed share. Does not change system settings.
"""

import os
import sys
import time
import json
import csv
import subprocess
import logging
from typing import Dict, List, Any
import psutil
import requests
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("part_d_diag")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "locust")
os.makedirs(OUT_DIR, exist_ok=True)
DIAG_CSV = os.path.join(OUT_DIR, "part_d_cpu_diagnostic.csv")
DIAG_JSON = os.path.join(OUT_DIR, "part_d_summary.json")

COLLECTOR_URL = "http://127.0.0.1:5000"
NUM_LOGICAL_CPUS = 16


def wait_healthy(url: str, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    print("=" * 80)
    print("PART D: 120-SECOND CPU DIAGNOSTIC AT 8 WORKERS (STEP PROFILE)")
    print("=" * 80)

    # 1. Start telemetry collector on port 5000 with 8 workers
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "2.0"

    log.info("Starting telemetry collector with 8 workers on port 5000...")
    collector_proc = subprocess.Popen(
        [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "8"],
        env=env,
        cwd=PROJECT_ROOT,
    )

    locust_proc = None
    try:
        if not wait_healthy(f"{COLLECTOR_URL}/health", timeout=30.0):
            raise RuntimeError("Telemetry collector failed to start on port 5000")

        # Set capacity to 8 workers
        requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=5.0)
        time.sleep(2.0)

        # 2. Start Locust open-loop traffic generator (step profile, seed 42, 120s)
        locust_env = os.environ.copy()
        locust_env["PERM_PROFILE"] = "step"
        locust_env["PERM_SEED"] = "42"
        locust_env["PERM_LOAD"] = "7"
        locust_env["PERM_TIMEOUT"] = "3.0"

        locust_cmd = [
            "locust",
            "-f", "locustfile_perm.py",
            "--headless",
            "-u", "1",
            "-r", "1",
            "--host", COLLECTOR_URL,
            "--run-time", "120s",
        ]

        log.info("Starting Locust generator (step profile, seed 42, 120s)...")
        locust_proc = subprocess.Popen(
            locust_cmd,
            env=locust_env,
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Prime psutil CPU monitoring
        psutil.cpu_percent(interval=None)
        all_procs = {}
        for p in psutil.process_iter(["pid", "name"]):
            try:
                p.cpu_percent(interval=None)
                all_procs[p.pid] = p
            except Exception:
                pass

        log.info("Commencing 120-second per-second CPU diagnostic logging...")
        samples = []
        process_totals: Dict[str, List[float]] = {}
        unattributed_samples = []
        total_sys_samples = []

        start_time = time.monotonic()
        for sec in range(1, 121):
            time.sleep(1.0)
            t_now = time.monotonic() - start_time
            sys_cpu = psutil.cpu_percent(interval=None)
            total_sys_samples.append(sys_cpu)

            # Collect CPU for all running processes
            current_snapshot = []
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    raw_cpu = p.cpu_percent(interval=None)
                    mach_cpu = raw_cpu / float(NUM_LOGICAL_CPUS)
                    p_name = p.info["name"] or "unknown"
                    cmd = " ".join(p.info["cmdline"] or [])

                    # Categorize key processes
                    display_name = p_name
                    if "telemetry_collector.py" in cmd:
                        display_name = "python.exe (telemetry_collector/Flask)"
                    elif "locustfile_perm.py" in cmd or "locust" in p_name.lower():
                        display_name = "locust.exe (traffic_dispatcher)"
                    elif "run_part_d_diagnostic.py" in cmd:
                        display_name = "python.exe (CPU_diagnostic_runner)"
                    elif "multiprocessing" in cmd and "telemetry_collector" in cmd:
                        display_name = "python.exe (worker_pool_process)"
                    elif "antigravity ide" in p_name.lower() or "antigravity" in cmd.lower():
                        display_name = "Antigravity IDE.exe"
                    elif "msmpeng" in p_name.lower():
                        display_name = "MsMpEng.exe (Windows_Defender)"
                    elif "system idle process" in p_name.lower():
                        display_name = "System Idle Process"
                    elif p_name.lower() == "system":
                        display_name = "System (Kernel)"

                    current_snapshot.append({
                        "pid": p.pid,
                        "name": display_name,
                        "raw_core_cpu": raw_cpu,
                        "mach_cpu": mach_cpu,
                    })
                except Exception:
                    pass

            # Sort by machine CPU descending
            non_idle_procs = [p for p in current_snapshot if p["name"] != "System Idle Process"]
            non_idle_procs.sort(key=lambda x: x["mach_cpu"], reverse=True)
            top10 = non_idle_procs[:10]

            sum_top_mach = sum(p["mach_cpu"] for p in non_idle_procs)
            # Unattributed share = sys_cpu - sum of all non-idle measured processes
            unattributed = max(0.0, sys_cpu - sum_top_mach)
            unattributed_samples.append(unattributed)

            # Record for aggregation
            for p in top10:
                process_totals.setdefault(p["name"], []).append(p["mach_cpu"])

            samples.append({
                "second": sec,
                "elapsed_sec": round(t_now, 2),
                "system_cpu_pct": round(sys_cpu, 2),
                "unattributed_cpu_pct": round(unattributed, 2),
                "top_10": [
                    {"name": p["name"], "mach_cpu_pct": round(p["mach_cpu"], 2)} for p in top10
                ],
            })

            if sec % 20 == 0 or sec == 120:
                log.info(
                    "Sec %3d/120: System CPU = %5.1f%% | Top Process: %s (%4.1f%%) | Unattributed = %4.1f%%",
                    sec, sys_cpu, top10[0]["name"] if top10 else "None", top10[0]["mach_cpu"] if top10 else 0.0, unattributed
                )

        log.info("120-second diagnostic run completed. Aggregating results...")

        # Calculate average machine CPU per process over all 120 seconds
        all_process_means = {}
        for p_name, vals in process_totals.items():
            # Mean over 120 seconds (padding with 0 for seconds where process was not in top list)
            padded_vals = vals + [0.0] * (120 - len(vals))
            all_process_means[p_name] = np.mean(padded_vals)

        sorted_process_means = sorted(all_process_means.items(), key=lambda x: x[1], reverse=True)[:10]

        mean_sys_cpu = round(float(np.mean(total_sys_samples)), 2)
        mean_unattributed = round(float(np.mean(unattributed_samples)), 2)
        unattributed_share_pct = round((mean_unattributed / mean_sys_cpu * 100.0) if mean_sys_cpu > 0 else 0.0, 2)

        summary_data = {
            "duration_sec": 120,
            "mean_system_cpu_pct": mean_sys_cpu,
            "mean_unattributed_cpu_pct": mean_unattributed,
            "unattributed_share_pct_of_system": unattributed_share_pct,
            "top_10_processes_by_machine_cpu": [
                {
                    "rank": i + 1,
                    "process_name": p[0],
                    "mean_machine_cpu_pct": round(float(p[1]), 2),
                    "mean_core_cpu_pct": round(float(p[1] * NUM_LOGICAL_CPUS), 2),
                    "thread_equivalents": round(float(p[1] * NUM_LOGICAL_CPUS / 100.0), 3),
                }
                for i, p in enumerate(sorted_process_means)
            ],
        }

        # Write summary JSON
        with open(DIAG_JSON, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Write detailed CSV
        with open(DIAG_CSV, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["second", "elapsed_sec", "system_cpu_pct", "unattributed_cpu_pct", "top1_name", "top1_cpu", "top2_name", "top2_cpu", "top3_name", "top3_cpu"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                top = s["top_10"]
                writer.writerow({
                    "second": s["second"],
                    "elapsed_sec": s["elapsed_sec"],
                    "system_cpu_pct": s["system_cpu_pct"],
                    "unattributed_cpu_pct": s["unattributed_cpu_pct"],
                    "top1_name": top[0]["name"] if len(top) > 0 else "",
                    "top1_cpu": top[0]["mach_cpu_pct"] if len(top) > 0 else 0.0,
                    "top2_name": top[1]["name"] if len(top) > 1 else "",
                    "top2_cpu": top[1]["mach_cpu_pct"] if len(top) > 1 else 0.0,
                    "top3_name": top[2]["name"] if len(top) > 2 else "",
                    "top3_cpu": top[2]["mach_cpu_pct"] if len(top) > 2 else 0.0,
                })

        # Print Output Table
        print("\n" + "=" * 95)
        print("PART D: TOP 10 PROCESSES BY CPU OVER 120-SECOND STEP PROFILE (8 WORKERS)")
        print("=" * 95)
        print(f"Overall System CPU Mean:       {mean_sys_cpu:>6.2f}% (percentage of 16 logical cores)")
        print(f"Mean Unattributed Share:       {mean_unattributed:>6.2f}% ({unattributed_share_pct:.1f}% of total system CPU)")
        print("-" * 95)
        header = f"{'Rank':<5} | {'Process Name':<42} | {'Machine %':<11} | {'Core %':<10} | {'Thread Equiv'}"
        print(header)
        print("-" * 95)
        for p in summary_data["top_10_processes_by_machine_cpu"]:
            print(f"{p['rank']:<5} | {p['process_name']:<42} | {p['mean_machine_cpu_pct']:>9.2f}% | {p['mean_core_cpu_pct']:>8.2f}% | {p['thread_equivalents']:>10.3f} threads")

        print("=" * 95)

    finally:
        log.info("Terminating started diagnostic processes...")
        if locust_proc and locust_proc.poll() is None:
            locust_proc.terminate()
            locust_proc.wait(timeout=5)
        if collector_proc and collector_proc.poll() is None:
            collector_proc.terminate()
            collector_proc.wait(timeout=5)
        log.info("Cleanup complete.")


if __name__ == "__main__":
    main()
