"""
PERM - collect_dataset.py
=========================
Collects benchmark dataset across traffic profiles with capacity FIXED at 8 workers.
Supports two traffic generators:
  1. locust (Default, canonical open-loop Locust dispatcher via locustfile_perm.py)
  2. burst_test (Reference thread-pool generator)

Runs:
  - 6 fixed profiles x 3 seeds (42, 43, 44) = 18 runs
    (step, ramp, spike, double_burst, sawtooth, random_walk)
  - random_mix x 30 seeds (0 to 29) = 30 runs
  Total: 48 runs.

Outputs:
  - data/runs_locust/<run_id>.csv
  - data/runs_locust/manifest.json
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
log = logging.getLogger("collect_dataset")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_URL = "http://127.0.0.1:5000"

FIXED_PROFILES = ["step", "ramp", "spike", "double_burst", "sawtooth", "random_walk"]
FIXED_SEEDS = [42, 43, 44]
RANDOM_MIX_SEEDS = list(range(30))

CSV_HEADER = [
    "run_id",
    "generator",
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
    "flask_cpu_pct",
    "locust_cpu_pct",
    "workers_cpu_pct",
    "other_cpu_pct",
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


def get_profile_duration(profile: str, seed: int = 42) -> float:
    _, _, dur = burst_test.generate_schedule(profile, seed)
    return dur


def check_no_competing_workload():
    """Verify that no other CPU-heavy benchmark, training, or pytest is running."""
    competing = []
    current_pid = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.pid == current_pid:
                continue
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if any(k in cmd for k in ["pytest", "eval_forecaster", "train_model", "locustfile_perm"]):
                # Avoid flagging child processes if already ours
                competing.append(f"{p.info['name']} (PID {p.pid})")
        except Exception:
            pass

    if competing:
        log.warning("Notice: potentially active background processes: %s", competing[:5])
    else:
        log.info("System check passed: no competing CPU-heavy processes running.")


def is_run_completed(csv_path: str, expected_dur: float) -> bool:
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Completed if rows span at least 85% of expected duration
        if len(rows) >= int(expected_dur * 0.85):
            return True
    except Exception:
        pass
    return False


def run_collection(
    generator: str = "locust",
    load: int = 7,
    runs_dir_override: Optional[str] = None,
    profiles_override: Optional[List[str]] = None,
    seeds_override: Optional[List[int]] = None,
):
    if runs_dir_override:
        runs_dir = runs_dir_override
    else:
        runs_dir = os.path.join(PROJECT_ROOT, "data", f"runs_{generator}" if generator == "locust" else "runs")

    os.makedirs(runs_dir, exist_ok=True)
    manifest_path = os.path.join(runs_dir, "manifest.json")

    # Build planned run list: 6 fixed profiles x 3 seeds + random_mix x 30 seeds
    planned_runs = []
    if profiles_override and seeds_override:
        for p in profiles_override:
            for s in seeds_override:
                planned_runs.append((p, s))
    else:
        for p in FIXED_PROFILES:
            for s in FIXED_SEEDS:
                planned_runs.append((p, s))
        for s in RANDOM_MIX_SEEDS:
            planned_runs.append(("random_mix", s))

    all_run_specs = []
    total_expected_sec = 0.0
    pending_expected_sec = 0.0

    for prof, s in planned_runs:
        dur = get_profile_duration(prof, s)
        csv_file = os.path.join(runs_dir, f"{prof}_seed{s}.csv")
        completed = is_run_completed(csv_file, dur)
        all_run_specs.append({
            "generator": generator,
            "profile": prof,
            "seed": s,
            "duration": dur,
            "csv_path": csv_file,
            "completed": completed,
        })
        total_expected_sec += dur
        if not completed:
            pending_expected_sec += dur

    pending_runs = [r for r in all_run_specs if not r["completed"]]

    print("=" * 75)
    print(f"PERM DATASET COLLECTION — Generator: {generator.upper()} (Fixed 8 Workers)")
    print("=" * 75)
    n_fixed = len([r for r in all_run_specs if r["profile"] != "random_mix"])
    n_mix = len([r for r in all_run_specs if r["profile"] == "random_mix"])
    print(f"Total planned runs:    {len(all_run_specs)} ({n_fixed} fixed + {n_mix} random_mix)")
    print(f"Already completed:     {len(all_run_specs) - len(pending_runs)}")
    print(f"Pending runs to run:   {len(pending_runs)}")
    print(f"Total planned runtime: {total_expected_sec:.0f}s ({total_expected_sec / 60.0:.1f} mins / {total_expected_sec / 3600.0:.2f} hours)")
    print(f"Expected runtime now:  {pending_expected_sec:.0f}s ({pending_expected_sec / 60.0:.1f} mins / {pending_expected_sec / 3600.0:.2f} hours)")
    print("=" * 75)

    if not pending_runs:
        log.info("All %d runs are already completed. Updating manifest...", len(all_run_specs))
        build_manifest(all_run_specs, manifest_path, generator)
        return

    # Start telemetry collector with 8 workers pinned to cores [4-15]
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "2.0"

    log.info("Starting telemetry collector with 8 workers on port 5000...")
    collector_proc = subprocess.Popen(
        [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "8"],
        env=env,
        cwd=PROJECT_ROOT,
    )

    try:
        if not _wait_healthy(f"{COLLECTOR_URL}/health", timeout=30.0):
            raise RuntimeError("Telemetry collector failed to start on port 5000")

        # Guarantee capacity fixed at 8
        requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=5.0)
        time.sleep(2.0)

        flask_ps = psutil.Process(collector_proc.pid)
        try:
            flask_ps.cpu_percent(interval=None)
        except Exception:
            pass

        for idx, run_spec in enumerate(pending_runs, 1):
            prof = run_spec["profile"]
            seed = run_spec["seed"]
            dur = run_spec["duration"]
            csv_path = run_spec["csv_path"]
            run_id = f"{prof}_seed{seed}"

            # Pre-run check
            check_no_competing_workload()

            log.info(
                "[%d/%d] Starting run %s (%s, seed=%d, dur=%.1fs)...",
                idx,
                len(pending_runs),
                run_id,
                prof,
                seed,
                dur,
            )

            # Ensure capacity is strictly 8
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=3.0)

            offsets, burst_offset, total_dur = burst_test.generate_schedule(prof, seed)

            collected_rows: List[Dict] = []
            stop_sampling = threading.Event()
            t0_mono = time.monotonic()
            traffic_start_mono: Optional[float] = None
            locust_proc: Optional[subprocess.Popen] = None
            locust_ps: Optional[psutil.Process] = None

            def _sample_loop():
                nonlocal traffic_start_mono, locust_ps
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

                        arr_cnt = int(m.get("arrivals_count", 0))
                        if traffic_start_mono is None and arr_cnt > 0:
                            traffic_start_mono = now_mono

                        elapsed = round(now_mono - (traffic_start_mono or t0_mono), 2)
                        burst_rel = round(elapsed - burst_offset, 2)

                        # Process CPU measurements
                        try:
                            f_cpu = flask_ps.cpu_percent(interval=None)
                        except Exception:
                            f_cpu = 0.0

                        # Worker CPU from collector telemetry
                        w_cpu = float(m.get("worker_cpu_percent", 0.0))
                        if w_cpu <= 0.0:
                            act_pct = float(m.get("worker_cpu_per_active_pct", 0.0))
                            act_w = int(m.get("active_workers", 8))
                            if act_pct > 0.0:
                                w_cpu = round((act_pct * act_w) / 16.0, 2)

                        # Locust CPU
                        lg_cpu = 0.0
                        if generator == "locust":
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

                        sys_cpu = float(m.get("cpu_percent", 0.0))
                        other_cpu = max(0.0, sys_cpu - w_cpu - (f_cpu / 16.0) - (lg_cpu / 16.0))

                        row = {
                            "run_id": run_id,
                            "generator": generator,
                            "profile": prof,
                            "seed": seed,
                            "burst_start_sec": round(burst_offset, 2),
                            "timestamp": ts,
                            "elapsed_sec": elapsed,
                            "burst_start_rel_sec": burst_rel,
                            "active_workers": m.get("active_workers", 8),
                            "queue_depth": m.get("queue_depth", 0),
                            "in_flight": m.get("in_flight", 0),
                            "arrivals_count": arr_cnt,
                            "ok_count": m.get("ok_count", 0),
                            "rejected_count": m.get("rejected_count", 0),
                            "timeout_count": m.get("timeout_count", 0),
                            "p95_latency_ms": m.get("p95_latency_ms", 0.0),
                            "ok_p95_latency_ms": m.get("ok_p95_latency_ms", 0.0),
                            "cpu_percent": sys_cpu,
                            "worker_cpu_percent": round(w_cpu, 2),
                            "busy_workers": m.get("busy_workers", 0),
                            "utilization": m.get("utilization", 0.0),
                            "worker_cpu_per_active_pct": m.get("worker_cpu_per_active_pct", 0.0),
                            "queue_wait_ms_avg": m.get("queue_wait_ms_avg", 0.0),
                            "cpu_freq_mhz": m.get("cpu_freq_mhz", 0.0),
                            "flask_cpu_pct": round(f_cpu, 2),
                            "locust_cpu_pct": round(lg_cpu, 2),
                            "workers_cpu_pct": round(w_cpu, 2),
                            "other_cpu_pct": round(other_cpu, 2),
                        }
                        collected_rows.append(row)

            sampler_th = threading.Thread(target=_sample_loop, daemon=True)
            sampler_th.start()

            # Launch generator
            if generator == "locust":
                loc_env = os.environ.copy()
                loc_env["PERM_PROFILE"] = prof
                loc_env["PERM_SEED"] = str(seed)
                loc_env["PERM_LOAD"] = str(load)
                loc_env["PERM_TIMEOUT"] = "3.0"

                csv_prefix = os.path.join(runs_dir, f"{run_id}_locust_stats")

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
                        f"{int(dur + 6)}s",
                        "--host",
                        COLLECTOR_URL,
                        "--csv",
                        csv_prefix,
                    ],
                    env=loc_env,
                    cwd=PROJECT_ROOT,
                )

                try:
                    p_loc = psutil.Process(locust_proc.pid)
                    p_loc.cpu_affinity([0, 1, 2, 3])
                except Exception:
                    pass

                locust_proc.wait(timeout=dur + 25)

            elif generator == "burst_test":
                burst_test.run_burst(
                    target_url=f"{COLLECTOR_URL}/api/process",
                    profile=prof,
                    seed=seed,
                    load=load,
                    timeout_sec=3.0,
                )

            # Allow trailing telemetry
            time.sleep(2.0)
            stop_sampling.set()
            sampler_th.join(timeout=3.0)

            # Filter valid run rows
            valid_rows = [r for r in collected_rows if 0.0 <= r["elapsed_sec"] <= dur + 2.0]
            if not valid_rows:
                valid_rows = collected_rows

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                writer.writerows(valid_rows)

            log.info("Completed run %s: logged %d rows to %s", run_id, len(valid_rows), os.path.basename(csv_path))
            run_spec["completed"] = True
            build_manifest(all_run_specs, manifest_path, generator)

            time.sleep(2.0)

    finally:
        _kill_proc(collector_proc)
        log.info("Collection session ended. Collector stopped.")


def build_manifest(all_run_specs: List[Dict], manifest_path: str, generator: str):
    """Rebuild manifest.json with target vs achieved rate, CPU in % and threads, queue, failures."""
    NUM_MACHINE_THREADS = 16
    runs_dir = os.path.dirname(manifest_path)

    manifest_records = []
    for spec in all_run_specs:
        csv_path = spec["csv_path"]
        prof = spec["profile"]
        s = spec["seed"]
        dur = spec["duration"]
        run_id = f"{prof}_seed{s}"

        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                arrivals = [int(r.get("arrivals_count", 0)) for r in rows]
                total_arr = sum(arrivals)
                total_ok = sum(int(r.get("ok_count", 0)) for r in rows)
                total_rej = sum(int(r.get("rejected_count", 0)) for r in rows)
                total_to = sum(int(r.get("timeout_count", 0)) for r in rows)
                failed = total_rej + total_to

                max_q = max((int(r.get("queue_depth", 0)) for r in rows), default=0)

                # Target schedule
                offsets, b_off, _ = burst_test.generate_schedule(prof, s)
                target_total = len(offsets)
                mean_target_rate = round(target_total / dur, 2)
                legacy_achieved_rate = round(total_arr / max(1.0, len(rows)), 2)
                legacy_bias_pct = round(((legacy_achieved_rate - mean_target_rate) / mean_target_rate) * 100.0, 2)
                mean_achieved_rate = round(total_arr / dur, 2)
                bias_pct = round(((mean_achieved_rate - mean_target_rate) / mean_target_rate) * 100.0, 2)

                # Step-response lag (if step profile)
                step_lag = None
                if prof == "step":
                    for r in rows:
                        el = float(r.get("elapsed_sec", 0.0))
                        if el >= 30.0 and int(r.get("arrivals_count", 0)) >= 36:
                            step_lag = round(el - 30.0, 2)
                            break

                # F5. Detrended Fano factor on residuals (observed - target per second)
                max_sec = int(np.ceil(dur)) + 2
                target_counts_sec = [0] * max_sec
                for off in offsets:
                    sec = int(np.floor(off))
                    if 0 <= sec < max_sec:
                        target_counts_sec[sec] += 1
                residuals = []
                target_vals = []
                for r in rows:
                    el = int(np.floor(float(r.get("elapsed_sec", 0.0))))
                    arr = int(r.get("arrivals_count", 0))
                    if 0 <= el < len(target_counts_sec):
                        tgt = target_counts_sec[el]
                        residuals.append(arr - tgt)
                        target_vals.append(tgt)
                detrended_var = float(np.var(residuals)) if residuals else 0.0
                mean_target_val = float(np.mean(target_vals)) if target_vals else 1.0
                detrended_fano = round(detrended_var / mean_target_val, 2) if mean_target_val > 0 else 0.0

                # Raw calm and burst Fano (legacy)
                calm_arr = [int(r["arrivals_count"]) for r in rows if 5.0 <= float(r.get("elapsed_sec", 0.0)) < 25.0]
                burst_arr = [int(r["arrivals_count"]) for r in rows if 35.0 <= float(r.get("elapsed_sec", 0.0)) < 85.0]
                calm_fano = round(float(np.var(calm_arr) / np.mean(calm_arr)), 2) if calm_arr and np.mean(calm_arr) > 0 else 0.0
                burst_fano = round(float(np.var(burst_arr) / np.mean(burst_arr)), 2) if burst_arr and np.mean(burst_arr) > 0 else 0.0

                # F2. Per-process CPU % normalized to % of machine
                flask_raw = [float(r.get("flask_cpu_pct", 0.0)) for r in rows]
                locust_raw = [float(r.get("locust_cpu_pct", 0.0)) for r in rows]
                workers_mach = [float(r.get("worker_cpu_percent", 0.0)) for r in rows]
                sys_mach = [float(r.get("cpu_percent", 0.0)) for r in rows]

                flask_mach = [c / float(NUM_MACHINE_THREADS) for c in flask_raw]
                locust_mach = [c / float(NUM_MACHINE_THREADS) for c in locust_raw]
                other_mach = [max(0.0, s - w - f - l) for s, w, f, l in zip(sys_mach, workers_mach, flask_mach, locust_mach)]
                nonworker_mach = [max(0.0, s - w) for s, w in zip(sys_mach, workers_mach)]

                flask_mean = round(float(np.mean(flask_mach)), 2) if flask_mach else 0.0
                locust_mean = round(float(np.mean(locust_mach)), 2) if locust_mach else 0.0
                workers_mean = round(float(np.mean(workers_mach)), 2) if workers_mach else 0.0
                other_mean = round(float(np.mean(other_mach)), 2) if other_mach else 0.0

                sorted_nwk = sorted(nonworker_mach)
                p95_idx = int(len(sorted_nwk) * 0.95)
                nonworker_p95 = round(sorted_nwk[min(p95_idx, len(sorted_nwk) - 1)], 2) if sorted_nwk else 0.0

                # Thread equivalents: mach_pct / 100 * 16 (or raw_core_pct / 100)
                flask_threads = round(float(np.mean(flask_raw)) / 100.0, 2) if flask_raw else 0.0
                locust_threads = round(float(np.mean(locust_raw)) / 100.0, 2) if locust_raw else 0.0
                workers_threads = round(workers_mean / 100.0 * NUM_MACHINE_THREADS, 2)
                other_threads = round(other_mean / 100.0 * NUM_MACHINE_THREADS, 2)

                # Contamination flag: other CPU > 10% or nonworker p95 > 25% or failures > 0
                flagged = (other_mean > 10.0) or (nonworker_p95 > 25.0) or (failed > 0)

                # Read Locust stats CSV if present
                locust_stats_file = os.path.join(runs_dir, f"{run_id}_locust_stats_stats.csv")
                locust_sent_count = None
                if os.path.exists(locust_stats_file):
                    try:
                        with open(locust_stats_file, "r", encoding="utf-8") as lsf:
                            for lrow in csv.DictReader(lsf):
                                if lrow.get("Name") == "Aggregated" or lrow.get("Type") == "POST":
                                    locust_sent_count = int(lrow.get("Request Count", 0))
                                    break
                    except Exception:
                        pass

                manifest_records.append({
                    "run_id": run_id,
                    "generator": generator,
                    "profile": prof,
                    "seed": s,
                    "duration_sec": dur,
                    "csv_file": os.path.basename(csv_path),
                    "rows_logged": len(rows),
                    "target_rate_mean_rps": mean_target_rate,
                    "legacy_achieved_rate_mean_rps": legacy_achieved_rate,
                    "legacy_rate_bias_pct": legacy_bias_pct,
                    "achieved_rate_mean_rps": mean_achieved_rate,
                    "rate_bias_pct": bias_pct,
                    "step_lag_sec": step_lag,
                    "detrended_fano_var_mean": detrended_fano,
                    "calm_fano_var_mean": calm_fano,
                    "burst_fano_var_mean": burst_fano,
                    "scheduled_count": target_total,
                    "locust_sent_count": locust_sent_count if locust_sent_count is not None else total_arr,
                    "total_arrivals": total_arr,
                    "total_ok": total_ok,
                    "failed_requests": failed,
                    "max_queue_depth": max_q,
                    "cpu_percent": {
                        "flask": flask_mean,
                        "workers": workers_mean,
                        "locust": locust_mean,
                        "other": other_mean,
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
                })
            except Exception as e:
                log.warning("Could not parse %s for manifest: %s", csv_path, e)
        else:
            manifest_records.append({
                "run_id": run_id,
                "generator": generator,
                "profile": prof,
                "seed": s,
                "duration_sec": dur,
                "csv_file": os.path.basename(csv_path),
                "rows_logged": 0,
                "completed": False,
            })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": generator,
        "target_capacity_workers": 8,
        "total_runs": len(all_run_specs),
        "completed_runs": sum(1 for m in manifest_records if m.get("completed")),
        "runs": manifest_records,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Saved manifest to %s (%d/%d completed)", manifest_path, payload["completed_runs"], payload["total_runs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect PERM dataset with fixed 8 workers")
    parser.add_argument("--generator", choices=["locust", "burst_test"], default="locust", help="Traffic generator to use")
    parser.add_argument("--profiles", nargs="+", default=None, help="Specific profiles to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Specific seeds to run")
    args = parser.parse_args()

    run_collection(
        generator=args.generator,
        profiles_override=args.profiles,
        seeds_override=args.seeds,
    )
