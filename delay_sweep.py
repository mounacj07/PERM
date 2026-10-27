"""
PERM - delay_sweep.py
=====================
Part A: Provisioning-delay sweep (baselines only).

Runs reactive and reactive_capacity_aware on step and ramp profiles
with PROVISION_DELAY_SEC in {0, 5, 15, 30}, 3 repeats each.

For each delay the telemetry_collector is started with PROVISION_DELAY_SEC
set in its environment. After a real OS spawn, the new worker stays
unavailable for that extra time (simulating config/warm-up).
ready_time is recorded in scale_events.

Output:
  - delay_sweep_results.csv   (per-run rows)
  - delay_sweep_summary.json  (aggregated table + UNVERIFIED section)
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
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

import burst_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("delay_sweep")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "delay_sweep_results.csv")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "delay_sweep_summary.json")

COLLECTOR_URL = "http://127.0.0.1:5000"

CSV_FIELDS = [
    "run_id", "mode", "delay_sec", "profile", "repeat", "seed",
    "success_rate", "failed_requests", "slo_pass",
    "worker_seconds", "cost_per_ok_request",
    "t_overload_elapsed_sec", "first_scaleup_elapsed_sec",
    "lead_time_sec",
    "max_workers_seen", "ready_time_sec",
    "slo_met_seconds_pct",
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


def _kill_proc(proc) -> None:
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


def _compute_t_overload(rows: List[Dict]) -> Optional[float]:
    window: deque = deque(maxlen=3)
    for r in rows:
        arr = float(r.get("arrivals_count", 0))
        window.append(arr)
        if len(window) < 3:
            continue
        smoothed = sum(window) / len(window)
        workers = max(1, int(r.get("active_workers", 1)))
        threshold = 0.9 * workers * 7.9
        if smoothed > threshold:
            return float(r.get("elapsed_sec", 0.0))
    return None


class _ReactiveScaler(threading.Thread):
    def __init__(self, url: str, max_workers: int = 8, min_workers: int = 2):
        super().__init__(daemon=True)
        self.url = url
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.stop_event = threading.Event()
        self.current_workers = min_workers
        self.calm_streak = 0
        self.scale_events: List[Dict] = []

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(1.0)
            try:
                r = requests.get(f"{self.url}/metrics/latest", timeout=1.5)
                if r.status_code != 200:
                    continue
                m = r.json()
            except Exception:
                continue
            p95 = m.get("p95_latency_ms", 0.0)
            depth = m.get("queue_depth", 0)
            in_flight = m.get("in_flight", 0)
            active = m.get("active_workers", self.current_workers)
            self.current_workers = active
            if (p95 > 300.0 or depth >= 4 or in_flight >= 6) and self.current_workers < self.max_workers:
                target = min(self.max_workers, self.current_workers + 2)
                t_now = time.monotonic()
                try:
                    requests.post(f"{self.url}/api/capacity", json={"active_workers": target}, timeout=5.0)
                    self.scale_events.append({"time_mono": t_now, "from_workers": active, "to_workers": target})
                    self.current_workers = target
                except Exception:
                    pass
                self.calm_streak = 0
            elif p95 < 200.0 and depth == 0 and in_flight <= 2 and self.current_workers > self.min_workers:
                self.calm_streak += 1
                if self.calm_streak >= 15:
                    target = max(self.min_workers, self.current_workers - 2)
                    try:
                        requests.post(f"{self.url}/api/capacity", json={"active_workers": target}, timeout=5.0)
                        self.current_workers = target
                    except Exception:
                        pass
                    self.calm_streak = 0
            else:
                self.calm_streak = 0


class _CapacityAwareScaler(threading.Thread):
    def __init__(self, url: str, max_workers: int = 8, min_workers: int = 2):
        super().__init__(daemon=True)
        self.url = url
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.stop_event = threading.Event()
        self.current_workers = min_workers
        self.smoothed_rps = 0.0
        self.calm_streak = 0
        self.scale_events: List[Dict] = []

    def run(self):
        alpha = 0.6
        while not self.stop_event.is_set():
            time.sleep(1.0)
            try:
                r = requests.get(f"{self.url}/metrics/latest", timeout=1.5)
                if r.status_code != 200:
                    continue
                m = r.json()
            except Exception:
                continue
            arr = float(m.get("arrivals_count", 0.0))
            active = int(m.get("active_workers", self.current_workers))
            self.current_workers = active
            self.smoothed_rps = arr if self.smoothed_rps == 0.0 else alpha * arr + (1.0 - alpha) * self.smoothed_rps
            raw_target = math.ceil(self.smoothed_rps * 0.126 / 0.7)
            target = max(self.min_workers, min(self.max_workers, raw_target))
            t_now = time.monotonic()
            if target > self.current_workers:
                try:
                    requests.post(f"{self.url}/api/capacity", json={"active_workers": target}, timeout=5.0)
                    self.scale_events.append({"time_mono": t_now, "from_workers": active, "to_workers": target})
                    self.current_workers = target
                except Exception:
                    pass
                self.calm_streak = 0
            elif target < self.current_workers:
                self.calm_streak += 1
                if self.calm_streak >= 10:
                    try:
                        requests.post(f"{self.url}/api/capacity", json={"active_workers": target}, timeout=5.0)
                        self.current_workers = target
                    except Exception:
                        pass
                    self.calm_streak = 0
            else:
                self.calm_streak = 0


def score_segment(rows: List[Dict]) -> Dict:
    total_arrivals = sum(int(r.get("arrivals_count", 0)) for r in rows)
    total_ok = sum(int(r.get("ok_count", 0)) for r in rows)
    total_rej = sum(int(r.get("rejected_count", 0)) for r in rows)
    total_to = sum(int(r.get("timeout_count", 0)) for r in rows)
    failed = total_rej + total_to
    total_finished = total_ok + failed
    success_rate = round(total_ok / total_finished * 100.0, 2) if total_finished > 0 else 100.0
    slo_pass = bool(success_rate >= 99.0)
    worker_seconds = sum(int(r.get("active_workers", 0)) for r in rows)
    cost_per_ok = round(worker_seconds / total_ok, 3) if total_ok > 0 else 0.0
    secs_with_arr = [r for r in rows if int(r.get("arrivals_count", 0)) > 0]
    slo_met_secs = sum(
        1 for r in secs_with_arr
        if (int(r.get("rejected_count", 0)) + int(r.get("timeout_count", 0))) / max(1, int(r.get("arrivals_count", 1))) <= 0.01
        and float(r.get("ok_p95_latency_ms", 0.0)) <= 300.0
    )
    slo_met_pct = round(slo_met_secs / len(secs_with_arr) * 100.0, 2) if secs_with_arr else 100.0
    return {
        "success_rate": success_rate,
        "failed_requests": failed,
        "slo_pass": slo_pass,
        "worker_seconds": worker_seconds,
        "cost_per_ok_request": cost_per_ok,
        "slo_met_seconds_pct": slo_met_pct,
        "total_arrivals": total_arrivals,
        "total_ok": total_ok,
    }


def run_one(mode: str, delay_sec: int, profile: str, repeat: int, seed: int) -> Dict:
    run_id = f"{mode}_delay{delay_sec}_{profile}_rep{repeat}_{int(time.time())}"
    log.info("=== %s (mode=%s, delay=%ds, profile=%s, repeat=%d, seed=%d) ===",
             run_id, mode, delay_sec, profile, repeat, seed)

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "2.0"
    env["PROVISION_DELAY_SEC"] = str(delay_sec)

    collector_proc = subprocess.Popen(
        [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "2"],
        env=env, cwd=PROJECT_ROOT,
    )

    scaler = None
    collected_rows: List[Dict] = []

    try:
        if not _wait_healthy(f"{COLLECTOR_URL}/health", timeout=30.0):
            raise RuntimeError("Collector did not start")
        requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=3.0)

        if mode == "reactive":
            scaler = _ReactiveScaler(COLLECTOR_URL)
        else:
            scaler = _CapacityAwareScaler(COLLECTOR_URL)
        scaler.start()

        time.sleep(2.0)
        stop_sampler = threading.Event()
        t0_mono = time.monotonic()
        burst_start_info: Dict = {"mono": None}

        def _poller():
            last_ts = None
            while not stop_sampler.is_set():
                now_mono = time.monotonic()
                elapsed = now_mono - t0_mono
                b_mono = burst_start_info["mono"]
                rel_burst = (now_mono - b_mono) if b_mono else 0.0
                try:
                    r = requests.get(f"{COLLECTOR_URL}/metrics/latest", timeout=1.0)
                    if r.status_code == 200:
                        m = r.json()
                        ts = m.get("timestamp")
                        if ts and ts != last_ts:
                            last_ts = ts
                            collected_rows.append({
                                "elapsed_sec": round(elapsed, 2),
                                "burst_start_rel_sec": round(rel_burst, 2),
                                "active_workers": m.get("active_workers", 2),
                                "queue_depth": m.get("queue_depth", 0),
                                "in_flight": m.get("in_flight", 0),
                                "arrivals_count": m.get("arrivals_count", 0),
                                "ok_count": m.get("ok_count", 0),
                                "rejected_count": m.get("rejected_count", 0),
                                "timeout_count": m.get("timeout_count", 0),
                                "ok_p95_latency_ms": m.get("ok_p95_latency_ms", 0.0),
                            })
                except Exception:
                    pass
                time.sleep(0.5)

        poller_th = threading.Thread(target=_poller, daemon=True)
        poller_th.start()

        def _on_burst_start(utc_dt: datetime, mono_t: float):
            burst_start_info["mono"] = mono_t

        burst_test.run_burst(
            target_url=f"{COLLECTOR_URL}/api/process",
            profile=profile,
            seed=seed,
            load=7,
            timeout_sec=3.0,
            on_burst_start=_on_burst_start,
        )

        time.sleep(2.5)
        stop_sampler.set()
        poller_th.join(timeout=3.0)

        cap_events: List[Dict] = []
        try:
            cap_resp = requests.get(f"{COLLECTOR_URL}/api/capacity", timeout=2.0).json()
            cap_events = cap_resp.get("scale_events", [])
        except Exception:
            pass

    finally:
        if scaler:
            scaler.stop_event.set()
        _kill_proc(collector_proc)
        time.sleep(1.5)

    score = score_segment(collected_rows)
    t_overload = _compute_t_overload(collected_rows)

    first_scaleup_elapsed: Optional[float] = None
    if scaler and scaler.scale_events:
        delta = scaler.scale_events[0]["time_mono"] - t0_mono
        first_scaleup_elapsed = round(delta, 2)

    lead_time = (
        round(t_overload - first_scaleup_elapsed, 2)
        if (t_overload is not None and first_scaleup_elapsed is not None)
        else None
    )

    max_workers_seen = max((int(r.get("active_workers", 0)) for r in collected_rows), default=0)

    ready_time_sec: Optional[float] = None
    for ev in cap_events:
        if ev.get("type") == "scale_up":
            ready_time_sec = ev.get("duration_seconds")
            break

    return {
        "run_id": run_id,
        "mode": mode,
        "delay_sec": delay_sec,
        "profile": profile,
        "repeat": repeat,
        "seed": seed,
        "success_rate": score["success_rate"],
        "failed_requests": score["failed_requests"],
        "slo_pass": score["slo_pass"],
        "worker_seconds": score["worker_seconds"],
        "cost_per_ok_request": score["cost_per_ok_request"],
        "t_overload_elapsed_sec": t_overload,
        "first_scaleup_elapsed_sec": first_scaleup_elapsed,
        "lead_time_sec": lead_time,
        "max_workers_seen": max_workers_seen,
        "ready_time_sec": ready_time_sec,
        "slo_met_seconds_pct": score["slo_met_seconds_pct"],
    }


def main():
    parser = argparse.ArgumentParser(description="PERM Part A - Provisioning Delay Sweep")
    parser.add_argument("--delays", type=str, default="0,5,15,30")
    parser.add_argument("--modes", type=str, default="reactive,reactive_capacity_aware")
    parser.add_argument("--profiles", type=str, default="step,ramp")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    delays = [int(d) for d in args.delays.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    profiles = [p.strip() for p in args.profiles.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    repeats = args.repeats

    total = len(delays) * len(modes) * len(profiles) * repeats
    print("=" * 70)
    print("PERM Part A: Provisioning Delay Sweep")
    print(f"Delays: {delays}s  Modes: {modes}  Profiles: {profiles}  Repeats: {repeats}")
    print(f"Total runs: {total} (approx {total * 130 / 60:.0f} min)")
    print("=" * 70)

    all_runs: List[Dict] = []

    for delay in delays:
        for prof in profiles:
            for mode in modes:
                for rep in range(1, repeats + 1):
                    seed = seeds[(rep - 1) % len(seeds)]
                    result = run_one(mode, delay, prof, rep, seed)
                    all_runs.append(result)

                    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(all_runs)

                    time.sleep(2.0)

    agg: Dict = {}
    for delay in delays:
        agg[delay] = {}
        for prof in profiles:
            agg[delay][prof] = {}
            for mode in modes:
                runs = [r for r in all_runs
                        if r["delay_sec"] == delay and r["profile"] == prof and r["mode"] == mode]
                if not runs:
                    continue
                sr_vals = [r["success_rate"] for r in runs]
                fail_vals = [r["failed_requests"] for r in runs]
                slo_passes = sum(1 for r in runs if r["slo_pass"])
                lt_vals = [r["lead_time_sec"] for r in runs if r["lead_time_sec"] is not None]
                rt_vals = [r["ready_time_sec"] for r in runs if r["ready_time_sec"] is not None]
                agg[delay][prof][mode] = {
                    "mean_success_rate": round(sum(sr_vals) / len(sr_vals), 2),
                    "mean_failed_requests": round(sum(fail_vals) / len(fail_vals), 1),
                    "slo_pass_count": f"{slo_passes}/{len(runs)}",
                    "slo_pass_all": slo_passes == len(runs),
                    "mean_lead_time_sec": round(sum(lt_vals) / len(lt_vals), 2) if lt_vals else None,
                    "mean_ready_time_sec": round(sum(rt_vals) / len(rt_vals), 2) if rt_vals else None,
                    "n_runs": len(runs),
                }

    modes_passing_all = set()
    for mode in modes:
        all_pass = all(
            agg.get(d, {}).get(p, {}).get(mode, {}).get("slo_pass_all", False)
            for d in delays for p in profiles
            if agg.get(d, {}).get(p, {}).get(mode)
        )
        if all_pass:
            modes_passing_all.add(mode)

    summary_rows = []
    for delay in delays:
        for prof in profiles:
            for mode in modes:
                a = agg.get(delay, {}).get(prof, {}).get(mode, {})
                if not a:
                    continue
                summary_rows.append({
                    "delay_sec": delay, "profile": prof, "mode": mode,
                    "success_%": a.get("mean_success_rate"),
                    "failed_reqs": a.get("mean_failed_requests"),
                    "SLO_pass": a.get("slo_pass_count"),
                    "lead_time_s": a.get("mean_lead_time_sec"),
                    "ready_time_s": a.get("mean_ready_time_sec"),
                })

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "delays_sec": delays,
            "modes": modes,
            "profiles": profiles,
            "repeats": repeats,
            "seeds": seeds[:repeats],
            "provision_delay_semantics": (
                "PROVISION_DELAY_SEC is applied AFTER OS-ready (ready_event.wait()). "
                "ready_time_sec = os_ready_time + provision_delay_sec. "
                "workers-ready time = first_scaleup_elapsed_sec + ready_time_sec."
            ),
        },
        "aggregated_table": agg,
        "summary_rows": summary_rows,
        "modes_passing_slo_all_delays_all_profiles": list(modes_passing_all),
        "cost_note": (
            "Cost is only compared between modes listed in "
            "modes_passing_slo_all_delays_all_profiles. "
            "Modes that fail SLO in any rep are excluded."
        ),
        "individual_runs": all_runs,
        "UNVERIFIED": [
            "Lead times at delay=0 are based on a single process-spawn cycle per run.",
            "At higher delays, reactive signal latency and PROVISION_DELAY_SEC both compound.",
            "Results may differ significantly on machines with fewer cores or VMs.",
        ],
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary to %s", SUMMARY_JSON)

    print("\n" + "=" * 95)
    print("PART A: PROVISIONING DELAY SWEEP RESULTS")
    print("=" * 95)
    print(f"{'delay_s':<8} {'profile':<8} {'mode':<26} {'success%':<10} {'failed':<8} {'SLO':<8} {'lead_t':<10} {'ready_t':<10}")
    print("-" * 95)
    for row in summary_rows:
        print(
            f"{row['delay_sec']:<8} {row['profile']:<8} {row['mode']:<26} "
            f"{str(row['success_%']):<10} {str(row['failed_reqs']):<8} "
            f"{str(row['SLO_pass']):<8} {str(row['lead_time_s']):<10} {str(row['ready_time_s']):<10}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
