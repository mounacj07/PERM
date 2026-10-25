"""
PERM - compare_scaling.py
=========================
Step 2 Scaling comparison runner across 5 scaling modes:
  1. fixed_2:                  Fixed 2 workers (baseline edge provisioning)
  2. fixed_8:                  Fixed 8 workers (overprovisioned static ceiling)
  3. reactive:                 Queue/latency threshold reactive autoscaler (+2 / -2)
  4. reactive_capacity_aware:  target = clip(ceil(smoothed_rps * 0.126 / 0.7), 2, 8), jumps directly
  5. perm:                     PERM predictive system (forecast-driven proactive scaling)

Evaluated across profiles: step, ramp, spike (3 repeats each: seeds 42, 43, 44).
Scored using the Step 2 honest metric set:
  - slo_met(second) = failed_share (rejected+timeout)/arrivals <= 1% AND ok_p95 <= 300 ms.
  - Both whole_run and burst_window (burst start to burst end + 30s) reported.
  - success_rate, failed_requests, SLO_PASS (success >= 99%), worker_seconds, cost_per_ok_request,
    time_to_adequate_capacity, time_to_first_scale_up, and alert lead times (burst_start - alert_time).

Outputs:
  - step2_comparison.csv: Per-run, per-second telemetry rows
  - step2_summary.json: Summary statistics and sensitivity tables
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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

import burst_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare_scaling")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPARISON_CSV = os.path.join(PROJECT_ROOT, "step2_comparison.csv")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "step2_summary.json")
ALERTS_LOG_PATH = os.path.join(PROJECT_ROOT, "alerts_log.csv")

COLLECTOR_URL = "http://127.0.0.1:5000"
INFERENCE_URL = "http://127.0.0.1:5001"
ALERT_URL = "http://127.0.0.1:5002"
REMEDIATION_URL = "http://127.0.0.1:5003"

CSV_ROW_FIELDS = [
    "run_id",
    "mode",
    "repeat",
    "seed",
    "profile",
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


class ReactiveAutoscaler(threading.Thread):
    """
    Original reactive autoscaler:
    Monitors collector metrics every 1s.
    Scales UP by +2 workers when p95 > 300ms OR queue_depth >= 4 OR in_flight >= 6.
    Scales DOWN by -2 workers when p95 < 200ms AND queue_depth == 0 for 15s.
    """
    def __init__(self, collector_url: str, max_workers: int = 8, min_workers: int = 2):
        super().__init__(daemon=True, name="reactive_autoscaler")
        self.collector_url = collector_url
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.stop_event = threading.Event()
        self.current_workers = min_workers
        self.calm_streak = 0
        self.scale_events = []

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(1.0)
            try:
                r = requests.get(f"{self.collector_url}/metrics/latest", timeout=1.5)
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

            # Scale up trigger
            if (p95 > 300.0 or depth >= 4 or in_flight >= 6) and self.current_workers < self.max_workers:
                target = min(self.max_workers, self.current_workers + 2)
                t_scale = time.monotonic()
                try:
                    requests.post(
                        f"{self.collector_url}/api/capacity",
                        json={"active_workers": target},
                        timeout=5.0,
                    )
                    self.current_workers = target
                    self.scale_events.append({
                        "time_mono": t_scale,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "from_workers": active,
                        "to_workers": target,
                        "reason": f"p95={p95:.1f}ms depth={depth} in_flight={in_flight}",
                    })
                    log.info("Reactive scale-up: %d -> %d workers (p95=%.1fms, depth=%d)", active, target, p95, depth)
                except Exception as exc:
                    log.warning("Reactive scale-up request error: %s", exc)
                self.calm_streak = 0

            # Scale down trigger
            elif p95 < 200.0 and depth == 0 and in_flight <= 2 and self.current_workers > self.min_workers:
                self.calm_streak += 1
                if self.calm_streak >= 15:
                    target = max(self.min_workers, self.current_workers - 2)
                    try:
                        requests.post(
                            f"{self.collector_url}/api/capacity",
                            json={"active_workers": target},
                            timeout=5.0,
                        )
                        self.current_workers = target
                        log.info("Reactive scale-down: %d -> %d workers", active, target)
                    except Exception as exc:
                        log.warning("Reactive scale-down request error: %s", exc)
                    self.calm_streak = 0
            else:
                self.calm_streak = 0


class ReactiveCapacityAwareAutoscaler(threading.Thread):
    """
    Capacity-aware reactive autoscaler (Part C):
    target = clip(ceil(smoothed_arrival_rate * 0.126 / 0.7), 2, 8).
    Jumps directly to target, experiencing real process spawn cold-start.
    """
    def __init__(self, collector_url: str, min_workers: int = 2, max_workers: int = 8):
        super().__init__(daemon=True, name="reactive_capacity_aware")
        self.collector_url = collector_url
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.stop_event = threading.Event()
        self.current_workers = min_workers
        self.smoothed_arrival_rate = 0.0
        self.scale_events = []
        self.calm_streak = 0

    def run(self):
        alpha = 0.6
        while not self.stop_event.is_set():
            time.sleep(1.0)
            try:
                r = requests.get(f"{self.collector_url}/metrics/latest", timeout=1.5)
                if r.status_code != 200:
                    continue
                m = r.json()
            except Exception:
                continue

            arr = float(m.get("arrivals_count", 0.0))
            active = int(m.get("active_workers", self.current_workers))
            self.current_workers = active

            # Update exponential moving average of arrival rate
            if self.smoothed_arrival_rate == 0.0:
                self.smoothed_arrival_rate = arr
            else:
                self.smoothed_arrival_rate = alpha * arr + (1.0 - alpha) * self.smoothed_arrival_rate

            raw_target = math.ceil(self.smoothed_arrival_rate * 0.126 / 0.7)
            target = max(self.min_workers, min(self.max_workers, raw_target))

            t_now = time.monotonic()
            if target > self.current_workers:
                try:
                    requests.post(
                        f"{self.collector_url}/api/capacity",
                        json={"active_workers": target},
                        timeout=5.0,
                    )
                    self.scale_events.append({
                        "time_mono": t_now,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "from_workers": active,
                        "to_workers": target,
                        "reason": f"smoothed_rps={self.smoothed_arrival_rate:.1f} target={target}",
                    })
                    log.info("Capacity-aware scale-up: %d -> %d (smoothed_rps=%.1f)", active, target, self.smoothed_arrival_rate)
                    self.current_workers = target
                except Exception as exc:
                    log.warning("Capacity-aware scale-up error: %s", exc)
                self.calm_streak = 0
            elif target < self.current_workers:
                self.calm_streak += 1
                if self.calm_streak >= 10:
                    try:
                        requests.post(
                            f"{self.collector_url}/api/capacity",
                            json={"active_workers": target},
                            timeout=5.0,
                        )
                        self.scale_events.append({
                            "time_mono": t_now,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "from_workers": active,
                            "to_workers": target,
                            "reason": f"smoothed_rps={self.smoothed_arrival_rate:.1f} scale_down",
                        })
                        log.info("Capacity-aware scale-down: %d -> %d (smoothed_rps=%.1f)", active, target, self.smoothed_arrival_rate)
                        self.current_workers = target
                    except Exception as exc:
                        log.warning("Capacity-aware scale-down error: %s", exc)
                    self.calm_streak = 0
            else:
                self.calm_streak = 0


def _start_perm_stack(env: dict) -> List[subprocess.Popen]:
    """Start the 4 PERM microservices and verify health."""
    procs = []
    services = [
        ("telemetry_collector", [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "2"], f"{COLLECTOR_URL}/health"),
        ("inference_service", [sys.executable, "inference_service.py", "--port", "5001", "--collector-url", COLLECTOR_URL], f"{INFERENCE_URL}/health"),
        ("alert_engine", [sys.executable, "alert_engine.py", "--port", "5002", "--inference-url", INFERENCE_URL], f"{ALERT_URL}/health"),
        ("remediation_service", [sys.executable, "remediation_service.py", "--port", "5003", "--alert-url", ALERT_URL, "--collector-url", COLLECTOR_URL], f"{REMEDIATION_URL}/health"),
    ]

    for name, cmd, health_url in services:
        p = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT)
        procs.append(p)
        if not _wait_healthy(health_url, timeout=30.0):
            for pr in procs:
                _kill_proc(pr)
            raise RuntimeError(f"Service {name} did not become healthy at {health_url}")

    return procs


def score_segment(rows: List[Dict]) -> Dict:
    """Score a subset of rows (whole run or burst window)."""
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

    # slo_met(second) = failed_share <= 1% AND ok_p95 <= 300 ms. Exclude seconds with no arrivals.
    secs_with_arr = [r for r in rows if int(r.get("arrivals_count", 0)) > 0]
    slo_met_secs = 0
    for r in secs_with_arr:
        arr = int(r.get("arrivals_count", 0))
        rej = int(r.get("rejected_count", 0))
        to = int(r.get("timeout_count", 0))
        ok_p95 = float(r.get("ok_p95_latency_ms", 0.0))
        failed_share = (rej + to) / arr
        if failed_share <= 0.01 and ok_p95 <= 300.0:
            slo_met_secs += 1

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


def run_single_experiment(
    mode: str,
    repeat_idx: int,
    seed: int,
    profile: str,
    burst_load: int = 7,
) -> Dict:
    run_id = f"{mode}_{profile}_rep{repeat_idx}_{int(time.time())}"
    log.info("=== STARTING RUN %s (mode=%s, profile=%s, repeat=%d, seed=%d) ===", run_id, mode, profile, repeat_idx, seed)

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "2.0"

    procs_to_kill: List[subprocess.Popen] = []
    reactive_scaler = None

    # Count prior data rows (not lines with header) to avoid slicing bug
    alerts_before_count = 0
    if os.path.exists(ALERTS_LOG_PATH):
        try:
            with open(ALERTS_LOG_PATH, "r", encoding="utf-8") as f:
                alerts_before_count = len(list(csv.DictReader(f)))
        except Exception:
            alerts_before_count = 0

    try:
        if mode == "fixed_2":
            p = subprocess.Popen(
                [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "2"],
                env=env, cwd=PROJECT_ROOT,
            )
            procs_to_kill.append(p)
            _wait_healthy(f"{COLLECTOR_URL}/health")
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=3.0)

        elif mode == "fixed_8":
            p = subprocess.Popen(
                [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "8"],
                env=env, cwd=PROJECT_ROOT,
            )
            procs_to_kill.append(p)
            _wait_healthy(f"{COLLECTOR_URL}/health")
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 8}, timeout=3.0)

        elif mode == "reactive":
            p = subprocess.Popen(
                [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "2"],
                env=env, cwd=PROJECT_ROOT,
            )
            procs_to_kill.append(p)
            _wait_healthy(f"{COLLECTOR_URL}/health")
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=3.0)
            reactive_scaler = ReactiveAutoscaler(COLLECTOR_URL, max_workers=8, min_workers=2)
            reactive_scaler.start()

        elif mode == "reactive_capacity_aware":
            p = subprocess.Popen(
                [sys.executable, "telemetry_collector.py", "--port", "5000", "--workers", "2"],
                env=env, cwd=PROJECT_ROOT,
            )
            procs_to_kill.append(p)
            _wait_healthy(f"{COLLECTOR_URL}/health")
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=3.0)
            reactive_scaler = ReactiveCapacityAwareAutoscaler(COLLECTOR_URL, max_workers=8, min_workers=2)
            reactive_scaler.start()

        elif mode == "perm":
            procs_to_kill = _start_perm_stack(env)
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=3.0)
            try:
                requests.post(f"{REMEDIATION_URL}/api/reset", timeout=2.0)
            except Exception:
                pass

        time.sleep(2.0)

        collected_rows: List[Dict] = []
        stop_sampler = threading.Event()
        t0_mono = time.monotonic()
        burst_start_info: Dict[str, Optional[object]] = {"mono": None, "utc": None}

        # Telemetry poller
        def _telemetry_poller():
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
                            row = {
                                "run_id": run_id,
                                "mode": mode,
                                "repeat": repeat_idx,
                                "seed": seed,
                                "profile": profile,
                                "timestamp": ts,
                                "elapsed_sec": round(elapsed, 2),
                                "burst_start_rel_sec": round(rel_burst, 2),
                                "active_workers": m.get("active_workers", 2),
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
                except Exception:
                    pass
                time.sleep(0.5)

        poller = threading.Thread(target=_telemetry_poller, daemon=True)
        poller.start()

        def _on_burst_start_cb(utc_dt: datetime, mono_t: float):
            burst_start_info["mono"] = mono_t
            burst_start_info["utc"] = utc_dt

        # Run traffic generator
        burst_res = burst_test.run_burst(
            target_url=f"{COLLECTOR_URL}/api/process",
            profile=profile,
            seed=seed,
            load=burst_load,
            timeout_sec=3.0,
            on_burst_start=_on_burst_start_cb,
        )

        time.sleep(2.5)
        stop_sampler.set()
        poller.join(timeout=3.0)

        scale_events = []
        try:
            cap_resp = requests.get(f"{COLLECTOR_URL}/api/capacity", timeout=2.0).json()
            scale_events = cap_resp.get("scale_events", [])
        except Exception:
            pass

    finally:
        if reactive_scaler:
            reactive_scaler.stop_event.set()
        for p in procs_to_kill:
            _kill_proc(p)
        time.sleep(1.5)

    # PERM alerts extraction
    perm_scale_events = []
    if mode == "perm" and os.path.exists(ALERTS_LOG_PATH):
        try:
            with open(ALERTS_LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_alerts = list(reader)
                matching_alert_rows = all_alerts[alerts_before_count:]
        except Exception as exc:
            log.warning("Could not read alerts_log.csv: %s", exc)
            matching_alert_rows = []

        burst_start_dt = burst_start_info["utc"]

        for alert in matching_alert_rows:
            alert_ts_str = alert.get("timestamp", "")
            try:
                alert_dt = datetime.fromisoformat(alert_ts_str.replace("Z", "+00:00"))
                lead_time_sec = round((burst_start_dt.timestamp() - alert_dt.timestamp()), 2) if burst_start_dt else 0.0
            except Exception:
                lead_time_sec = 0.0

            cur_v = float(alert.get("current_value", 0.0))
            fc_v = float(alert.get("forecast_value_30s", 0.0))
            slo_th = float(alert.get("slo_threshold", 0.0))

            provenance = "FORECAST" if (fc_v >= slo_th and cur_v < slo_th) else "CURRENT"

            perm_scale_events.append({
                "alert_id": alert.get("alert_id"),
                "provenance": provenance,
                "metric": alert.get("metric"),
                "current_value": cur_v,
                "forecast_value_30s": fc_v,
                "slo_threshold": slo_th,
                "lead_time_sec": lead_time_sec,
                "is_before_burst": lead_time_sec > 0,
            })

    # Scoring calculations (Part A)
    # Burst window definition: burst_start to burst_end + 30s
    _, burst_start_nominal, total_dur_nominal = burst_test.generate_schedule(profile, seed)
    # Nominal burst duration:
    if profile == "step":
        burst_dur = 60.0
    elif profile == "ramp":
        burst_dur = 75.0
    elif profile == "spike":
        burst_dur = 15.0
    else:
        burst_dur = 60.0

    burst_window_rows = [
        r for r in collected_rows
        if 0.0 <= r["burst_start_rel_sec"] <= (burst_dur + 30.0)
    ]

    whole_run_score = score_segment(collected_rows)
    burst_window_score = score_segment(burst_window_rows)

    # Time to adequate capacity: first second when active_workers * 7.9 >= arrival rate
    time_to_adequate_capacity = None
    for r in collected_rows:
        arr = int(r.get("arrivals_count", 0))
        active = int(r.get("active_workers", 0))
        if arr > 15:  # under burst load
            if active * 7.9 >= arr:
                time_to_adequate_capacity = round(r["burst_start_rel_sec"], 2)
                break

    # Time to first scale up relative to burst start (negative = before)
    time_to_first_scale_up = None
    burst_mono = burst_start_info["mono"]

    if mode == "fixed_2":
        time_to_first_scale_up = None
    elif mode == "fixed_8":
        time_to_first_scale_up = round(-burst_start_nominal, 2)
    elif mode in ["reactive", "reactive_capacity_aware"]:
        if reactive_scaler and reactive_scaler.scale_events and burst_mono:
            first_ev = reactive_scaler.scale_events[0]
            time_to_first_scale_up = round(first_ev["time_mono"] - burst_mono, 2)
    elif mode == "perm":
        scale_ups = [ev for ev in scale_events if ev.get("type") == "scale_up"]
        if scale_ups and burst_start_info["utc"]:
            try:
                first_ev = scale_ups[0]
                ev_dt = datetime.fromisoformat(first_ev["timestamp"].replace("Z", "+00:00"))
                time_to_first_scale_up = round(ev_dt.timestamp() - burst_start_info["utc"].timestamp(), 2)
            except Exception:
                pass

    run_summary = {
        "run_id": run_id,
        "mode": mode,
        "repeat": repeat_idx,
        "seed": seed,
        "profile": profile,
        "duration_sec": collected_rows[-1]["elapsed_sec"] if collected_rows else 0.0,
        "time_to_first_scale_up": time_to_first_scale_up,
        "time_to_adequate_capacity": time_to_adequate_capacity,
        "whole_run": whole_run_score,
        "burst_window": burst_window_score,
        "perm_scale_events": perm_scale_events,
    }

    return {"summary": run_summary, "rows": collected_rows}


def aggregate_mode_profile(summaries: List[Dict]) -> Dict:
    metrics = ["success_rate", "failed_requests", "worker_seconds", "cost_per_ok_request", "slo_met_seconds_pct"]
    res = {}
    for scope in ["whole_run", "burst_window"]:
        res[scope] = {}
        for m in metrics:
            vals = [s[scope][m] for s in summaries if s.get(scope, {}).get(m) is not None]
            if vals:
                res[scope][m] = {
                    "mean": round(sum(vals) / len(vals), 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                }
            else:
                res[scope][m] = {"mean": None, "min": None, "max": None}
        passes = sum(1 for s in summaries if s.get(scope, {}).get("slo_pass"))
        res[scope]["slo_pass_count"] = passes

    for timing in ["time_to_first_scale_up", "time_to_adequate_capacity"]:
        vals = [s[timing] for s in summaries if s.get(timing) is not None]
        if vals:
            res[timing] = {
                "mean": round(sum(vals) / len(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }
        else:
            res[timing] = {"mean": None, "min": None, "max": None}

    return res


def compute_sensitivity_table(all_summaries: List[Dict]) -> Dict:
    """Compute sensitivity table at 98%, 99%, and 99.9% thresholds."""
    thresholds = [98.0, 99.0, 99.9]
    table = {}
    for th in thresholds:
        th_key = f"{th:.1f}%"
        table[th_key] = {}
        # Group by mode
        by_mode: Dict[str, List[Dict]] = {}
        for s in all_summaries:
            m = s["mode"]
            by_mode.setdefault(m, []).append(s)

        for m, m_runs in by_mode.items():
            pass_count = sum(1 for r in m_runs if r["whole_run"]["success_rate"] >= th)
            avg_cost = round(sum(r["whole_run"]["worker_seconds"] for r in m_runs) / len(m_runs), 1) if m_runs else 0.0
            avg_sr = round(sum(r["whole_run"]["success_rate"] for r in m_runs) / len(m_runs), 2) if m_runs else 0.0
            table[th_key][m] = {
                "passes": f"{pass_count}/{len(m_runs)}",
                "avg_worker_seconds": avg_cost,
                "avg_success_rate": avg_sr,
            }
    return table


def run_all_experiments(
    modes: List[str] = ["fixed_2", "fixed_8", "reactive", "reactive_capacity_aware", "perm"],
    profiles: List[str] = ["step", "ramp", "spike"],
    repeats: int = 3,
    seeds: List[int] = [42, 43, 44],
    load: int = 7,
):
    print("=" * 80)
    print("PERM STEP 2 BENCHMARK RUNNER (5 Modes x 3 Profiles x 3 Repeats)")
    print("=" * 80)
    print(f"Modes:    {modes}")
    print(f"Profiles: {profiles}")
    print(f"Repeats:  {repeats} (Seeds: {seeds[:repeats]})")
    print(f"Total planned runs: {len(modes) * len(profiles) * repeats}")
    print("=" * 80)

    # Load existing runs if CSV exists to support resume
    completed_keys = set()
    existing_rows = []
    if os.path.exists(COMPARISON_CSV):
        try:
            with open(COMPARISON_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
            for r in existing_rows:
                key = (r["mode"], r.get("profile", "step"), int(r["repeat"]))
                completed_keys.add(key)
            log.info("Found %d completed run configurations in %s", len(completed_keys), COMPARISON_CSV)
        except Exception:
            existing_rows = []

    all_rows = list(existing_rows)
    all_summaries: List[Dict] = []
    perm_all_alerts: List[Dict] = []

    for prof in profiles:
        for mode in modes:
            for rep in range(1, repeats + 1):
                key = (mode, prof, rep)
                if key in completed_keys:
                    log.info("Skipping already completed run: mode=%s profile=%s repeat=%d", mode, prof, rep)
                    continue

                seed = seeds[rep - 1]
                res = run_single_experiment(
                    mode=mode,
                    repeat_idx=rep,
                    seed=seed,
                    profile=prof,
                    burst_load=load,
                )
                all_summaries.append(res["summary"])
                all_rows.extend(res["rows"])
                if res["summary"].get("perm_scale_events"):
                    perm_all_alerts.extend(res["summary"]["perm_scale_events"])

                # Incrementally save CSV
                with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_ROW_FIELDS)
                    writer.writeheader()
                    for r in all_rows:
                        writer.writerow(r)

                time.sleep(2.0)

    # -----------------------------------------------------------------------
    # Aggregation & Summary JSON
    # -----------------------------------------------------------------------
    summary_by_mode_profile: Dict[str, Dict[str, Dict]] = {}
    for prof in profiles:
        summary_by_mode_profile[prof] = {}
        for m in modes:
            matching = [s for s in all_summaries if s["profile"] == prof and s["mode"] == m]
            if matching:
                summary_by_mode_profile[prof][m] = aggregate_mode_profile(matching)

    sensitivity = compute_sensitivity_table(all_summaries)

    final_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "modes": modes,
        "repeats": repeats,
        "sensitivity_table": sensitivity,
        "summary_by_profile_and_mode": summary_by_mode_profile,
        "individual_runs": all_summaries,
        "perm_scale_up_alerts": perm_all_alerts,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    log.info("Saved summary to %s", SUMMARY_JSON)

    # -----------------------------------------------------------------------
    # Print Markdown Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 2 COMPARISON RESULTS (Honest Scoring)")
    print("=" * 90)
    headers = ["Profile", "Mode", "Run %OK", "Burst %OK", "SLO (>=99%)", "Cost (w-s)", "Cost/OK", "1st Scale", "Adequate Cap"]
    print(f"{headers[0]:<10} {headers[1]:<24} {headers[2]:<10} {headers[3]:<10} {headers[4]:<12} {headers[5]:<12} {headers[6]:<10} {headers[7]:<10} {headers[8]:<12}")
    print("-" * 115)

    for prof in profiles:
        for m in modes:
            agg = summary_by_mode_profile.get(prof, {}).get(m)
            if not agg:
                continue
            r_sr = agg["whole_run"]["success_rate"].get("mean", "N/A")
            b_sr = agg["burst_window"]["success_rate"].get("mean", "N/A")
            passes = f"{agg['whole_run']['slo_pass_count']}/{repeats}"
            ws = agg["whole_run"]["worker_seconds"].get("mean", "N/A")
            cp_ok = agg["whole_run"]["cost_per_ok_request"].get("mean", "N/A")
            sc = agg["time_to_first_scale_up"].get("mean", "N/A")
            ad = agg["time_to_adequate_capacity"].get("mean", "N/A")

            print(f"{prof:<10} {m:<24} {str(r_sr):<10} {str(b_sr):<10} {passes:<12} {str(ws):<12} {str(cp_ok):<10} {str(sc):<10} {str(ad):<12}")
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="PERM Step 2 comparison runner")
    parser.add_argument("--modes", type=str, default="fixed_2,fixed_8,reactive,reactive_capacity_aware,perm")
    parser.add_argument("--profiles", type=str, default="step,ramp,spike")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--load", type=int, default=7)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    run_all_experiments(modes=modes, profiles=profiles, repeats=args.repeats, seeds=seeds, load=args.load)


if __name__ == "__main__":
    main()
