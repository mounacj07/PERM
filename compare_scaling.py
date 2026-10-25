"""
PERM - compare_scaling.py
=========================
Comparison runner across 4 scaling modes under identical open-loop traffic bursts:
  1. fixed_2:     Fixed 2 workers (baseline edge provisioning)
  2. fixed_8:     Fixed 8 workers (overprovisioned static ceiling)
  3. reactive:    Simple reactive autoscaler (scales up when p95 > 300ms or queue depth is high,
                  using same scale step and real process spawn cold-start as PERM)
  4. perm:        PERM predictive system (telemetry collector, inference service,
                  proactive alert engine, and auto remediation)

Each mode is evaluated across 3 seeded repeats.
Outputs:
  - step1b_comparison.csv: Per-run, per-second telemetry rows
  - step1b_summary.json: Summary statistics per mode (mean, min, max) and PERM alert provenance

Usage:
    python compare_scaling.py [--modes fixed_2,fixed_8,reactive,perm] [--repeats 3] [--profile step]
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

import requests

import burst_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare_scaling")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPARISON_CSV = os.path.join(PROJECT_ROOT, "step1b_comparison.csv")
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "step1b_summary.json")
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
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            proc.kill()


class ReactiveAutoscaler(threading.Thread):
    """
    Simple reactive autoscaler:
    Monitors collector metrics every 1s.
    Scales UP by +2 workers when p95 > 300ms OR queue_depth >= 4 OR in_flight >= 6.
    Scales DOWN by -2 workers when p95 < 200ms AND queue_depth == 0 for 15s.
    Incurs the same real cold-start process spawn via POST /api/capacity.
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
                    res = requests.post(
                        f"{self.collector_url}/api/capacity",
                        json={"active_workers": target},
                        timeout=5.0,
                    ).json()
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


def run_single_experiment(
    mode: str,
    repeat_idx: int,
    seed: int,
    profile: str,
    burst_load: int = 7,
    stage_durations: Optional[Tuple[float, float, float]] = None,
) -> Dict:
    """
    Run one experimental evaluation for a given mode and repeat.
    Collects per-second metrics rows and computes run summary statistics.
    """
    run_id = f"{mode}_rep{repeat_idx}_{int(time.time())}"
    log.info("=== STARTING RUN %s (mode=%s, repeat=%d, seed=%d) ===", run_id, mode, repeat_idx, seed)

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MS_PER_LOAD"] = "18"
    env["QUEUE_CAPACITY"] = "20"
    env["REQUEST_TIMEOUT_SEC"] = "3.0"

    procs_to_kill: List[subprocess.Popen] = []
    reactive_scaler: Optional[ReactiveAutoscaler] = None

    # Baseline alerts row count for PERM provenance tracking
    alerts_before_count = 0
    if os.path.exists(ALERTS_LOG_PATH):
        try:
            with open(ALERTS_LOG_PATH, "r", encoding="utf-8") as f:
                alerts_before_count = sum(1 for _ in f)
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

        elif mode == "perm":
            procs_to_kill = _start_perm_stack(env)
            # Ensure starting with 2 workers in remediation & collector
            requests.post(f"{COLLECTOR_URL}/api/capacity", json={"active_workers": 2}, timeout=3.0)
            try:
                requests.post(f"{REMEDIATION_URL}/capacity/mode", json={"mode": "AUTO"}, timeout=2.0)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        time.sleep(2.0)  # let system settle and sampler initialize

        # Track per-second collector telemetry throughout traffic burst
        collected_rows: List[Dict] = []
        stop_sampler = threading.Event()
        burst_start_info = {"mono": 0.0, "utc": None}

        def _telemetry_poller():
            t0 = time.monotonic()
            while not stop_sampler.is_set():
                now_mono = time.monotonic()
                elapsed = now_mono - t0
                rel_burst = (now_mono - burst_start_info["mono"]) if burst_start_info["mono"] > 0 else 0.0

                try:
                    r = requests.get(f"{COLLECTOR_URL}/metrics/latest", timeout=1.0)
                    if r.status_code == 200:
                        m = r.json()
                        row = {
                            "run_id": run_id,
                            "mode": mode,
                            "repeat": repeat_idx,
                            "seed": seed,
                            "timestamp": m.get("timestamp", datetime.now(timezone.utc).isoformat()),
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
                        }
                        collected_rows.append(row)
                except Exception:
                    pass
                time.sleep(1.0)

        poller = threading.Thread(target=_telemetry_poller, daemon=True)
        poller.start()

        def _on_burst_start_cb(utc_dt: datetime, mono_t: float):
            burst_start_info["mono"] = mono_t
            burst_start_info["utc"] = utc_dt

        # Run open-loop traffic generator
        burst_res = burst_test.run_burst(
            target_url=f"{COLLECTOR_URL}/api/process",
            profile=profile,
            seed=seed,
            load=burst_load,
            on_burst_start=_on_burst_start_cb,
        )

        # Allow final window to drain into telemetry
        time.sleep(2.5)
        stop_sampler.set()
        poller.join(timeout=3.0)

        # Fetch scale events from collector
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

    # -----------------------------------------------------------------------
    # PERM Alert Provenance Extraction
    # -----------------------------------------------------------------------
    perm_scale_events = []
    matching_alert_rows = []

    if mode == "perm" and os.path.exists(ALERTS_LOG_PATH):
        try:
            with open(ALERTS_LOG_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_alerts = list(reader)
                matching_alert_rows = all_alerts[alerts_before_count:]
        except Exception as exc:
            log.warning("Could not read alerts_log.csv: %s", exc)

        burst_start_dt = None
        if burst_start_info["utc"]:
            burst_start_dt = burst_start_info["utc"]

        for alert in matching_alert_rows:
            # Parse alert timestamp
            alert_ts_str = alert.get("timestamp", "")
            try:
                alert_dt = datetime.fromisoformat(alert_ts_str.replace("Z", "+00:00"))
                rel_sec = (alert_dt.timestamp() - burst_start_dt.timestamp()) if burst_start_dt else 0.0
            except Exception:
                rel_sec = 0.0

            cur_v = float(alert.get("current_value", 0.0))
            fc_v = float(alert.get("forecast_value_30s", 0.0))
            slo_th = float(alert.get("slo_threshold", 0.0))

            if fc_v >= slo_th and cur_v < slo_th:
                provenance = "FORECAST"
            else:
                provenance = "CURRENT"

            perm_scale_events.append({
                "alert_id": alert.get("alert_id"),
                "provenance": provenance,
                "metric": alert.get("metric"),
                "current_value": cur_v,
                "forecast_value_30s": fc_v,
                "slo_threshold": slo_th,
                "relative_to_burst_start_sec": round(rel_sec, 2),
                "is_before_burst": rel_sec < 0,
                "alert_raw": alert,
            })

    # -----------------------------------------------------------------------
    # Run Summary Calculations
    # -----------------------------------------------------------------------
    total_arrivals = sum(r["arrivals_count"] for r in collected_rows)
    total_ok = sum(r["ok_count"] for r in collected_rows)
    total_rejected = sum(r["rejected_count"] for r in collected_rows)
    total_timeout = sum(r["timeout_count"] for r in collected_rows)
    total_sent = total_ok + total_rejected + total_timeout

    pct_ok = (total_ok / total_sent * 100.0) if total_sent > 0 else 0.0
    run_duration = collected_rows[-1]["elapsed_sec"] if collected_rows else 1.0
    goodput = total_ok / run_duration if run_duration > 0 else 0.0

    p95_values = [r["p95_latency_ms"] for r in collected_rows if r["p95_latency_ms"] > 0]
    peak_p95 = max(p95_values) if p95_values else 0.0
    sec_p95_ok = sum(1 for r in collected_rows if r["p95_latency_ms"] <= 300.0)
    worker_seconds = sum(r["active_workers"] for r in collected_rows)

    # Timing relative to burst
    # 1. seconds from burst start to first scale-up
    sec_to_first_scale_up = None
    burst_mono = burst_start_info["mono"]

    if mode == "fixed_2":
        sec_to_first_scale_up = None
    elif mode == "fixed_8":
        sec_to_first_scale_up = 0.0
    elif mode == "reactive":
        if reactive_scaler and reactive_scaler.scale_events:
            first_ev = reactive_scaler.scale_events[0]
            sec_to_first_scale_up = round(first_ev["time_mono"] - burst_mono, 2)
    elif mode == "perm":
        # From scale_events on collector
        scale_ups = [ev for ev in scale_events if ev.get("type") == "scale_up"]
        if scale_ups and burst_start_info["utc"]:
            try:
                first_ev = scale_ups[0]
                ev_dt = datetime.fromisoformat(first_ev["timestamp"].replace("Z", "+00:00"))
                sec_to_first_scale_up = round(ev_dt.timestamp() - burst_start_info["utc"].timestamp(), 2)
            except Exception:
                pass
        if sec_to_first_scale_up is None and perm_scale_events:
            sec_to_first_scale_up = perm_scale_events[0]["relative_to_burst_start_sec"]

    # 2. seconds from burst start until p95 <= 300ms again
    # We examine rows during and after burst
    post_burst_rows = [r for r in collected_rows if r["burst_start_rel_sec"] >= 0]
    sec_to_recovery = None
    breached = False

    for r in post_burst_rows:
        if r["p95_latency_ms"] > 300.0:
            breached = True
        elif breached and r["p95_latency_ms"] <= 300.0 and r["ok_count"] > 0:
            sec_to_recovery = r["burst_start_rel_sec"]
            break

    if not breached:
        sec_to_recovery = 0.0  # never breached SLO
    elif sec_to_recovery is None:
        sec_to_recovery = post_burst_rows[-1]["burst_start_rel_sec"] if post_burst_rows else None

    run_summary = {
        "run_id": run_id,
        "mode": mode,
        "repeat": repeat_idx,
        "seed": seed,
        "duration_sec": run_duration,
        "total_arrivals": total_arrivals,
        "total_ok": total_ok,
        "total_rejected": total_rejected,
        "total_timeout": total_timeout,
        "pct_ok": round(pct_ok, 2),
        "goodput_req_s": round(goodput, 2),
        "sec_p95_ok": sec_p95_ok,
        "peak_p95_ms": round(peak_p95, 2),
        "sec_to_first_scale_up": sec_to_first_scale_up,
        "sec_to_p95_recovery": sec_to_recovery,
        "worker_seconds_cost": round(worker_seconds, 2),
        "perm_scale_events": perm_scale_events,
        "matching_alert_rows": matching_alert_rows,
    }

    return {
        "summary": run_summary,
        "rows": collected_rows,
    }


def aggregate_mode_results(run_summaries: List[Dict]) -> Dict:
    """Calculate mean, min, and max across repeats for all summary metrics."""
    metrics = [
        "pct_ok",
        "goodput_req_s",
        "sec_p95_ok",
        "peak_p95_ms",
        "sec_to_first_scale_up",
        "sec_to_p95_recovery",
        "worker_seconds_cost",
    ]

    agg = {}
    for m in metrics:
        vals = [r[m] for r in run_summaries if r.get(m) is not None]
        if vals:
            agg[m] = {
                "mean": round(sum(vals) / len(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }
        else:
            agg[m] = {"mean": None, "min": None, "max": None}

    return agg


def main():
    parser = argparse.ArgumentParser(description="PERM Step 1b scaling comparison runner")
    parser.add_argument("--modes", type=str, default="fixed_2,fixed_8,reactive,perm",
                        help="Comma-separated modes: fixed_2,fixed_8,reactive,perm")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per mode (default 3)")
    parser.add_argument("--profile", choices=["step", "ramp"], default="step")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--load", type=int, default=7)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < args.repeats:
        seeds = [42 + i for i in range(args.repeats)]

    log.info("Starting comparison: modes=%s repeats=%d profile=%s seeds=%s", modes, args.repeats, args.profile, seeds[:args.repeats])

    all_rows: List[Dict] = []
    mode_summaries: Dict[str, List[Dict]] = {m: [] for m in modes}
    perm_all_alerts: List[Dict] = []

    for mode in modes:
        for rep in range(1, args.repeats + 1):
            seed = seeds[rep - 1]
            res = run_single_experiment(
                mode=mode,
                repeat_idx=rep,
                seed=seed,
                profile=args.profile,
                burst_load=args.load,
            )
            mode_summaries[mode].append(res["summary"])
            all_rows.extend(res["rows"])
            if res["summary"].get("perm_scale_events"):
                perm_all_alerts.extend(res["summary"]["perm_scale_events"])

    # -----------------------------------------------------------------------
    # Save step1b_comparison.csv
    # -----------------------------------------------------------------------
    with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_ROW_FIELDS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    log.info("Saved per-second rows to %s (%d rows)", COMPARISON_CSV, len(all_rows))

    # -----------------------------------------------------------------------
    # Aggregate and Save step1b_summary.json
    # -----------------------------------------------------------------------
    mode_aggregates = {}
    for mode in modes:
        mode_aggregates[mode] = aggregate_mode_results(mode_summaries[mode])

    final_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "load": args.load,
        "repeats": args.repeats,
        "modes": modes,
        "mode_aggregates": mode_aggregates,
        "individual_runs": mode_summaries,
        "perm_scale_up_alerts": perm_all_alerts,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    log.info("Saved summary to %s", SUMMARY_JSON)

    # -----------------------------------------------------------------------
    # Print Markdown Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1b COMPARISON RESULTS SUMMARY")
    print("=" * 80)
    headers = ["Mode", "% OK", "Goodput (r/s)", "p95 <= 300ms (s)", "Peak p95 (ms)", "1st Scale (s)", "Recovery (s)", "Cost (w-s)"]
    print(f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<15} {headers[3]:<18} {headers[4]:<14} {headers[5]:<14} {headers[6]:<14} {headers[7]:<12}")
    print("-" * 115)

    for m in modes:
        agg = mode_aggregates[m]
        def _fmt(key):
            item = agg.get(key, {})
            mean = item.get("mean")
            mn = item.get("min")
            mx = item.get("max")
            if mean is None:
                return "N/A"
            return f"{mean} [{mn}-{mx}]"

        print(
            f"{m:<12} "
            f"{_fmt('pct_ok'):<12} "
            f"{_fmt('goodput_req_s'):<15} "
            f"{_fmt('sec_p95_ok'):<18} "
            f"{_fmt('peak_p95_ms'):<14} "
            f"{_fmt('sec_to_first_scale_up'):<14} "
            f"{_fmt('sec_to_p95_recovery'):<14} "
            f"{_fmt('worker_seconds_cost'):<12}"
        )
    print("=" * 115)


if __name__ == "__main__":
    main()
