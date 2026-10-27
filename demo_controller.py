#!/usr/bin/env python3
"""
PERM - demo_controller.py
=========================
Standalone opt-in demo controller for live evaluation and demo runs.

Behavior:
  - 1 s loop querying /metrics/latest.
  - Pool utilization = clip(worker_cpu_percent * logical_cpus / active_workers, 0, 100),
    smoothed over 3 s rolling window.
  - Displays queue depth, arrivals, failures, and pool utilization.
  - Alerts written to controller_events.csv:
      WARN when pool utilization crosses >= 70%
      CRITICAL when pool utilization crosses >= 85%
      RECOVERY when pool utilization falls back < 70%
  - Scaling decision each second = max(reactive_target, predictive_target):
      reactive_target = ceil(smoothed_arrivals * service_time / 0.7),
      applied when utilization > 70% or queue_depth >= 5;
      if utilization > 85% for 2 s, add 2 workers more (safety boost).
      predictive_target = ceil(forecast_upper_bound / (capacity_per_worker * 0.7))
      from /forecast/demand (PERM_FORECASTER=stagbdt) with last 60 s of arrivals.
      If forecast service is offline, falls back to reactive only and logs "forecast offline".
  - Target clipped to [2, 8].
  - Scale-up executed immediately via POST /api/capacity {"active_workers": n}.
  - Scale-down executed only after target stays below current worker count for 15 s.
  - Logs every scale event with triggering rule (forecast / reactive / safety) and
    pool utilization before and after.

Usage:
  python demo_controller.py --mode predictive --initial-workers 2
  python demo_controller.py --mode reactive --initial-workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo_controller")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "testbed_calibration.json")
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "controller_events.csv")
DEFAULT_STATE_PATH = os.path.join(PROJECT_ROOT, "controller_state.json")
CSV_HEADER = ["time", "type", "signal", "value", "reason"]


class DemoController:
    """Standalone edge capacity controller with reactive and predictive scaling logic."""

    def __init__(
        self,
        mode: str = "predictive",
        initial_workers: int = 2,
        collector_url: str = "http://127.0.0.1:5000",
        inference_url: str = "http://127.0.0.1:5001",
        calibration_path: str = DEFAULT_CALIBRATION_PATH,
        csv_path: str = DEFAULT_CSV_PATH,
        state_path: str = DEFAULT_STATE_PATH,
    ):
        self.mode = mode.lower()
        if self.mode not in ("predictive", "reactive", "capacity_aware"):
            raise ValueError(f"Invalid mode '{mode}'; must be 'predictive', 'reactive', or 'capacity_aware'")

        self.collector_url = collector_url.rstrip("/")
        self.inference_url = inference_url.rstrip("/")
        self.csv_path = csv_path
        self.state_path = state_path
        self.min_workers = 2
        self.max_workers = 8

        # Load constants from calibration file
        self.load_calibration(calibration_path)

        # Runtime state
        self.current_workers = max(self.min_workers, min(self.max_workers, int(initial_workers)))
        self.raw_util_history: Deque[float] = deque(maxlen=3)
        self.arrivals_history: Deque[float] = deque(maxlen=60)
        self.alert_state: str = "NORMAL"  # NORMAL, WARN, CRITICAL
        self.consecutive_above_85: int = 0
        self.scale_down_timer_start: Optional[float] = None
        self.pending_scale_down_target: Optional[int] = None
        self.events_log: List[Dict[str, Any]] = []

        self._ensure_csv_header()

    def load_calibration(self, path: str) -> None:
        """Load service_time, capacity_per_worker, and logical_cpus from calibration file."""
        if not os.path.exists(path):
            log.warning("Calibration file %s not found. Using defaults.", path)
            self.service_time = 0.126
            self.capacity_per_worker = 7.89
            self.logical_cpus = 16
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        load = float(data.get("load", 7))
        ms_per_load = float(data.get("ms_per_load", 18.0))
        self.service_time = (load * ms_per_load) / 1000.0  # 0.126 s

        self.logical_cpus = int(data.get("cpu_count", 16))

        results = data.get("results", [])
        if results and "saturation_rps" in results[0]:
            self.capacity_per_worker = float(results[0]["saturation_rps"])  # 7.89
        else:
            self.capacity_per_worker = 7.89

        log.info(
            "Loaded calibration: service_time=%.3fs, capacity_per_worker=%.2f, logical_cpus=%d",
            self.service_time,
            self.capacity_per_worker,
            self.logical_cpus,
        )

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)
            except Exception as exc:
                log.warning("Could not initialize %s: %s", self.csv_path, exc)

    def write_event(
        self,
        event_type: str,
        signal: str,
        value: Any,
        reason: str,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append an event to controller_events.csv."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        row = [ts, event_type, signal, str(value), reason]
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as exc:
            log.warning("Failed writing to %s: %s", self.csv_path, exc)

        self.events_log.append({
            "time": ts,
            "type": event_type,
            "signal": signal,
            "value": value,
            "reason": reason,
        })

    def compute_pool_utilization(
        self, worker_cpu_percent: float, active_workers: int
    ) -> float:
        """
        Pool utilization = clip(worker_cpu_percent * logical_cpus / active_workers, 0, 100).
        """
        denom = max(1, active_workers)
        raw = (worker_cpu_percent * self.logical_cpus) / denom
        return float(max(0.0, min(100.0, raw)))

    def update_alerts(self, smoothed_util: float, timestamp: Optional[str] = None) -> Optional[str]:
        """
        Alerts:
          WARN when pool utilization crosses >= 70%
          CRITICAL at >= 85%
          RECOVERY when it falls back below 70%
        """
        event_written = None
        if smoothed_util >= 85.0:
            if self.alert_state != "CRITICAL":
                self.alert_state = "CRITICAL"
                self.write_event(
                    "CRITICAL",
                    "pool_utilization",
                    round(smoothed_util, 2),
                    "pool utilization crossed 85% critical threshold",
                    timestamp=timestamp,
                )
                log.warning("CRITICAL ALERT: Pool utilization reached %.1f%%", smoothed_util)
                event_written = "CRITICAL"
        elif smoothed_util >= 70.0:
            if self.alert_state == "NORMAL":
                self.alert_state = "WARN"
                self.write_event(
                    "WARN",
                    "pool_utilization",
                    round(smoothed_util, 2),
                    "pool utilization crossed 70% warning threshold",
                    timestamp=timestamp,
                )
                log.warning("WARN ALERT: Pool utilization crossed %.1f%%", smoothed_util)
                event_written = "WARN"
            elif self.alert_state == "CRITICAL":
                # Stepped down from CRITICAL to WARN
                self.alert_state = "WARN"
        else:
            # smoothed_util < 70.0
            if self.alert_state in ("WARN", "CRITICAL"):
                old_state = self.alert_state
                self.alert_state = "NORMAL"
                self.write_event(
                    "RECOVERY",
                    "pool_utilization",
                    round(smoothed_util, 2),
                    f"pool utilization recovered below 70% from {old_state}",
                    timestamp=timestamp,
                )
                log.info("RECOVERY: Pool utilization recovered to %.1f%%", smoothed_util)
                event_written = "RECOVERY"

        return event_written

    def compute_reactive_target(
        self,
        smoothed_arrivals: float,
        smoothed_util: float,
        queue_depth: int,
        current_workers: int,
    ) -> Tuple[int, str]:
        """
        reactive_target = ceil(smoothed_arrivals * service_time / 0.7),
        applied when utilization > 70% or queue_depth >= 5;
        if utilization > 85% for 2 s, add 2 workers more.
        Returns: (target, reason_rule)
        """
        base_target = int(math.ceil(smoothed_arrivals * self.service_time / 0.7))
        base_target = max(self.min_workers, min(self.max_workers, base_target))

        is_stressed = (smoothed_util > 70.0) or (queue_depth >= 5)

        if is_stressed:
            rule = "reactive"
            target = max(current_workers, base_target)
            if self.consecutive_above_85 >= 2:
                target = min(self.max_workers, target + 2)
                rule = "safety"
            return target, rule
        else:
            # Not stressed: demand may warrant scale-down
            target = min(current_workers, base_target)
            return target, "reactive"

    def query_forecast(self, arrivals_list: List[float]) -> Optional[Dict[str, Any]]:
        """Query /forecast/demand endpoint from inference service."""
        if not arrivals_list:
            return None
        url = f"{self.inference_url}/forecast/demand"
        try:
            resp = requests.post(
                url,
                json={"arrivals": arrivals_list, "horizon": 20},
                timeout=1.5,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                log.warning("Forecast endpoint returned HTTP %d", resp.status_code)
                return None
        except Exception:
            return None

    def compute_predictive_target(
        self, arrivals_list: List[float]
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        predictive_target = ceil(forecast_upper_bound / (capacity_per_worker * 0.7)).
        If offline, falls back to reactive and logs 'forecast offline'.
        """
        if self.mode != "predictive":
            return 0, None

        if len(arrivals_list) < 10:
            log.info("predictive warming up")
            return 0, None

        forecast_data = self.query_forecast(arrivals_list)
        if forecast_data is None:
            log.warning("forecast offline")
            return 0, None

        upper_bound = float(forecast_data.get("upper_bound", 0.0))
        eff_capacity = self.capacity_per_worker * 0.7
        target = int(math.ceil(upper_bound / eff_capacity))
        target = max(self.min_workers, min(self.max_workers, target))
        return target, forecast_data

    def compute_capacity_aware_target(self, smoothed_arrivals: float) -> int:
        """
        capacity_aware: ceil(smoothed_arrivals * service_time / 0.7)
        from arrivals only, no utilization threshold.
        """
        target = int(math.ceil(smoothed_arrivals * self.service_time / 0.7))
        return max(self.min_workers, min(self.max_workers, target))

    def evaluate_step(
        self,
        record: Dict[str, Any],
        now_monotonic: Optional[float] = None,
        now_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single 1s metrics sample and make scaling & alert decisions.
        """
        t_mono = time.monotonic() if now_monotonic is None else now_monotonic
        ts_iso = record.get("timestamp") or (
            datetime.now(timezone.utc).isoformat() if now_iso is None else now_iso
        )

        active_workers = int(record.get("active_workers", self.current_workers))
        self.current_workers = active_workers

        worker_cpu_pct = float(record.get("worker_cpu_percent", 0.0))
        queue_depth = int(record.get("queue_depth", 0))
        arrivals = float(record.get("arrivals_count", 0.0))
        error_count = int(record.get("error_count", 0))

        # 1. Pool utilization computation & 3s smoothing
        raw_util = self.compute_pool_utilization(worker_cpu_pct, active_workers)
        self.raw_util_history.append(raw_util)
        smoothed_util = sum(self.raw_util_history) / len(self.raw_util_history)

        # 2. Arrivals tracking
        self.arrivals_history.append(arrivals)
        recent_arrivals = list(self.arrivals_history)[-3:]
        smoothed_arrivals = sum(recent_arrivals) / len(recent_arrivals) if recent_arrivals else arrivals

        # 3. Alert engine updates
        alert_event = self.update_alerts(smoothed_util, timestamp=ts_iso)

        # 4. Track consecutive seconds above 85%
        if smoothed_util > 85.0:
            self.consecutive_above_85 += 1
        else:
            self.consecutive_above_85 = 0

        # 5. Compute scaling targets
        if self.mode == "capacity_aware":
            cap_target = self.compute_capacity_aware_target(smoothed_arrivals)
            reactive_target = cap_target
            predictive_target = 0
            forecast_info = None
            scaling_target = cap_target
            if scaling_target > self.current_workers:
                trigger_rule = "capacity_aware"
            elif scaling_target < self.current_workers:
                trigger_rule = "scale_down"
            else:
                trigger_rule = "capacity_aware"
        else:
            reactive_target, reactive_rule = self.compute_reactive_target(
                smoothed_arrivals, smoothed_util, queue_depth, self.current_workers
            )

            predictive_target, forecast_info = self.compute_predictive_target(list(self.arrivals_history))

            # Scaling decision each second = max(reactive_target, predictive_target)
            if predictive_target > reactive_target:
                scaling_target = predictive_target
                trigger_rule = "forecast"
            elif reactive_target > self.current_workers:
                scaling_target = reactive_target
                trigger_rule = reactive_rule
            else:
                scaling_target = reactive_target
                trigger_rule = "reactive"

        # Clip target to [2, 8]
        scaling_target = max(self.min_workers, min(self.max_workers, scaling_target))

        # 6. Scaling action decision (Scale-Up vs Scale-Down Hysteresis)
        scale_action: Optional[str] = None
        target_to_apply: Optional[int] = None

        if scaling_target > self.current_workers:
            # Immediate scale up
            scale_action = "scale_up"
            target_to_apply = scaling_target
            self.scale_down_timer_start = None
            self.pending_scale_down_target = None
        elif scaling_target < self.current_workers:
            # Scale-down candidate: require staying below current count for 15 s
            if self.scale_down_timer_start is None:
                self.scale_down_timer_start = t_mono
                self.pending_scale_down_target = scaling_target
            else:
                self.pending_scale_down_target = max(self.pending_scale_down_target, scaling_target)
                elapsed = t_mono - self.scale_down_timer_start
                if elapsed >= 15.0:
                    scale_action = "scale_down"
                    target_to_apply = self.pending_scale_down_target
                    self.scale_down_timer_start = None
                    self.pending_scale_down_target = None
        else:
            # target == current_workers
            self.scale_down_timer_start = None
            self.pending_scale_down_target = None

        # 7. Apply scaling if triggered
        scale_event_info = None
        if scale_action and target_to_apply and target_to_apply != self.current_workers:
            util_before = smoothed_util
            # Request resize from collector
            applied_workers = self.apply_capacity(target_to_apply)
            util_after = self.compute_pool_utilization(worker_cpu_pct, applied_workers)

            scale_rule = trigger_rule if scale_action == "scale_up" else "scale_down"
            reason_str = (
                f"rule={scale_rule};from={self.current_workers};to={applied_workers};"
                f"util_before={util_before:.1f}%;util_after={util_after:.1f}%"
            )
            self.write_event(
                scale_action.upper(),
                "active_workers",
                applied_workers,
                reason_str,
                timestamp=ts_iso,
            )
            log.info(
                "SCALE EVENT [%s]: %d -> %d workers (trigger=%s, util_before=%.1f%%, util_after=%.1f%%)",
                scale_action.upper(),
                self.current_workers,
                applied_workers,
                scale_rule,
                util_before,
                util_after,
            )
            scale_event_info = {
                "action": scale_action,
                "rule": scale_rule,
                "from_workers": self.current_workers,
                "to_workers": applied_workers,
                "util_before": util_before,
                "util_after": util_after,
            }
            self.current_workers = applied_workers

        res = {
            "timestamp": ts_iso,
            "mode": self.mode,
            "current_workers": self.current_workers,
            "raw_util": round(raw_util, 2),
            "smoothed_util": round(smoothed_util, 2),
            "smoothed_arrivals": round(smoothed_arrivals, 2),
            "queue_depth": queue_depth,
            "error_count": error_count,
            "alert_state": self.alert_state,
            "alert_event": alert_event,
            "reactive_target": reactive_target,
            "predictive_target": predictive_target,
            "chosen_target": scaling_target,
            "trigger_rule": trigger_rule,
            "scale_event": scale_event_info,
        }
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(res, f)
        except Exception:
            pass
        return res

    def apply_capacity(self, target_workers: int) -> int:
        """POST /api/capacity to update active workers."""
        try:
            resp = requests.post(
                f"{self.collector_url}/api/capacity",
                json={"active_workers": target_workers},
                timeout=2.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return int(data.get("active_workers", target_workers))
        except Exception as exc:
            log.warning("Failed to POST /api/capacity: %s", exc)
        return target_workers

    def fetch_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """GET /metrics/latest from collector."""
        try:
            resp = requests.get(f"{self.collector_url}/metrics/latest", timeout=1.5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            log.debug("GET /metrics/latest failed: %s", exc)
        return None

    def run_loop(self, duration_sec: float = 0.0) -> None:
        """Run the 1 s control loop indefinitely or for duration_sec."""
        log.info(
            "Starting demo_controller (mode=%s, workers=%d, collector=%s, inference=%s)",
            self.mode,
            self.current_workers,
            self.collector_url,
            self.inference_url,
        )

        # Set initial capacity
        self.apply_capacity(self.current_workers)

        t_start = time.monotonic()
        next_tick = t_start + 1.0

        try:
            while True:
                now = time.monotonic()
                if duration_sec > 0.0 and (now - t_start) >= duration_sec:
                    log.info("Reached target duration of %.1f s. Stopping controller.", duration_sec)
                    break

                record = self.fetch_latest_metrics()
                if record:
                    step_res = self.evaluate_step(record, now_monotonic=now)
                    # Formatted status output
                    print(
                        f"[{step_res['timestamp'][11:19]}] Mode: {self.mode.upper():10s} | "
                        f"Workers: {step_res['current_workers']} | "
                        f"Pool Util: {step_res['smoothed_util']:5.1f}% | "
                        f"Arrivals: {step_res['smoothed_arrivals']:4.1f}/s | "
                        f"QDepth: {step_res['queue_depth']:2d} | "
                        f"Errors: {step_res['error_count']:2d} | "
                        f"Target: {step_res['chosen_target']} ({step_res['trigger_rule']}) | "
                        f"Alert: {step_res['alert_state']}",
                        flush=True,
                    )
                else:
                    log.debug("No telemetry record available from %s", self.collector_url)

                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_tick = time.monotonic() + 1.0

        except KeyboardInterrupt:
            log.info("Controller interrupted by user. Exiting cleanly.")


def main():
    parser = argparse.ArgumentParser(description="PERM Standalone Demo Controller")
    parser.add_argument(
        "--mode",
        choices=["predictive", "reactive", "capacity_aware"],
        default="predictive",
        help="Scaling mode: 'predictive', 'reactive', or 'capacity_aware'",
    )
    parser.add_argument(
        "--initial-workers",
        type=int,
        default=2,
        help="Initial worker count (default: 2)",
    )
    parser.add_argument(
        "--collector-url",
        type=str,
        default="http://127.0.0.1:5000",
        help="Telemetry collector base URL",
    )
    parser.add_argument(
        "--inference-url",
        type=str,
        default="http://127.0.0.1:5001",
        help="Inference service base URL",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=DEFAULT_CALIBRATION_PATH,
        help="Path to testbed_calibration.json",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to controller_events.csv",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds to run (0 = run indefinitely)",
    )
    args = parser.parse_args()

    controller = DemoController(
        mode=args.mode,
        initial_workers=args.initial_workers,
        collector_url=args.collector_url,
        inference_url=args.inference_url,
        calibration_path=args.calibration,
        csv_path=args.csv,
    )
    controller.run_loop(duration_sec=args.duration)


if __name__ == "__main__":
    main()
