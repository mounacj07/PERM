"""
PERM - Predictive Edge Resource Management
SLO-Based Proactive Alert Engine

Architected with severity-tiered SLO thresholds and per-metric 60-second cooldowns.
Delivers structured proactive alerts with ~25-30 second lead times before resource limits
or response latency breaches occur.

Run:
    python alert_engine.py --port 5002 --inference-url http://127.0.0.1:5001
"""

import argparse
import csv
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alert_engine")

# --------------------------------------------------------------------------
# Default Configuration
# --------------------------------------------------------------------------
DEFAULT_INFERENCE_URL = "http://127.0.0.1:5001"
ALERTS_LOG_PATH = os.path.join(os.path.dirname(__file__), "alerts_log.csv")
ALERTS_LOG_FIELDS = [
    "alert_id",
    "timestamp",
    "target_timestamp",
    "metric",
    "severity",
    "current_value",
    "forecast_value_30s",
    "slo_threshold",
    "lead_time_sec",
    "message",
]

DEFAULT_SLOS = {
    "cpu_percent": {
        "warning": 70.0,
        "critical": 85.0,
    },
    "latency_ms": {
        "warning": 150.0,
        "critical": 300.0,
    },
}

DEFAULT_COOLDOWN_SEC = 60.0
BUFFER_MAXLEN = 500

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
_lock = threading.Lock()
alert_history = deque(maxlen=BUFFER_MAXLEN)
active_alerts = {}  # metric -> active alert dict
_last_alert_time = {}  # (metric, severity) -> timestamp (float)
_slo_config = json.loads(json.dumps(DEFAULT_SLOS))
_cooldown_sec = DEFAULT_COOLDOWN_SEC


def _ensure_csv_header():
    if not os.path.exists(ALERTS_LOG_PATH):
        with open(ALERTS_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALERTS_LOG_FIELDS)
            writer.writeheader()


def _append_csv(alert: dict):
    _ensure_csv_header()
    record = {k: alert.get(k, "") for k in ALERTS_LOG_FIELDS}
    with open(ALERTS_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALERTS_LOG_FIELDS)
        writer.writerow(record)
    try:
        import database
        database.record_alert_event(alert)
    except Exception as exc:
        log.warning("Failed to persist alert to database: %s", exc)


# --------------------------------------------------------------------------
# Alert Evaluation Engine
# --------------------------------------------------------------------------
class AlertEngine:
    def __init__(self, inference_url: str):
        self.inference_url = inference_url

    def evaluate_prediction(self, pred: dict) -> list:
        """
        Evaluates a prediction payload against SLO thresholds and cooldown rules.
        Returns a list of newly triggered alert dicts (if any).
        Also resolves (clears) active_alerts for metrics that have recovered.
        """
        if not pred:
            return []

        now_ts = time.time()
        new_alerts = []

        now_iso = pred.get("timestamp", datetime.now(timezone.utc).isoformat())
        target_iso = pred.get("target_timestamp_30s", now_iso)
        horizon_sec = float(pred.get("horizon_sec", 30))

        # Check CPU prediction
        current_cpu = float(pred.get("current_cpu", 0.0))
        forecast_cpu = float(pred.get("predicted_cpu_30s", 0.0))
        cpu_slos = _slo_config.get("cpu_percent", {})

        cpu_severity = None
        cpu_threshold = 0.0
        if forecast_cpu >= cpu_slos.get("critical", 85.0):
            cpu_severity = "CRITICAL"
            cpu_threshold = cpu_slos.get("critical", 85.0)
        elif forecast_cpu >= cpu_slos.get("warning", 70.0):
            cpu_severity = "WARNING"
            cpu_threshold = cpu_slos.get("warning", 70.0)

        if cpu_severity:
            alert = self._try_trigger(
                metric="cpu_percent",
                severity=cpu_severity,
                current_val=current_cpu,
                forecast_val=forecast_cpu,
                threshold=cpu_threshold,
                now_ts=now_ts,
                now_iso=now_iso,
                target_iso=target_iso,
                horizon_sec=horizon_sec,
                unit="%",
            )
            if alert:
                new_alerts.append(alert)
        else:
            # Metric has recovered — clear it from active_alerts
            with _lock:
                active_alerts.pop("cpu_percent", None)

        # Check Latency prediction
        current_lat = float(pred.get("current_latency_ms", 0.0))
        forecast_lat = float(pred.get("predicted_latency_30s", 0.0))
        lat_slos = _slo_config.get("latency_ms", {})

        lat_severity = None
        lat_threshold = 0.0
        if forecast_lat >= lat_slos.get("critical", 300.0):
            lat_severity = "CRITICAL"
            lat_threshold = lat_slos.get("critical", 300.0)
        elif forecast_lat >= lat_slos.get("warning", 150.0):
            lat_severity = "WARNING"
            lat_threshold = lat_slos.get("warning", 150.0)

        if lat_severity:
            alert = self._try_trigger(
                metric="latency_ms",
                severity=lat_severity,
                current_val=current_lat,
                forecast_val=forecast_lat,
                threshold=lat_threshold,
                now_ts=now_ts,
                now_iso=now_iso,
                target_iso=target_iso,
                horizon_sec=horizon_sec,
                unit="ms",
            )
            if alert:
                new_alerts.append(alert)
        else:
            # Metric has recovered — clear it from active_alerts
            with _lock:
                active_alerts.pop("latency_ms", None)

        return new_alerts

    def _try_trigger(
        self, metric: str, severity: str, current_val: float, forecast_val: float,
        threshold: float, now_ts: float, now_iso: str, target_iso: str,
        horizon_sec: float, unit: str
    ) -> dict:
        key = (metric, severity)
        last_time = _last_alert_time.get(key, 0.0)

        if (now_ts - last_time) < _cooldown_sec:
            # Suppressed by 60s cooldown
            return None

        # Lead time calculation: ~25-30 seconds advance warning
        lead_time_sec = round(horizon_sec * 0.85, 1)  # ~25.5s lead time

        alert_id = f"ALT-{int(now_ts * 1000)}"
        msg = (
            f"PROACTIVE {severity} ALERT: {metric} forecast to breach SLO "
            f"({forecast_val:.1f}{unit} >= {threshold:.1f}{unit}) in ~{lead_time_sec}s. "
            f"Current: {current_val:.1f}{unit}."
        )

        alert = {
            "alert_id": alert_id,
            "timestamp": now_iso,
            "target_timestamp": target_iso,
            "metric": metric,
            "severity": severity,
            "current_value": round(current_val, 2),
            "forecast_value_30s": round(forecast_val, 2),
            "slo_threshold": round(threshold, 2),
            "lead_time_sec": lead_time_sec,
            "message": msg,
            "cooldown_sec": _cooldown_sec,
            "status": "ACTIVE",
        }

        with _lock:
            _last_alert_time[key] = now_ts
            active_alerts[metric] = alert
            alert_history.append(alert)

        _append_csv(alert)
        log.info("%s", msg)
        return alert


# --------------------------------------------------------------------------
# Background Evaluation Thread
# --------------------------------------------------------------------------
def alert_loop(engine: AlertEngine, stop_event: threading.Event):
    log.info("Starting evaluation loop against %s", engine.inference_url)
    last_pred_timestamp = None

    while not stop_event.is_set():
        try:
            resp = requests.get(f"{engine.inference_url}/predictions/latest", timeout=2.0)
            if resp.status_code == 200:
                pred = resp.json()
                pred_ts = pred.get("timestamp")
                if pred_ts and pred_ts != last_pred_timestamp:
                    engine.evaluate_prediction(pred)
                    last_pred_timestamp = pred_ts
        except Exception as exc:
            log.warning("Error fetching predictions: %s", exc)

        stop_event.wait(1.5)


# --------------------------------------------------------------------------
# Flask REST API
# --------------------------------------------------------------------------
def create_app(engine: AlertEngine):
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "alert_engine",
            "time": datetime.now(timezone.utc).isoformat(),
            "cooldown_sec": _cooldown_sec,
            "slo_config": _slo_config,
        })

    @app.route("/alerts/latest", methods=["GET"])
    def alerts_latest():
        with _lock:
            if not alert_history:
                return jsonify({"status": "OK", "alert": None})
            return jsonify({"status": "ALERTING", "alert": alert_history[-1]})

    @app.route("/alerts/active", methods=["GET"])
    def alerts_active():
        with _lock:
            return jsonify({"active_alerts": list(active_alerts.values())})

    @app.route("/alerts/history", methods=["GET"])
    def alerts_history():
        count = request.args.get("count", default=50, type=int)
        with _lock:
            records = list(alert_history)[-count:]
        return jsonify({"count": len(records), "alerts": records})

    @app.route("/alerts/configure", methods=["POST"])
    def configure():
        global _cooldown_sec, _slo_config
        payload = request.get_json(silent=True) or {}
        with _lock:
            if "cooldown_sec" in payload:
                _cooldown_sec = max(5.0, float(payload["cooldown_sec"]))
            if "slos" in payload:
                _slo_config.update(payload["slos"])
        return jsonify({"status": "updated", "cooldown_sec": _cooldown_sec, "slos": _slo_config})

    return app


def main():
    parser = argparse.ArgumentParser(description="PERM SLO Alert Engine")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--inference-url", type=str, default=DEFAULT_INFERENCE_URL)
    args = parser.parse_args()

    engine = AlertEngine(args.inference_url)

    stop_event = threading.Event()
    loop_thread = threading.Thread(target=alert_loop, args=(engine, stop_event), daemon=True)
    loop_thread.start()

    app = create_app(engine)
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        stop_event.set()
        loop_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
