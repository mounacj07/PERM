"""
PERM - Predictive Edge Resource Management
Remediation Service & Edge Resource Capacity Manager

Manages edge resource allocation (Worker Cores / Compute Quota) in response to
proactive SLO alerts from the Alert Engine.

Supports two operational modes:
  1. AUTO MODE: Automatically provisions +2 worker cores when proactive alerts fire
     before SLO breach occurs.
  2. MANUAL MODE: Queues pending scaling requests for human approval on the Web Dashboard,
     displaying ~25-30s lead time countdowns.

Run:
    python remediation_service.py --port 5003 --alert-url http://127.0.0.1:5002
"""

import argparse
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
log = logging.getLogger("remediation_service")

DEFAULT_ALERT_URL = "http://127.0.0.1:5002"
DEFAULT_COLLECTOR_URL = "http://127.0.0.1:5000"
BASE_WORKERS = 2
MAX_WORKERS = 10
SCALE_DOWN_CPU_THRESHOLD = 50.0   # scale down when CPU stays below this
SCALE_DOWN_STREAK_SEC = 20        # for this many consecutive seconds

# State
_lock = threading.Lock()
_mode = "AUTO"  # "AUTO" or "MANUAL"
_active_workers = BASE_WORKERS
_pending_approvals = deque(maxlen=5)  # thread-safe, automatically caps at 5
_scaling_history = []
_handled_alerts = set()
_collector_url = DEFAULT_COLLECTOR_URL
_low_cpu_streak = 0   # consecutive seconds CPU has been below threshold


def _sync_capacity_to_collector(workers: int):
    try:
        requests.post(f"{_collector_url}/api/capacity", json={"active_workers": workers}, timeout=0.8)
    except Exception:
        pass


class RemediationEngine:
    def __init__(self, alert_url: str):
        self.alert_url = alert_url

    def check_alerts(self):
        try:
            resp = requests.get(f"{self.alert_url}/alerts/active", timeout=2.0)
            if resp.status_code != 200:
                return
            active_dict = resp.json().get("active_alerts", [])
            for alert in active_dict:
                alert_id = alert.get("alert_id")
                if not alert_id or alert_id in _handled_alerts:
                    continue
                self.handle_alert(alert)
        except Exception:
            pass

    def check_cooldown(self):
        """Auto scale-down: if workers > base and CPU stays below threshold for N seconds."""
        global _active_workers, _low_cpu_streak
        with _lock:
            if _active_workers <= BASE_WORKERS:
                _low_cpu_streak = 0
                return
        try:
            resp = requests.get(f"{_collector_url}/metrics/latest", timeout=1.0)
            if resp.status_code != 200:
                return
            cpu = float(resp.json().get("cpu_percent", 100.0))
        except Exception:
            return

        with _lock:
            if cpu < SCALE_DOWN_CPU_THRESHOLD:
                _low_cpu_streak += 1
            else:
                _low_cpu_streak = 0

            if _low_cpu_streak >= SCALE_DOWN_STREAK_SEC and _active_workers > BASE_WORKERS:
                old_workers = _active_workers
                _active_workers = max(BASE_WORKERS, _active_workers - 2)
                _low_cpu_streak = 0
                log_entry = {
                    "id": f"SCL-{int(time.time()*1000)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_id": "N/A",
                    "mode": _mode,
                    "action": f"Auto scale-down: CPU stable below {SCALE_DOWN_CPU_THRESHOLD:.0f}% for {SCALE_DOWN_STREAK_SEC}s",
                    "workers_before": old_workers,
                    "workers_after": _active_workers,
                    "status": "EXECUTED",
                    "approved_by": "SYSTEM (AUTO COOLDOWN)",
                }
                _scaling_history.append(log_entry)
                log.info("AUTO SCALE-DOWN: Workers %d -> %d (CPU cooldown)", old_workers, _active_workers)
            _sync_capacity_to_collector(_active_workers)

    def handle_alert(self, alert: dict):
        global _active_workers, _handled_alerts
        with _lock:
            _handled_alerts.add(alert.get("alert_id"))
            current_mode = _mode

        now_iso = datetime.now(timezone.utc).isoformat()
        alert_id = alert.get("alert_id", "ALT-UNKNOWN")
        metric = alert.get("metric", "resource")
        severity = alert.get("severity", "WARNING")
        lead_time = alert.get("lead_time_sec", 25.5)

        if current_mode == "AUTO":
            with _lock:
                if _active_workers >= MAX_WORKERS:
                    return
                old_workers = _active_workers
                _active_workers = min(MAX_WORKERS, _active_workers + 2)
                log_entry = {
                    "id": f"SCL-{int(time.time()*1000)}",
                    "timestamp": now_iso,
                    "alert_id": alert_id,
                    "mode": "AUTO",
                    "action": f"Auto-allocated +2 Cores ({metric} forecast breach)",
                    "workers_before": old_workers,
                    "workers_after": _active_workers,
                    "status": "EXECUTED",
                    "approved_by": "SYSTEM (AUTO)",
                }
                _scaling_history.append(log_entry)
                log.info("AUTO-SCALED: Workers %d -> %d (Alert: %s)", old_workers, _active_workers, alert_id)

            _sync_capacity_to_collector(_active_workers)
            try:
                import database
                database.record_remediation_event(log_entry)
            except Exception as exc:
                log.warning("Failed to persist scaling event to database: %s", exc)

        else:  # MANUAL MODE
            with _lock:
                if _active_workers >= MAX_WORKERS:
                    return

                req_id = f"REQ-{int(time.time()*1000)}"
                pending = {
                    "request_id": req_id,
                    "timestamp": now_iso,
                    "alert_id": alert_id,
                    "metric": metric,
                    "severity": severity,
                    "lead_time_sec": lead_time,
                    "current_workers": _active_workers,
                    "proposed_workers": min(MAX_WORKERS, _active_workers + 2),
                    "message": f"Proactive {severity} Alert: Forecast breach in ~{lead_time}s. Approve scaling to {min(MAX_WORKERS, _active_workers + 2)} cores?",
                    "created_at": time.time(),
                }
                _pending_approvals.append(pending)  # deque(maxlen=5) auto-caps
                log.info("MANUAL APPROVAL REQUIRED: %s", pending['message'])


def remediation_loop(engine: RemediationEngine, stop_event: threading.Event):
    log.info("Starting loop against %s", engine.alert_url)
    while not stop_event.is_set():
        engine.check_alerts()
        engine.check_cooldown()
        stop_event.wait(1.0)


def create_app():
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "remediation_service",
            "time": datetime.now(timezone.utc).isoformat(),
        })

    @app.route("/remediation/status", methods=["GET"])
    def status():
        with _lock:
            return jsonify({
                "mode": _mode,
                "active_workers": _active_workers,
                "base_workers": BASE_WORKERS,
                "max_workers": MAX_WORKERS,
                "pending_approvals": list(_pending_approvals),
                "scaling_history": list(reversed(_scaling_history[-20:])),
            })

    @app.route("/remediation/mode", methods=["POST"])
    def set_mode():
        global _mode
        payload = request.get_json(silent=True) or {}
        new_mode = payload.get("mode", "").upper()
        if new_mode in ("AUTO", "MANUAL"):
            with _lock:
                _mode = new_mode
            return jsonify({"status": "updated", "mode": _mode})
        return jsonify({"error": "Invalid mode. Use 'AUTO' or 'MANUAL'"}), 400

    @app.route("/remediation/approve", methods=["POST"])
    def approve():
        global _active_workers, _pending_approvals
        payload = request.get_json(silent=True) or {}
        req_id = payload.get("request_id")

        with _lock:
            target_req = None
            if req_id:
                for item in _pending_approvals:
                    if item["request_id"] == req_id:
                        target_req = item
                        break
            elif _pending_approvals:
                target_req = _pending_approvals[-1]

            if not target_req:
                return jsonify({"error": "Pending request not found"}), 404

            _pending_approvals.remove(target_req)
            old_workers = _active_workers
            _active_workers = min(MAX_WORKERS, _active_workers + 2)

            log_entry = {
                "id": f"SCL-{int(time.time()*1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": target_req.get("alert_id"),
                "mode": "MANUAL",
                "action": f"Manual scaling approved (+2 Cores)",
                "workers_before": old_workers,
                "workers_after": _active_workers,
                "status": "EXECUTED",
                "approved_by": "HUMAN OPERATOR",
            }
            _scaling_history.append(log_entry)

        _sync_capacity_to_collector(_active_workers)
        try:
            import database
            database.record_remediation_event(log_entry)
        except Exception:
            pass

        return jsonify({
            "status": "approved",
            "request_id": target_req["request_id"],
            "workers_before": old_workers,
            "active_workers": _active_workers,
        })

    @app.route("/remediation/decline", methods=["POST"])
    def decline():
        global _pending_approvals, _scaling_history
        payload = request.get_json(silent=True) or {}
        req_id = payload.get("request_id")

        with _lock:
            target_req = None
            if req_id:
                for item in _pending_approvals:
                    if item["request_id"] == req_id:
                        target_req = item
                        break
                _pending_approvals = deque([item for item in _pending_approvals if item["request_id"] != req_id], maxlen=5)
            else:
                if _pending_approvals:
                    target_req = _pending_approvals[-1]
                _pending_approvals.clear()

            log_entry = {
                "id": f"SCL-{int(time.time()*1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": target_req["alert_id"] if target_req else "N/A",
                "mode": "MANUAL",
                "action": "Scaling request dismissed (+0 Cores)",
                "workers_before": _active_workers,
                "workers_after": _active_workers,
                "status": "DECLINED",
                "approved_by": "HUMAN OPERATOR (DISMISSED)",
            }
            _scaling_history.append(log_entry)

        try:
            import database
            database.record_remediation_event(log_entry)
        except Exception:
            pass

        return jsonify({"status": "declined", "request_id": req_id})

    @app.route("/remediation/reset", methods=["POST"])
    def reset():
        global _active_workers, _pending_approvals, _handled_alerts
        with _lock:
            _active_workers = BASE_WORKERS
            _pending_approvals.clear()
            _handled_alerts.clear()
            log_entry = {
                "id": f"SCL-{int(time.time()*1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": "N/A",
                "mode": _mode,
                "action": "Reset edge capacity to base state (2 cores)",
                "workers_before": _active_workers,
                "workers_after": BASE_WORKERS,
                "status": "RESET",
                "approved_by": "OPERATOR RESET",
            }
            _scaling_history.append(log_entry)

        _sync_capacity_to_collector(BASE_WORKERS)
        return jsonify({"status": "reset", "active_workers": BASE_WORKERS})

    return app


def main():
    parser = argparse.ArgumentParser(description="PERM Remediation Service")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--alert-url", type=str, default=DEFAULT_ALERT_URL)
    parser.add_argument("--collector-url", type=str, default=DEFAULT_COLLECTOR_URL)
    args = parser.parse_args()

    global _collector_url
    _collector_url = args.collector_url

    engine = RemediationEngine(args.alert_url)

    stop_event = threading.Event()
    loop_thread = threading.Thread(target=remediation_loop, args=(engine, stop_event), daemon=True)
    loop_thread.start()

    app = create_app()
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        stop_event.set()
        loop_thread.join(timeout=2.0)



if __name__ == "__main__":
    main()
