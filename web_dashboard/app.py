"""
PERM - Predictive Edge Resource Management
Web Dashboard Application

Serves an interactive dark-glassmorphism Web UI for demonstrating edge monitoring,
proactive SLO alerts with ~25-30s lead time, and edge resource scaling (AUTO allocation vs MANUAL approval).

Run:
    python web_dashboard/app.py --port 8000
    python web_dashboard/app.py --port 8000 \
        --collector-url http://telemetry-collector:5000 \
        --inference-url http://inference-service:5001 \
        --alert-url http://alert-engine:5002 \
        --remediation-url http://remediation-service:5003
"""

import argparse
import csv
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, render_template, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("web_dashboard")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_EVENTS_PATH = os.path.join(PROJECT_ROOT, "controller_events.csv")
STATE_JSON_PATH = os.path.join(PROJECT_ROOT, "controller_state.json")

def _read_controller_events():
    events = []
    if os.path.exists(CSV_EVENTS_PATH):
        try:
            with open(CSV_EVENTS_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    events.append(r)
        except Exception:
            pass
    return events

def _read_controller_state():
    if os.path.exists(STATE_JSON_PATH):
        try:
            with open(STATE_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def compute_worker_cpu_active(worker_cpu_percent: float, active_workers: int, cpu_count: int = 16) -> float:
    """
    Worker CPU (% of active workers) = clip(worker_cpu_percent * cpu_count / active_workers, 0, 100).
    """
    if active_workers <= 0:
        return 0.0
    val = (float(worker_cpu_percent) * float(cpu_count)) / float(active_workers)
    return round(float(max(0.0, min(100.0, val))), 2)

# Defaults — overridable via CLI args (important for Docker deployments)
_DEFAULT_COLLECTOR_URL    = "http://127.0.0.1:5000"
_DEFAULT_INFERENCE_URL    = "http://127.0.0.1:5001"
_DEFAULT_ALERT_URL        = "http://127.0.0.1:5002"
_DEFAULT_REMEDIATION_URL  = "http://127.0.0.1:5003"

# Populated in main() and used by request handlers via closure
_service_urls = {
    "collector":   _DEFAULT_COLLECTOR_URL,
    "inference":   _DEFAULT_INFERENCE_URL,
    "alert":       _DEFAULT_ALERT_URL,
    "remediation": _DEFAULT_REMEDIATION_URL,
}


# Thread-safe cached state
_cache_lock = threading.Lock()
_cached_state = {
    "telemetry_latest":   None,
    "telemetry_history":  [],
    "prediction_latest":  None,
    "prediction_history": [],
    "alert_status":       {"status": "OK", "alert": None},
    "active_alerts":      [],
    "alerts_history":     [],
    "remediation": {
        "mode": "AUTO",
        "active_workers": 2,
        "base_workers": 2,
        "max_workers": 10,
        "pending_approvals": [],
        "scaling_history": [],
    },
}

# Reusable HTTP Session with connection pool
_http_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
_http_session.mount("http://", _adapter)
_http_session.mount("https://", _adapter)


def _safe_get(url: str, timeout: float = 0.8):
    """
    Fire a fast GET request and return the parsed JSON on success, or None on any error.
    """
    try:
        r = _http_session.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        log.debug("GET %s failed: %s", url, exc)
    return None


def create_app(service_urls: dict = None):
    urls = service_urls or _service_urls
    collector_url   = urls["collector"]
    inference_url   = urls["inference"]
    alert_url       = urls["alert"]
    remediation_url = urls["remediation"]

    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    @app.route("/overview")
    def overview():
        if request.args.get("legacy") == "1":
            return render_template("index.html")
        return render_template("overview.html", active_page="overview")

    @app.route("/cpu")
    def cpu():
        return render_template("cpu.html", active_page="cpu")

    @app.route("/latency")
    def latency():
        return render_template("latency.html", active_page="latency")

    @app.route("/forecast")
    def forecast():
        return render_template("forecast.html", active_page="forecast")

    @app.route("/alerts")
    @app.route("/scaling")
    def alerts():
        return render_template("alerts.html", active_page="alerts")

    @app.route("/experiments")
    def experiments():
        return render_template("experiments.html", active_page="experiments")

    @app.route("/api/dashboard/state", methods=["GET"])
    def get_state():
        global _cached_state

        # Define all upstream fetches — fired in parallel via ThreadPoolExecutor
        fetches = {
            "telemetry_history":  f"{collector_url}/metrics/history?seconds=60",
            "prediction_history": f"{inference_url}/predictions/history?seconds=60",
            "alert_latest":       f"{alert_url}/alerts/latest",
            "alert_active":       f"{alert_url}/alerts/active",
            "alert_history":      f"{alert_url}/alerts/history?count=20",
            "remediation":        f"{remediation_url}/remediation/status",
        }

        results = {}
        with ThreadPoolExecutor(max_workers=len(fetches)) as pool:
            future_to_key = {pool.submit(_safe_get, url): key for key, url in fetches.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()

        with _cache_lock:
            # Telemetry
            telem_data = results.get("telemetry_history")
            if telem_data and telem_data.get("records"):
                recs = telem_data["records"]
                _cached_state["telemetry_history"] = recs
                _cached_state["telemetry_latest"] = recs[-1]

            # Inference
            pred_data = results.get("prediction_history")
            if pred_data and pred_data.get("predictions"):
                preds = pred_data["predictions"]
                _cached_state["prediction_history"] = preds
                _cached_state["prediction_latest"] = preds[-1]

            # Alert Engine
            if results.get("alert_latest"):
                _cached_state["alert_status"] = results["alert_latest"]
            if results.get("alert_active"):
                _cached_state["active_alerts"] = results["alert_active"].get("active_alerts", [])
            if results.get("alert_history"):
                _cached_state["alerts_history"] = results["alert_history"].get("alerts", [])

            # Remediation
            if results.get("remediation"):
                _cached_state["remediation"] = results["remediation"]

            snapshot = dict(_cached_state)

        ctrl_events = _read_controller_events()
        ctrl_state = _read_controller_state()
        snapshot["controller_events"] = ctrl_events
        snapshot["controller_state"] = ctrl_state

        worker_cpu_active = 0.0
        whole_cpu = 0.0
        if snapshot.get("telemetry_latest"):
            telem = snapshot["telemetry_latest"]
            w_cpu = float(telem.get("worker_cpu_percent", 0.0))
            w_cnt = int(telem.get("active_workers", 2))
            worker_cpu_active = compute_worker_cpu_active(w_cpu, w_cnt, 16)
            whole_cpu = float(telem.get("cpu_percent", 0.0))

        snapshot["worker_cpu_pct_active"] = worker_cpu_active
        snapshot["whole_machine_cpu"] = whole_cpu
        snapshot["slo_latency_ms"] = 300.0

        pool_util = ctrl_state.get("smoothed_util")
        if pool_util is None:
            pool_util = worker_cpu_active
        snapshot["pool_utilization"] = pool_util if pool_util is not None else 0.0

        return jsonify(snapshot)

    @app.route("/api/scaling/mode", methods=["POST"])
    def set_mode():
        try:
            r = _http_session.post(f"{remediation_url}/remediation/mode", json=request.json, timeout=2.0)
            return jsonify(r.json()), r.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/scaling/approve", methods=["POST"])
    def approve_scaling():
        try:
            r = _http_session.post(f"{remediation_url}/remediation/approve", json=request.json, timeout=2.0)
            return jsonify(r.json()), r.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/scaling/decline", methods=["POST"])
    def decline_scaling():
        try:
            r = _http_session.post(f"{remediation_url}/remediation/decline", json=request.json, timeout=2.0)
            return jsonify(r.json()), r.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/scaling/reset", methods=["POST"])
    def reset_scaling():
        try:
            r = _http_session.post(f"{remediation_url}/remediation/reset", json=request.json, timeout=2.0)
            return jsonify(r.json()), r.status_code
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="PERM Web Dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--collector-url",   type=str, default=_DEFAULT_COLLECTOR_URL,
                        help="Telemetry Collector base URL")
    parser.add_argument("--inference-url",   type=str, default=_DEFAULT_INFERENCE_URL,
                        help="Inference Service base URL")
    parser.add_argument("--alert-url",       type=str, default=_DEFAULT_ALERT_URL,
                        help="Alert Engine base URL")
    parser.add_argument("--remediation-url", type=str, default=_DEFAULT_REMEDIATION_URL,
                        help="Remediation Service base URL")
    args = parser.parse_args()

    urls = {
        "collector":   args.collector_url,
        "inference":   args.inference_url,
        "alert":       args.alert_url,
        "remediation": args.remediation_url,
    }
    log.info("Dashboard connecting to services: %s", urls)

    app = create_app(service_urls=urls)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
