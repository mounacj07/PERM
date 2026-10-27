"""
PERM - Predictive Edge Resource Management
Web Dashboard Application

Serves an interactive ops-console Web UI for edge monitoring,
SLO alert management, live forecasting, and scenario testing.

Run:
    python web_dashboard/app.py --port 8000
"""

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, render_template, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("web_dashboard")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def atomic_write_json(file_path: str, data: Any) -> None:
    """Write data as JSON atomically using a temp file and os.replace."""
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise

CSV_EVENTS_PATH = os.path.join(PROJECT_ROOT, "controller_events.csv")
STATE_JSON_PATH = os.path.join(PROJECT_ROOT, "controller_state.json")
STATUS_JSON_PATH = os.path.join(PROJECT_ROOT, "controller_status.json")
LAUNCHER_LOG_PATH = os.path.join(PROJECT_ROOT, "results", "launcher_last.log")
CONTROLLER_LOG_PATH = os.path.join(PROJECT_ROOT, "results", "controller_last.log")
INFERENCE_LOG_PATH = os.path.join(PROJECT_ROOT, "results", "inference_last.log")


def _read_controller_events(since_iso: Optional[str] = None):
    events = []
    if os.path.exists(CSV_EVENTS_PATH):
        try:
            with open(CSV_EVENTS_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if since_iso:
                        t = r.get("time") or r.get("timestamp")
                        if t and t < since_iso:
                            continue
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


def _read_controller_status():
    if os.path.exists(STATUS_JSON_PATH):
        try:
            with open(STATUS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _read_launcher_log_tail(n: int = 15) -> List[str]:
    if not os.path.exists(LAUNCHER_LOG_PATH):
        return []
    try:
        with open(LAUNCHER_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.rstrip() for line in f.readlines()]
            return lines[-n:]
    except Exception:
        return []


def _read_inference_log_tail(n: int = 15) -> str:
    if not os.path.exists(INFERENCE_LOG_PATH):
        return ""
    try:
        with open(INFERENCE_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception:
        return ""


def get_system_logical_cpus() -> int:
    """Read logical CPUs from the system (defaulting to 16 if undetermined)."""
    try:
        cnt = os.cpu_count()
        if cnt and cnt > 0:
            return int(cnt)
    except Exception:
        pass
    return 16


def compute_worker_cpu_active(worker_cpu_percent: float, active_workers: int, cpu_count: Optional[int] = None) -> float:
    """
    Worker CPU (% of active workers) = clip(worker_cpu_percent * cpu_count / active_workers, 0, 100).
    Reads logical_cpus from system if cpu_count is None.
    """
    if active_workers <= 0:
        return 0.0
    if cpu_count is None:
        cpu_count = get_system_logical_cpus()
    val = (float(worker_cpu_percent) * float(cpu_count)) / float(active_workers)
    return round(float(max(0.0, min(100.0, val))), 2)


def get_controller_mode_label() -> str:
    """
    Mode label: read controller_status.json.
    Show "Demo controller: <mode> (forecast online/offline)".
    If the file is missing or older than 5 s, show "Controller not running".
    """
    if not os.path.exists(STATUS_JSON_PATH):
        return "Controller not running"

    try:
        mtime = os.path.getmtime(STATUS_JSON_PATH)
        now_epoch = time.time()
        file_age = now_epoch - mtime

        with open(STATUS_JSON_PATH, "r", encoding="utf-8") as f:
            status = json.load(f)

        ts_str = status.get("time")
        if ts_str:
            try:
                ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                age_ts = (now_dt - ts_dt).total_seconds()
                if age_ts > 5.0 or age_ts < -5.0:
                    return "Controller not running"
            except Exception:
                if file_age > 5.0:
                    return "Controller not running"
        else:
            if file_age > 5.0:
                return "Controller not running"

        mode = status.get("mode", "unknown")
        policy = status.get("policy")
        forecast_online = bool(status.get("forecast_online", False))
        fc_str = "forecast online" if forecast_online else "forecast offline"
        if policy == "predictive_only" and mode == "predictive":
            return f"Demo controller: predictive_only ({fc_str})"
        return f"Demo controller: {mode} ({fc_str})"
    except Exception:
        return "Controller not running"


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


def _safe_get(url: str, timeout: float = 1.5):
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
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    # Module state for controller and launcher
    valid_profiles = ["step", "ramp", "spike", "double_burst", "random_walk", "random_mix", "ramp_burst"]

    launcher_state = {
        "process": None,
        "profile": None,
        "seed": 44,
        "duration_sec": 125,
        "start_time": None,
        "status": "idle",
        "returncode": None,
        "pid": None,
        "last_elapsed": None,
        "note": None,
        "log_file": None,
        "controller_mode": None,
        "controller_policy": None,
        "last_run_summary": None,
    }
    launcher_lock = threading.RLock()
    app.launcher_state = launcher_state
    app.launcher_lock = launcher_lock

    controller_mgmt = {
        "process": None,
        "mode": None,
        "policy": None,
        "pid": None,
        "start_time": None,
        "log_file": None,
    }
    controller_lock = threading.RLock()

    inference_mgmt = {
        "process": None,
        "pid": None,
        "failed": False,
        "status": "idle",
        "log_file": None,
    }

    def _spawn_inference_service_bg():
        """H1: Spawn inference_service.py --port 5001 with PERM_FORECASTER=stagbdt in background thread."""
        with controller_lock:
            if inference_mgmt["process"] is not None and inference_mgmt["process"].poll() is None:
                return

            results_dir = os.path.join(PROJECT_ROOT, "results")
            os.makedirs(results_dir, exist_ok=True)
            try:
                inf_log = open(INFERENCE_LOG_PATH, "w", encoding="utf-8")
                inference_mgmt["log_file"] = inf_log
            except Exception as exc:
                log.warning("Could not open %s: %s", INFERENCE_LOG_PATH, exc)
                inf_log = subprocess.DEVNULL

            inf_env = os.environ.copy()
            inf_env["PERM_FORECASTER"] = "stagbdt"
            inf_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "inference_service.py"), "--port", "5001"]

            try:
                proc = subprocess.Popen(
                    inf_cmd,
                    cwd=PROJECT_ROOT,
                    env=inf_env,
                    stdout=inf_log,
                    stderr=inf_log,
                )
                inference_mgmt["process"] = proc
                inference_mgmt["pid"] = proc.pid
                inference_mgmt["failed"] = False
                inference_mgmt["status"] = "starting"
                log.info("Spawned inference_service.py (PID %d)", proc.pid)
            except Exception as exc:
                log.error("Failed to spawn inference_service.py: %s", exc)
                inference_mgmt["failed"] = True
                inference_mgmt["status"] = "failed"
                return

        t0 = time.time()
        healthy = False
        while time.time() - t0 < 15.0:
            time.sleep(0.5)
            try:
                r_check = _http_session.get(f"{inference_url}/health", timeout=1.0)
                if r_check.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                break

        if healthy:
            inference_mgmt["failed"] = False
            inference_mgmt["status"] = "online"
            log.info("Inference service became healthy on port 5001")
        else:
            inference_mgmt["failed"] = True
            inference_mgmt["status"] = "failed"
            log.warning("Inference service did not become healthy within 15s")

    def _ensure_inference_probe(timeout_sec: float = 15.0) -> Tuple[bool, str]:
        """
        P1: A predictive_only run REFUSES to start with a clear error unless inference /health
        is OK and a probe forecast call returns a non-null forecast. Auto-start inference first
        (spawn inference_service.py --port 5001 with PERM_FORECASTER=stagbdt via Popen env=,
        background thread, wait up to 15s, log to results/inference_last.log).
        """
        healthy = False
        try:
            r = _http_session.get(f"{inference_url}/health", timeout=1.0)
            if r.status_code == 200:
                healthy = True
        except Exception:
            healthy = False

        if not healthy:
            log.info("Inference service offline before predictive_only run; auto-starting in background thread...")
            threading.Thread(target=_spawn_inference_service_bg, daemon=True).start()
            t0 = time.time()
            while time.time() - t0 < timeout_sec:
                time.sleep(0.5)
                try:
                    r_check = _http_session.get(f"{inference_url}/health", timeout=1.0)
                    if r_check.status_code == 200:
                        healthy = True
                        break
                except Exception:
                    pass

        if not healthy:
            return False, "Inference service offline or /health did not return 200 within 15s"

        # Probe forecast call
        try:
            probe_payload = {"arrivals": [4.0] * 30, "horizon": 20}
            resp = _http_session.post(f"{inference_url}/forecast/demand", json=probe_payload, timeout=3.0)
            if resp.status_code != 200:
                return False, f"Inference probe call returned HTTP {resp.status_code}"
            data = resp.json()
            if data.get("forecast") is None or data.get("upper_bound") is None:
                return False, f"Inference probe returned null forecast: {data}"
            return True, "OK"
        except Exception as exc:
            return False, f"Inference probe failed: {exc}"

    def _compute_and_save_run_summary():
        """
        P2: Run summary card when a run ends, from controller_trace.csv and collector history:
        burst onset time (first second with arrivals > 2x baseline), first scale-up time,
        lead time in seconds (negative = late), peak workers, worker-seconds, over-provision
        ratio (allocated / needed capacity averaged over the run), seconds with pool util > 85%,
        seconds with queue_depth >= 15, dropped requests. Save raw numbers to
        results/run_summaries/<timestamp>.json. Numbers only, no interpretation text.
        """
        trace_path = os.path.join(PROJECT_ROOT, "results", "controller_trace.csv")
        trace_rows = []
        if os.path.exists(trace_path):
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        try:
                            trace_rows.append({
                                "t": float(row.get("t", 0.0)),
                                "arrivals": float(row.get("arrivals", 0.0)),
                                "smoothed_arrivals": float(row.get("smoothed_arrivals", 0.0)),
                                "workers": int(float(row.get("workers", 2))),
                                "util": float(row.get("util", 0.0)),
                            })
                        except Exception:
                            pass
            except Exception:
                pass

        if not trace_rows:
            return None

        # Filter out invalid / mock t < 0 rows and isolate the latest run
        valid_rows = []
        for row in trace_rows:
            t_val = row["t"]
            if t_val < 0.0:
                continue
            if valid_rows and t_val < valid_rows[-1]["t"] - 2.0:
                valid_rows = [row]
            else:
                valid_rows.append(row)
        trace_rows = valid_rows

        if not trace_rows:
            return None

        cap_per_worker = 7.89
        calib_file = os.path.join(PROJECT_ROOT, "testbed_calibration.json")
        if os.path.exists(calib_file):
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                    res_list = calib_data.get("results", [])
                    if res_list and "saturation_rps" in res_list[0]:
                        cap_per_worker = float(res_list[0]["saturation_rps"])
            except Exception:
                pass

        # 1. Burst onset time (first second with arrivals > 2x baseline)
        baseline = 4.0
        early_arrivals = [r["arrivals"] for r in trace_rows if 1.0 <= r["t"] <= 15.0]
        if early_arrivals:
            baseline = max(4.0, sum(early_arrivals) / len(early_arrivals))
        burst_threshold = 2.0 * baseline

        burst_onset_time = None
        for r in trace_rows:
            if r["t"] >= 1.0 and (r.get("smoothed_arrivals", r["arrivals"]) > burst_threshold or r["arrivals"] > burst_threshold):
                burst_onset_time = round(r["t"], 2)
                break

        # 2. First scale-up time
        base_workers = 2
        early_w = [r["workers"] for r in trace_rows if 1.0 <= r["t"] <= 5.0]
        if early_w:
            base_workers = min(early_w)

        first_scale_up_time = None
        for r in trace_rows:
            if r["t"] >= 1.0 and r["workers"] > base_workers:
                first_scale_up_time = round(r["t"], 2)
                break

        # 3. Lead time (negative = late)
        lead_time_seconds = None
        if burst_onset_time is not None and first_scale_up_time is not None:
            lead_time_seconds = round(burst_onset_time - first_scale_up_time, 2)

        # 4. Peak workers
        peak_workers = int(max(r["workers"] for r in trace_rows)) if trace_rows else init_workers

        # 5. Worker-seconds
        worker_seconds = round(sum(r["workers"] for r in trace_rows), 2)

        # 6. Over-provision ratio (allocated / needed capacity averaged over the run)
        ratios = []
        for r in trace_rows:
            needed_w = max(1, math.ceil(r["smoothed_arrivals"] / cap_per_worker))
            needed_cap = needed_w * cap_per_worker
            allocated_cap = r["workers"] * cap_per_worker
            ratios.append(allocated_cap / needed_cap)
        over_provision_ratio = round(sum(ratios) / len(ratios), 3) if ratios else 1.0

        # 7. Seconds with pool util > 85%
        seconds_util_gt_85 = sum(1 for r in trace_rows if r["util"] > 85.0)

        # 8. Collector history metrics
        telem_hist = _cached_state.get("telemetry_history", [])
        seconds_queue_ge_15 = sum(1 for h in telem_hist if int(h.get("queue_depth", 0)) >= 15)
        dropped_requests = sum(int(h.get("error_count", 0)) for h in telem_hist)

        now_ts = int(time.time())
        summary_payload = {
            "timestamp": now_ts,
            "burst_onset_time": burst_onset_time,
            "first_scale_up_time": first_scale_up_time,
            "lead_time_seconds": lead_time_seconds,
            "peak_workers": peak_workers,
            "worker_seconds": worker_seconds,
            "over_provision_ratio": over_provision_ratio,
            "seconds_util_gt_85": seconds_util_gt_85,
            "seconds_queue_ge_15": seconds_queue_ge_15,
            "dropped_requests": dropped_requests,
        }

        summaries_dir = os.path.join(PROJECT_ROOT, "results", "run_summaries")
        os.makedirs(summaries_dir, exist_ok=True)
        summary_file = os.path.join(summaries_dir, f"{now_ts}.json")
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
            log.info("Saved run summary to %s", summary_file)
        except Exception as exc:
            log.warning("Could not save run summary: %s", exc)

        launcher_state["last_run_summary"] = summary_payload
        return summary_payload

    @app.route("/")
    @app.route("/overview")
    @app.route("/live")
    def overview():
        if request.args.get("legacy") == "1":
            return render_template("index.html")
        return render_template("overview.html", active_page="overview")

    @app.route("/analysis")
    def analysis():
        tab = request.args.get("tab", "cpu").lower()
        if tab not in ("cpu", "latency", "forecast", "alerts"):
            tab = "cpu"
        return render_template("analysis.html", active_page="analysis", active_tab=tab)

    @app.route("/cpu")
    def cpu():
        return render_template("analysis.html", active_page="analysis", active_tab="cpu")

    @app.route("/latency")
    def latency():
        return render_template("analysis.html", active_page="analysis", active_tab="latency")

    @app.route("/forecast")
    def forecast():
        return render_template("analysis.html", active_page="analysis", active_tab="forecast")

    @app.route("/alerts")
    @app.route("/scaling")
    def alerts():
        return render_template("analysis.html", active_page="analysis", active_tab="alerts")

    @app.route("/experiments")
    @app.route("/run")
    def experiments():
        return render_template("experiments.html", active_page="experiments")

    @app.route("/api/dashboard/health", methods=["GET"])
    def dashboard_health():
        """B1. Health check for collector, inference, alert, remediation, and controller."""
        results = {}
        services = [
            ("collector", f"{collector_url}/health", "python telemetry_collector.py --port 5000 --workers 2"),
            ("inference", f"{inference_url}/health", "python inference_service.py --port 5001"),
            ("alert", f"{alert_url}/health", "python alert_engine.py --port 5002"),
            ("remediation", f"{remediation_url}/health", "python remediation_service.py --port 5003"),
        ]
        def _check_service(s_name, s_url, s_cmd):
            t0 = time.time()
            online = False
            latency = None
            try:
                r = _http_session.get(s_url, timeout=0.8)
                if r.status_code == 200:
                    online = True
                    latency = round((time.time() - t0) * 1000.0, 1)
            except Exception:
                pass
            return s_name, {
                "status": "online" if online else "offline",
                "latency_ms": latency,
                "command": s_cmd,
                "start_cmd": s_cmd,
            }

        with ThreadPoolExecutor(max_workers=len(services)) as pool:
            futures = [pool.submit(_check_service, name, url, cmd) for name, url, cmd in services]
            for f in as_completed(futures):
                s_name, s_res = f.result()
                results[s_name] = s_res

        # Controller health via controller_status.json
        ctrl_mode = get_controller_mode_label()
        ctrl_online = (ctrl_mode != "Controller not running")
        results["controller"] = {
            "status": "online" if ctrl_online else "offline",
            "latency_ms": 0.0 if ctrl_online else None,
            "command": "python demo_controller.py --mode predictive --initial-workers 2",
            "start_cmd": "python demo_controller.py --mode predictive --initial-workers 2",
            "mode_label": ctrl_mode,
        }
        return jsonify(results)

    @app.route("/api/dashboard/state", methods=["GET"])
    def get_state():
        global _cached_state

        fetches = {
            "telemetry_history":  (f"{collector_url}/metrics/history?seconds=60", 2.5),
            "prediction_history": (f"{inference_url}/predictions/history?seconds=60", 1.5),
            "alert_latest":       (f"{alert_url}/alerts/latest", 1.0),
            "alert_active":       (f"{alert_url}/alerts/active", 1.0),
            "alert_history":      (f"{alert_url}/alerts/history?count=20", 1.0),
            "remediation":        (f"{remediation_url}/remediation/status", 1.0),
            "capacity":           (f"{collector_url}/api/capacity", 1.5),
        }

        results = {}
        with ThreadPoolExecutor(max_workers=len(fetches)) as pool:
            future_to_key = {pool.submit(_safe_get, target[0], target[1]): key for key, target in fetches.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()

        history_fetch_ok = False
        with _cache_lock:
            telem_data = results.get("telemetry_history")
            if telem_data:
                recs = []
                if isinstance(telem_data, dict):
                    recs = telem_data.get("records") or telem_data.get("history") or []
                elif isinstance(telem_data, list):
                    recs = telem_data
                if recs:
                    _cached_state["telemetry_history"] = recs
                    _cached_state["telemetry_latest"] = recs[-1]
                    history_fetch_ok = True

            pred_data = results.get("prediction_history")
            if pred_data and pred_data.get("predictions"):
                preds = pred_data["predictions"]
                _cached_state["prediction_history"] = preds
                _cached_state["prediction_latest"] = preds[-1]

            if results.get("alert_latest"):
                _cached_state["alert_status"] = results["alert_latest"]
            if results.get("alert_active"):
                _cached_state["active_alerts"] = results["alert_active"].get("active_alerts", [])
            if results.get("alert_history"):
                _cached_state["alerts_history"] = results["alert_history"].get("alerts", [])

            if results.get("remediation"):
                _cached_state["remediation"] = results["remediation"]

            if results.get("capacity"):
                _cached_state["capacity"] = results["capacity"]

            snapshot = dict(_cached_state)

        # C4. Show only controller_events.csv rows newer than current run's start
        run_start_iso = None
        if launcher_state.get("start_time"):
            run_start_iso = datetime.fromtimestamp(launcher_state["start_time"], tz=timezone.utc).isoformat()

        ctrl_events = _read_controller_events(since_iso=run_start_iso)
        ctrl_state = _read_controller_state()
        ctrl_status = _read_controller_status()
        snapshot["controller_events"] = ctrl_events
        snapshot["controller_state"] = ctrl_state
        snapshot["controller_status"] = ctrl_status
        snapshot["capacity"] = _cached_state.get("capacity", {})

        # Mode label from controller_status.json
        mode_label = get_controller_mode_label()
        snapshot["mode_label"] = mode_label
        snapshot["model_mode_label"] = mode_label

        # B3 & F1. telemetry_age_sec and stale flag (true if newest sample is older than 5s or fetch failed with no history)
        telemetry_age_sec = None
        stale = True
        if snapshot.get("telemetry_latest"):
            telem = snapshot["telemetry_latest"]
            ts_str = telem.get("timestamp")
            if ts_str:
                try:
                    ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    else:
                        ts_dt = ts_dt.astimezone(timezone.utc)
                    now_dt = datetime.now(timezone.utc)
                    telemetry_age_sec = round(max(0.0, (now_dt - ts_dt).total_seconds()), 2)
                    stale = (telemetry_age_sec > 5.0) or (not history_fetch_ok and len(snapshot.get("telemetry_history", [])) == 0)
                except Exception:
                    stale = True
        snapshot["telemetry_age_sec"] = telemetry_age_sec
        snapshot["stale"] = stale

        # Dynamic logical CPUs
        logical_cpus = get_system_logical_cpus()
        snapshot["logical_cpus"] = logical_cpus

        # F1 & G1. Ensure both telemetry_history and history aliases are attached
        telemetry_hist = snapshot.get("telemetry_history", [])
        snapshot["history"] = telemetry_hist

        # Load capacity per worker from calibration
        cap_per_worker = 7.89
        calib_file = os.path.join(PROJECT_ROOT, "testbed_calibration.json")
        if os.path.exists(calib_file):
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                    res_list = calib_data.get("results", [])
                    if res_list and "saturation_rps" in res_list[0]:
                        cap_per_worker = float(res_list[0]["saturation_rps"])
            except Exception:
                pass
        snapshot["capacity_per_worker"] = cap_per_worker
        snapshot["forecast_horizon"] = 20

        # Normalization helper for telemetry records
        def _norm_record(r):
            if not isinstance(r, dict):
                return r
            arr = r.get("arrivals_count", r.get("arrival_rate_req_per_s", r.get("arrivals", 0.0)))
            r["arrivals_count"] = float(arr)
            r["arrival_rate_req_per_s"] = float(arr)
            r["arrivals"] = float(arr)

            errs = r.get("error_count", r.get("failed_requests_total", r.get("failed_requests", 0)))
            r["error_count"] = int(errs)
            r["failed_requests"] = int(errs)
            r["failed_requests_total"] = int(errs)
            r["dropped_requests_total"] = int(r.get("rejected_count", errs))

            if "system_cpu_percent" not in r:
                r["system_cpu_percent"] = float(r.get("cpu_percent", 0.0))

            if "worker_cpu_per_active_pct" not in r or r["worker_cpu_per_active_pct"] is None:
                r["worker_cpu_per_active_pct"] = compute_worker_cpu_active(
                    r.get("worker_cpu_percent", 0.0),
                    r.get("active_workers", 2),
                    cpu_count=logical_cpus
                )
            return r

        for r in telemetry_hist:
            _norm_record(r)

        worker_cpu_active = 0.0
        whole_cpu = 0.0
        latest_arr = 0.0
        w_cnt = 2
        telem = {}
        if snapshot.get("telemetry_latest"):
            telem = _norm_record(snapshot["telemetry_latest"])
            latest_arr = telem.get("arrivals_count", 0.0)
            w_cpu = float(telem.get("worker_cpu_percent", 0.0))
            w_cnt = int(telem.get("active_workers", 2))
            worker_cpu_active = compute_worker_cpu_active(w_cpu, w_cnt, cpu_count=logical_cpus)
            whole_cpu = float(telem.get("cpu_percent", 0.0))

        snapshot["arrivals"] = latest_arr
        snapshot["arrival_rate_req_per_s"] = latest_arr
        snapshot["arrivals_count"] = latest_arr
        snapshot["active_workers"] = w_cnt
        snapshot["queue_depth"] = int(telem.get("queue_depth", 0)) if snapshot.get("telemetry_latest") else 0
        snapshot["p95_latency_ms"] = float(telem.get("p95_latency_ms", 0.0)) if snapshot.get("telemetry_latest") else 0.0
        snapshot["error_count"] = int(telem.get("error_count", 0)) if snapshot.get("telemetry_latest") else 0
        snapshot["worker_cpu_pct_active"] = worker_cpu_active
        snapshot["whole_machine_cpu"] = whole_cpu
        snapshot["slo_latency_ms"] = 300.0

        # Read last 90 rows of controller trace if available
        trace_file = os.path.join(PROJECT_ROOT, "results", "controller_trace.csv")
        trace_rows = []
        if os.path.exists(trace_file):
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            trace_rows.append({
                                "t": float(row.get("t", 0.0)),
                                "arrivals": float(row.get("arrivals", 0.0)),
                                "forecast_raw": float(row.get("forecast_raw", 0.0)),
                                "forecast_upper": float(row.get("forecast_upper", 0.0)),
                                "horizon": int(float(row.get("horizon", 20))),
                                "workers": int(row.get("workers", 2)),
                                "approval_mode": row.get("approval_mode", "auto"),
                                "forecast_online": str(row.get("forecast_online", "True")).strip().lower() in ("true", "1", "yes"),
                            })
                        except Exception:
                            pass
                trace_rows = trace_rows[-90:]
            except Exception:
                pass
        snapshot["controller_trace"] = trace_rows

        horizon = 20
        if ctrl_state.get("forecast_horizon"):
            horizon = int(ctrl_state["forecast_horizon"])
        elif trace_rows and trace_rows[-1].get("horizon"):
            horizon = int(trace_rows[-1]["horizon"])
        snapshot["forecast_horizon"] = horizon

        pool_util = ctrl_state.get("smoothed_util")
        if pool_util is None:
            pool_util = worker_cpu_active
        snapshot["pool_utilization"] = pool_util if pool_util is not None else 0.0

        # H1: Check if predictive mode is running without forecasts (inference offline)
        ctrl_status_data = _read_controller_status()
        active_ctrl_mode = ctrl_status_data.get("mode") or controller_mgmt.get("mode")
        ctrl_fc_online = ctrl_status_data.get("forecast_online", False)
        is_ctrl_running = (ctrl_status_data.get("mode") is not None or (controller_mgmt.get("process") and controller_mgmt["process"].poll() is None))

        forecast_fallback = False
        forecast_banner = {"active": False, "message": "", "log_tail": ""}

        if active_ctrl_mode == "predictive" and is_ctrl_running:
            if not ctrl_fc_online or inference_mgmt.get("failed"):
                forecast_fallback = True
                log_tail = _read_inference_log_tail(15)
                forecast_banner = {
                    "active": True,
                    "message": "Predictive mode is running WITHOUT forecasts (inference offline); behaving as reactive",
                    "log_tail": log_tail,
                }

        snapshot["forecast_fallback"] = forecast_fallback
        snapshot["forecast_offline_banner"] = forecast_banner

        # P2: Forecast accuracy strip under chart: rolling MAE and bias of forecast vs actual arrivals
        # at the SAME target time, over the last 30 matured forecasts; "n/a" until 10 have matured.
        accuracy_info = {"mae": None, "bias": None, "matured_count": 0}
        if os.path.exists(trace_file):
            try:
                all_trace_rows = []
                with open(trace_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        try:
                            all_trace_rows.append({
                                "t": float(r.get("t", 0.0)),
                                "arrivals": float(r.get("arrivals", 0.0)),
                                "forecast_raw": float(r.get("forecast_raw", 0.0)),
                                "horizon": int(float(r.get("horizon", 20))),
                                "forecast_online": str(r.get("forecast_online", "True")).strip().lower() in ("true", "1", "yes"),
                            })
                        except Exception:
                            pass
                arrivals_by_sec = {int(round(r["t"])): r["arrivals"] for r in all_trace_rows}
                matured_pairs = []
                for r in all_trace_rows:
                    if r["forecast_online"] and r["forecast_raw"] > 0:
                        target_sec = int(round(r["t"] + r["horizon"]))
                        if target_sec in arrivals_by_sec:
                            actual_val = arrivals_by_sec[target_sec]
                            matured_pairs.append((r["forecast_raw"], actual_val))

                if len(matured_pairs) >= 10:
                    last_30 = matured_pairs[-30:]
                    mae_val = round(sum(abs(f - a) for f, a in last_30) / len(last_30), 2)
                    bias_val = round(sum((f - a) for f, a in last_30) / len(last_30), 2)
                    accuracy_info = {
                        "mae": mae_val,
                        "bias": bias_val,
                        "matured_count": len(matured_pairs),
                    }
                else:
                    accuracy_info["matured_count"] = len(matured_pairs)
            except Exception as exc:
                log.debug("Forecast accuracy calculation error: %s", exc)

        snapshot["forecast_accuracy"] = accuracy_info
        snapshot["run_summary"] = launcher_state.get("last_run_summary")
        snapshot["launcher_status"] = _get_launcher_status_dict()

        return jsonify(snapshot)

    @app.route("/api/forecast/runs", methods=["GET"])
    def list_forecast_runs():
        data_dir = os.path.join(PROJECT_ROOT, "data", "runs_locust")
        cfg_file = os.path.join(PROJECT_ROOT, "models", "stagbdt_final_config.json")

        training_ids = set()
        calib_ids = set()
        test_ids = set()
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                splits = cfg.get("splits", {})
                training_ids = set(splits.get("training_run_ids", []))
                calib_ids = set(splits.get("calibration_run_ids", []))
                test_ids = set(splits.get("test_run_ids", []))
            except Exception:
                pass

        runs = []
        if os.path.exists(data_dir):
            for fname in sorted(os.listdir(data_dir)):
                if fname.endswith(".csv") and not fname.startswith("sawtooth"):
                    run_id = fname[:-4]
                    is_train = run_id in training_ids
                    split = "test" if run_id in test_ids else ("calibration" if run_id in calib_ids else ("training" if is_train else "held-out"))
                    runs.append({
                        "run_id": run_id,
                        "filename": fname,
                        "is_training": is_train,
                        "split": split,
                    })
        return jsonify({"count": len(runs), "runs": runs})

    @app.route("/api/forecast/replay", methods=["GET"])
    def get_forecast_replay():
        run_id = request.args.get("run_id", "random_mix_seed20")
        horizon = request.args.get("horizon", default=20, type=int)

        cfg_file = os.path.join(PROJECT_ROOT, "models", "stagbdt_final_config.json")
        model_file = os.path.join(PROJECT_ROOT, f"models/stagbdt_h{horizon}_lean.json")
        if not (os.path.exists(cfg_file) and os.path.exists(model_file)):
            return jsonify({
                "model_available": False,
                "error": "model not available",
                "schema": {
                    "second": "int",
                    "current_arrivals": "float",
                    "raw_forecast": "float",
                    "upper_bound": "float",
                    "recommended_workers": "int",
                    "actual_workers_needed": "int"
                },
                "sample": [
                    {"second": 10, "current_arrivals": 4.0, "raw_forecast": 4.1, "upper_bound": 9.2, "recommended_workers": 2, "actual_workers_needed": 2}
                ]
            })

        try:
            from replay_demo import replay_run
            res = replay_run(run_id, horizon=horizon, save_artifacts=False)

            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            training_ids = set(cfg.get("splits", {}).get("training_run_ids", []))
            is_training = run_id in training_ids
            banner = (
                "Training run: This run was used during offline model training."
                if is_training
                else "ST-AGBDT trained offline; replay uses runs not used in training"
            )
            return jsonify({
                "model_available": True,
                "run_id": run_id,
                "horizon": horizon,
                "is_training": is_training,
                "banner": banner,
                "summary": {
                    "mae": res.get("mae", 0.0),
                    "miss_pct": res.get("miss_pct", 0.0),
                    "mean_extra": res.get("mean_extra_workers", 0.0),
                    "valid_steps": res.get("valid_steps", 0),
                },
                "records": res.get("records", []),
            })
        except Exception as exc:
            return jsonify({"model_available": False, "error": str(exc)}), 500

    @app.route("/api/forecast/live", methods=["GET"])
    def get_forecast_live():
        horizon = request.args.get("horizon", default=20, type=int)
        try:
            r = _http_session.get(f"{inference_url}/forecast/demand?horizon={horizon}", timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                data["status"] = "online"
                return jsonify(data)
            else:
                return jsonify({"status": "offline", "message": "forecast service offline"}), 200
        except Exception:
            return jsonify({"status": "offline", "message": "forecast service offline"}), 200

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

    # ------------------------------------------------------------------
    # PART C. Controller management endpoints
    # ------------------------------------------------------------------
    @app.route("/api/controller/status", methods=["GET"])
    def controller_status():
        with controller_lock:
            proc = controller_mgmt["process"]
            is_running = (proc is not None and proc.poll() is None)
            return jsonify({
                "status": "running" if is_running else "stopped",
                "mode": controller_mgmt["mode"] if is_running else None,
                "policy": controller_mgmt.get("policy") if is_running else None,
                "pid": proc.pid if is_running else None,
            })

    @app.route("/api/controller/start", methods=["POST"])
    def controller_start():
        data = request.get_json(silent=True) or {}
        raw_mode = data.get("mode")
        raw_policy = str(data.get("policy", "")).strip().lower()

        if raw_mode is None:
            mode = "predictive"
            policy = "predictive_only"
        elif raw_mode == "predictive_only":
            mode = "predictive"
            policy = "predictive_only"
        elif raw_mode in ("hybrid", "predictive"):
            mode = "predictive"
            policy = "predictive_only" if raw_policy == "predictive_only" else "hybrid"
        elif raw_mode in ("reactive", "capacity_aware"):
            mode = raw_mode
            policy = "hybrid"
        else:
            mode = raw_mode
            policy = "predictive_only" if raw_policy == "predictive_only" else "hybrid"

        approval_mode = str(data.get("approval_mode", "auto")).strip().lower()
        if mode not in ("predictive", "reactive", "capacity_aware"):
            return jsonify({"error": f"Invalid mode '{mode}'. Must be reactive, capacity_aware, or predictive"}), 400
        if policy not in ("hybrid", "predictive_only"):
            policy = "hybrid"

        with controller_lock:
            proc = controller_mgmt["process"]
            if proc is not None and proc.poll() is None:
                return jsonify({
                    "error": "Controller is already running",
                    "mode": controller_mgmt["mode"],
                    "policy": controller_mgmt.get("policy"),
                    "pid": proc.pid,
                }), 409

            # P1: A predictive_only run REFUSES to start with a clear error unless inference /health
            # is OK and a probe forecast call returns a non-null forecast.
            if mode == "predictive" and policy == "predictive_only":
                ok, err_msg = _ensure_inference_probe(timeout_sec=15.0)
                if not ok:
                    log.warning("Predictive_only start refused: %s", err_msg)
                    return jsonify({"error": f"Refusing to start predictive_only: {err_msg}"}), 503
            elif mode == "predictive":
                # Hybrid: auto-start inference in bg if offline
                inf_online = False
                try:
                    r_inf = _http_session.get(f"{inference_url}/health", timeout=1.0)
                    if r_inf.status_code == 200:
                        inf_online = True
                except Exception:
                    pass

                if not inf_online:
                    log.info("Predictive mode requested and inference offline. Spawning in background thread...")
                    threading.Thread(target=_spawn_inference_service_bg, daemon=True).start()

            # Spawn demo_controller.py
            results_dir = os.path.join(PROJECT_ROOT, "results")
            os.makedirs(results_dir, exist_ok=True)
            try:
                log_file = open(CONTROLLER_LOG_PATH, "w", encoding="utf-8")
                ctrl_cmd = [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "demo_controller.py"),
                    "--mode", mode,
                    "--policy", policy,
                    "--approval", approval_mode,
                    "--initial-workers", "2",
                    "--collector-url", collector_url,
                    "--inference-url", inference_url,
                ]
                proc = subprocess.Popen(
                    ctrl_cmd,
                    cwd=PROJECT_ROOT,
                    stdout=log_file,
                    stderr=log_file,
                )
                controller_mgmt["process"] = proc
                controller_mgmt["mode"] = mode
                controller_mgmt["policy"] = policy
                controller_mgmt["pid"] = proc.pid
                controller_mgmt["start_time"] = time.time()
                controller_mgmt["log_file"] = log_file

                log.info("Demo controller started (mode=%s, policy=%s, PID %d)", mode, policy, proc.pid)
                return jsonify({
                    "status": "running",
                    "mode": mode,
                    "policy": policy,
                    "pid": proc.pid,
                    "message": f"Demo controller started in {mode} mode ({policy})",
                })
            except Exception as e:
                log.error("Failed to spawn demo_controller: %s", e)
                return jsonify({"error": f"Failed to start controller: {e}"}), 500

    def _do_controller_stop():
        with controller_lock:
            proc = controller_mgmt["process"]
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
                except Exception as e:
                    log.warning("Error stopping controller: %s", e)
                if controller_mgmt.get("log_file"):
                    try:
                        controller_mgmt["log_file"].close()
                    except Exception:
                        pass
                controller_mgmt["process"] = None

                # Also terminate inference process if dashboard spawned it
                inf_proc = inference_mgmt.get("process")
                if inf_proc is not None and inf_proc.poll() is None:
                    try:
                        inf_proc.terminate()
                        try:
                            inf_proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            inf_proc.kill()
                            inf_proc.wait(timeout=1.0)
                    except Exception as e:
                        log.warning("Error stopping spawned inference service: %s", e)
                    if inference_mgmt.get("log_file"):
                        try:
                            inference_mgmt["log_file"].close()
                        except Exception:
                            pass
                    inference_mgmt["process"] = None
                    inference_mgmt["status"] = "stopped"

                return True
            return False

    @app.route("/api/controller/stop", methods=["POST"])
    def controller_stop():
        stopped = _do_controller_stop()
        if stopped:
            return jsonify({"status": "stopped", "message": "Demo controller stopped"})
        return jsonify({"status": "stopped", "message": "No active controller running"})

    # ------------------------------------------------------------------
    # PART A. Scenario launcher backend
    # ------------------------------------------------------------------
    def _compute_is_burst(profile: str, seed: int, elapsed: float) -> bool:
        if not profile or elapsed <= 0:
            return False
        prof = profile.lower()
        if prof == "spike":
            return 30.0 <= elapsed <= 45.0
        elif prof == "step":
            return 30.0 <= elapsed <= 55.0
        elif prof == "double_burst":
            return (30.0 <= elapsed <= 45.0) or (80.0 <= elapsed <= 95.0)
        elif prof == "ramp":
            return 30.0 <= elapsed <= 90.0
        elif prof == "ramp_burst":
            return 30.0 <= elapsed <= 70.0
        return False

    def _get_launcher_status_dict():
        _update_launcher_status()
        with launcher_lock:
            elapsed = None
            if launcher_state["start_time"] is not None:
                if launcher_state["status"] == "running":
                    elapsed = round(time.time() - launcher_state["start_time"], 1)
                else:
                    elapsed = launcher_state.get("last_elapsed", None)

            note = launcher_state.get("note")
            msg = note or (f"Scenario run is {launcher_state['status']}" if launcher_state["status"] != "idle" else "Ready to launch scenario")
            is_burst = False
            if launcher_state["status"] == "running" and elapsed is not None:
                is_burst = _compute_is_burst(launcher_state.get("profile", ""), launcher_state.get("seed", 44), elapsed)

            ctrl_status_data = _read_controller_status()
            active_ctrl_mode = ctrl_status_data.get("mode") or controller_mgmt.get("mode")
            ctrl_policy = ctrl_status_data.get("policy") or controller_mgmt.get("policy") or launcher_state.get("controller_policy")
            ctrl_fc_online = ctrl_status_data.get("forecast_online", False)
            forecast_fallback = bool(active_ctrl_mode == "predictive" and not ctrl_fc_online)

            return {
                "status": launcher_state["status"],
                "profile": launcher_state["profile"],
                "seed": launcher_state["seed"],
                "duration_sec": launcher_state["duration_sec"],
                "elapsed_sec": elapsed,
                "is_burst": is_burst,
                "returncode": launcher_state["returncode"],
                "note": note,
                "message": msg,
                "log_tail": _read_launcher_log_tail(15),
                "pid": launcher_state["pid"] if launcher_state["status"] == "running" else None,
                "valid_profiles": valid_profiles,
                "forecast_fallback": forecast_fallback,
                "controller_mode": launcher_state.get("controller_mode") or active_ctrl_mode,
                "controller_policy": ctrl_policy,
                "last_run_summary": launcher_state.get("last_run_summary"),
            }

    def _update_launcher_status():
        with launcher_lock:
            proc = launcher_state["process"]
            if proc is not None:
                ret = proc.poll()
                if ret is None:
                    launcher_state["status"] = "running"
                else:
                    if not isinstance(ret, int):
                        try:
                            ret = int(ret)
                        except Exception:
                            ret = 0
                    launcher_state["returncode"] = ret
                    now = time.time()
                    elapsed = (now - launcher_state["start_time"]) if launcher_state["start_time"] else 0.0
                    dur = launcher_state.get("duration_sec")
                    if dur and ret in (0, 1) and elapsed > dur:
                        elapsed = float(dur)
                    launcher_state["last_elapsed"] = round(elapsed, 1)

                    # A1. returncode 0 or 1 -> status "finished"; note on 1; other or <5s death -> "failed"
                    if elapsed < 5.0 and ret != 0:
                        launcher_state["status"] = "failed"
                        launcher_state["note"] = f"Process died within 5s of start (exit code {ret})"
                    elif ret == 0:
                        launcher_state["status"] = "finished"
                        launcher_state["note"] = None
                        if not launcher_state.get("last_run_summary"):
                            _compute_and_save_run_summary()
                    elif ret == 1:
                        launcher_state["status"] = "finished"
                        launcher_state["note"] = "Finished. Requests were dropped (expected during bursts)"
                        if not launcher_state.get("last_run_summary"):
                            _compute_and_save_run_summary()
                    else:
                        launcher_state["status"] = "failed"
                        launcher_state["note"] = f"Locust exited with error code {ret}"

                    if launcher_state.get("log_file"):
                        try:
                            launcher_state["log_file"].close()
                        except Exception:
                            pass
                    launcher_state["process"] = None

    @app.route("/api/launcher/status", methods=["GET"])
    def launcher_status():
        return jsonify(_get_launcher_status_dict())

    @app.route("/api/launcher/start", methods=["POST"])
    def launcher_start():
        _update_launcher_status()
        with launcher_lock:
            if launcher_state["process"] is not None and launcher_state["process"].poll() is None:
                return jsonify({
                    "error": "A run is already active. Refusing to start another run.",
                    "status": "running",
                    "profile": launcher_state["profile"],
                    "pid": launcher_state["pid"],
                }), 409

            data = request.get_json(silent=True) or {}
            profile = str(data.get("profile", "")).strip().lower()
            if profile not in valid_profiles:
                return jsonify({
                    "error": f"Invalid profile '{profile}'. Allowed profiles: {valid_profiles}",
                }), 400

            try:
                seed = int(data.get("seed", 44))
            except (ValueError, TypeError):
                seed = 44

            target_host = str(data.get("host", collector_url)).strip() or collector_url

            # A3. Preflight check 1: collector /health reachable at target_host
            try:
                r_health = _http_session.get(f"{target_host}/health", timeout=1.5)
                if r_health.status_code != 200:
                    return jsonify({
                        "error": f"Collector not reachable at {target_host}. Start: python telemetry_collector.py --port 5000 --workers 2"
                    }), 503
            except Exception:
                return jsonify({
                    "error": f"Collector not reachable at {target_host}. Start: python telemetry_collector.py --port 5000 --workers 2"
                }), 503

            # A3. Preflight check 2: locust in sys.executable
            if importlib.util.find_spec("locust") is None:
                return jsonify({
                    "error": f"locust is not installed in interpreter '{sys.executable}'. Run: python -m pip install locust"
                }), 500

            # A3. Preflight check 3: locustfile exists
            locustfile_name = "locustfile_ramp_burst.py" if profile == "ramp_burst" else "locustfile_perm.py"
            locustfile_path = os.path.join(PROJECT_ROOT, locustfile_name)
            if not os.path.exists(locustfile_path):
                return jsonify({
                    "error": f"{locustfile_name} not found at {locustfile_path}"
                }), 500

            # A4. Duration derivation: total_dur + 5s
            req_duration = data.get("duration")
            if req_duration is None or int(req_duration) <= 0:
                try:
                    if profile == "ramp_burst":
                        import ramp_burst
                        _, _, total_dur = ramp_burst.generate_schedule(profile, seed)
                        duration = int(total_dur + 5.0)
                    else:
                        import burst_test
                        _, _, total_dur = burst_test.generate_schedule(profile, seed)
                        duration = int(total_dur + 5.0)
                except Exception:
                    duration = 95 if profile == "ramp_burst" else 125
            else:
                duration = int(req_duration)

            # C3. Run with controller if requested (defaults to Predictive only)
            if data.get("run_with_controller"):
                raw_ctrl_mode = data.get("controller_mode")
                ctrl_policy = str(data.get("controller_policy", "")).strip().lower()
                if raw_ctrl_mode is None:
                    ctrl_mode = "predictive"
                    ctrl_policy = "predictive_only"
                elif raw_ctrl_mode == "predictive_only":
                    ctrl_mode = "predictive"
                    ctrl_policy = "predictive_only"
                elif raw_ctrl_mode in ("hybrid", "predictive"):
                    ctrl_mode = "predictive"
                    ctrl_policy = "predictive_only" if ctrl_policy == "predictive_only" else "hybrid"
                elif raw_ctrl_mode in ("reactive", "capacity_aware"):
                    ctrl_mode = raw_ctrl_mode
                    ctrl_policy = "hybrid"
                else:
                    ctrl_mode = raw_ctrl_mode
                    ctrl_policy = "predictive_only" if ctrl_policy == "predictive_only" else "hybrid"

                approval_mode = str(data.get("approval_mode", "auto")).strip().lower()

                # P1: A predictive_only run REFUSES to start with a clear error unless inference /health
                # is OK and a probe forecast call returns a non-null forecast.
                if ctrl_mode == "predictive" and ctrl_policy == "predictive_only":
                    ok, err_msg = _ensure_inference_probe(timeout_sec=15.0)
                    if not ok:
                        return jsonify({"error": f"Refusing to start predictive_only run: {err_msg}"}), 503

                # Also write control file immediately
                ctrl_file = os.path.join(PROJECT_ROOT, "results", "controller_control.json")
                os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
                atomic_write_json(ctrl_file, {"approval_mode": approval_mode, "ask_before_release": False, "policy": ctrl_policy})

                launcher_state["last_run_summary"] = None
                launcher_state["controller_mode"] = ctrl_mode
                launcher_state["controller_policy"] = ctrl_policy

                def _bg_start_ctrl(cmode, cpolicy, app_mode):
                    try:
                        with app.test_client() as c:
                            c.post("/api/controller/stop")
                            c.post("/api/controller/start", json={"mode": cmode, "policy": cpolicy, "approval_mode": app_mode})
                    except Exception as e:
                        log.warning("Could not auto-start controller: %s", e)
                threading.Thread(target=_bg_start_ctrl, args=(ctrl_mode, ctrl_policy, approval_mode), daemon=True).start()
            else:
                launcher_state["last_run_summary"] = None

            loc_env = os.environ.copy()
            loc_env["PERM_PROFILE"] = profile
            loc_env["PERM_SEED"] = str(seed)

            cmd = [
                sys.executable,
                "-m",
                "locust",
                "-f",
                locustfile_path,
                "--headless",
                "-u",
                "1",
                "-r",
                "1",
                "-t",
                f"{duration}s",
                "--host",
                target_host,
            ]

            results_dir = os.path.join(PROJECT_ROOT, "results")
            os.makedirs(results_dir, exist_ok=True)
            log_file = open(LAUNCHER_LOG_PATH, "w", encoding="utf-8")

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    env=loc_env,
                    stdout=log_file,
                    stderr=log_file,
                )
                launcher_state["process"] = proc
                launcher_state["profile"] = profile
                launcher_state["seed"] = seed
                launcher_state["duration_sec"] = duration
                launcher_state["start_time"] = time.time()
                launcher_state["status"] = "running"
                launcher_state["returncode"] = None
                launcher_state["pid"] = proc.pid
                launcher_state["last_elapsed"] = None
                launcher_state["note"] = None
                launcher_state["log_file"] = log_file

                log.info("Launcher started Locust for profile=%s seed=%d (PID %d, duration=%ds)", profile, seed, proc.pid, duration)
                st_dict = _get_launcher_status_dict()
                st_dict["message"] = f"Successfully launched scenario '{profile}'"
                return jsonify(st_dict)
            except Exception as e:
                log.error("Failed to spawn locust: %s", e)
                return jsonify({"error": f"Failed to spawn locust process: {e}"}), 500

    @app.route("/api/launcher/stop", methods=["POST"])
    def launcher_stop():
        data = request.get_json(silent=True) or {}
        with launcher_lock:
            proc = launcher_state["process"]
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
                except Exception as e:
                    log.warning("Error stopping process %s: %s", launcher_state.get("pid"), e)

                if launcher_state["start_time"]:
                    launcher_state["last_elapsed"] = round(time.time() - launcher_state["start_time"], 1)
                launcher_state["status"] = "stopped"
                launcher_state["returncode"] = proc.poll()
                if launcher_state.get("log_file"):
                    try:
                        launcher_state["log_file"].close()
                    except Exception:
                        pass
                launcher_state["process"] = None
                log.info("Launcher stopped scenario %s", launcher_state.get("profile"))

        # C3. Stop controller as well if requested (default: True)
        if data.get("stop_controller", True):
            _do_controller_stop()

        return jsonify({
            "status": "stopped",
            "profile": launcher_state.get("profile"),
            "message": "Scenario run stopped successfully",
        })

    # P2: Run summary endpoint
    @app.route("/api/run/summary", methods=["GET"])
    def get_run_summary():
        if launcher_state.get("last_run_summary"):
            return jsonify(launcher_state["last_run_summary"])
        summaries_dir = os.path.join(PROJECT_ROOT, "results", "run_summaries")
        if os.path.exists(summaries_dir):
            files = sorted(os.listdir(summaries_dir))
            if files:
                latest_f = os.path.join(summaries_dir, files[-1])
                try:
                    with open(latest_f, "r", encoding="utf-8") as f:
                        return jsonify(json.load(f))
                except Exception:
                    pass
        return jsonify({"summary": None})

    # ------------------------------------------------------------------
    # PART G3. Controller Auto / Manual Approval Endpoints
    # ------------------------------------------------------------------
    @app.route("/api/controller/approval_mode", methods=["GET", "POST"])
    def controller_approval_mode():
        ctrl_file = os.path.join(PROJECT_ROOT, "results", "controller_control.json")
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            mode = str(data.get("approval_mode", "auto")).lower()
            ask_release = bool(data.get("ask_before_release", False))
            cfg = {
                "approval_mode": mode if mode in ("auto", "manual") else "auto",
                "ask_before_release": ask_release,
                "updated_at": time.time(),
            }
            atomic_write_json(ctrl_file, cfg)
            return jsonify({"status": "ok", "config": cfg})

        # GET request
        cfg = {"approval_mode": "auto", "ask_before_release": False}
        if os.path.exists(ctrl_file):
            try:
                with open(ctrl_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        return jsonify(cfg)

    @app.route("/api/controller/pending", methods=["GET"])
    def controller_pending():
        pending_file = os.path.join(PROJECT_ROOT, "results", "controller_pending.json")
        if os.path.exists(pending_file):
            try:
                with open(pending_file, "r", encoding="utf-8") as f:
                    proposal = json.load(f)
                return jsonify({"pending": True, "proposal": proposal})
            except Exception:
                pass
        return jsonify({"pending": False, "proposal": None})

    @app.route("/api/controller/approve", methods=["POST"])
    def controller_approve():
        data = request.get_json(silent=True) or {}
        prop_id = data.get("id")
        appr_file = os.path.join(PROJECT_ROOT, "results", "controller_approval.json")
        payload = {"id": prop_id, "decision": "approve", "timestamp": time.time()}
        atomic_write_json(appr_file, payload)
        return jsonify({"status": "approved", "proposal_id": prop_id})

    @app.route("/api/controller/dismiss", methods=["POST"])
    def controller_dismiss():
        data = request.get_json(silent=True) or {}
        prop_id = data.get("id")
        appr_file = os.path.join(PROJECT_ROOT, "results", "controller_approval.json")
        payload = {"id": prop_id, "decision": "dismiss", "timestamp": time.time()}
        atomic_write_json(appr_file, payload)
        return jsonify({"status": "dismissed", "proposal_id": prop_id})

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
