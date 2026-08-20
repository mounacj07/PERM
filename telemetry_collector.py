"""
PERM - Predictive Edge Resource Management
Telemetry Collector

Flask-based edge monitoring service that:
  1. Serves a synthetic workload endpoint (/api/process) for Locust to drive
     traffic against (steady-state, ramp, burst profiles).
  2. Measures per-request latency via before/after request hooks.
  3. Runs a background daemon thread that samples CPU % and memory % via
     psutil every 1 second, independent of request traffic.
  4. Every 1 second, merges the latest system metrics + latency stats into
     a single telemetry record, keeps a rolling in-memory buffer, and
     appends the record to a CSV log for the downstream feature-engineering
     pipeline.

Run:
    python telemetry_collector.py --port 5000

Endpoints:
    GET  /health                       -> liveness check
    GET  /metrics/latest               -> most recent telemetry record
    GET  /metrics/history?seconds=60   -> records from the last N seconds
    POST /api/process                  -> synthetic workload endpoint
                                           body: {"load": <int 1-10>} optional
"""

import argparse
import csv
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

import psutil
from flask import Flask, jsonify, request, g

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SAMPLE_INTERVAL_SEC = 1.0          # how often we sample CPU/mem + emit a record
BUFFER_MAXLEN = 3600               # keep last 1 hour of 1-second records in memory
CSV_LOG_PATH = os.path.join(os.path.dirname(__file__), "telemetry_log.csv")
CSV_FIELDS = [
    "timestamp",
    "cpu_percent",
    "mem_percent",
    "avg_latency_ms",
    "p95_latency_ms",
    "request_count",
]

# --------------------------------------------------------------------------
# Thread-safe shared state
# --------------------------------------------------------------------------
_lock = threading.Lock()

# rolling buffer of finalized 1-second telemetry records (for API + CSV)
telemetry_buffer = deque(maxlen=BUFFER_MAXLEN)

# latency samples (in ms) accumulated during the current 1-second window
_current_window_latencies = []


def _record_latency(latency_ms: float) -> None:
    with _lock:
        _current_window_latencies.append(latency_ms)


def _drain_window_latencies():
    """Pop all latencies accumulated in the current window and reset it."""
    with _lock:
        latencies = _current_window_latencies[:]
        _current_window_latencies.clear()
    return latencies


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


# --------------------------------------------------------------------------
# CSV persistence
# --------------------------------------------------------------------------
def _ensure_csv_header():
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def _append_csv(record: dict):
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(record)


# --------------------------------------------------------------------------
# Background sampling thread
# --------------------------------------------------------------------------
def sampling_loop(stop_event: threading.Event):
    """
    Every SAMPLE_INTERVAL_SEC seconds:
      - read instantaneous CPU% and memory% from psutil
      - drain latency samples collected from live requests in that window
      - build one telemetry record, store it, persist it
    """
    # First call to cpu_percent(interval=None) always returns 0.0 / garbage
    # (it needs a baseline). Prime it once before the loop starts.
    psutil.cpu_percent(interval=None)

    next_tick = time.monotonic()
    while not stop_event.is_set():
        next_tick += SAMPLE_INTERVAL_SEC

        cpu_pct = psutil.cpu_percent(interval=None)
        mem_pct = psutil.virtual_memory().percent

        latencies = _drain_window_latencies()
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = _percentile(latencies, 95)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": round(cpu_pct, 2),
            "mem_percent": round(mem_pct, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "request_count": len(latencies),
        }

        with _lock:
            telemetry_buffer.append(record)
        _append_csv(record)

        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            stop_event.wait(sleep_time)
        else:
            # we're behind schedule (system overloaded); resync immediately
            next_tick = time.monotonic()


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
def create_app():
    app = Flask(__name__)

    @app.before_request
    def _start_timer():
        g._start_time = time.perf_counter()

    @app.after_request
    def _stop_timer(response):
        # Only track latency for the synthetic workload endpoint, so
        # /health and /metrics/* polling doesn't pollute the signal.
        if request.path == "/api/process" and hasattr(g, "_start_time"):
            elapsed_ms = (time.perf_counter() - g._start_time) * 1000.0
            _record_latency(elapsed_ms)
        return response

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

    @app.route("/metrics/latest", methods=["GET"])
    def metrics_latest():
        with _lock:
            if not telemetry_buffer:
                return jsonify({"error": "no telemetry collected yet"}), 503
            latest = telemetry_buffer[-1]
        return jsonify(latest)

    @app.route("/metrics/history", methods=["GET"])
    def metrics_history():
        seconds = request.args.get("seconds", default=60, type=int)
        seconds = max(1, min(seconds, BUFFER_MAXLEN))
        with _lock:
            records = list(telemetry_buffer)[-seconds:]
        return jsonify({"count": len(records), "records": records})

    @app.route("/api/process", methods=["POST"])
    def process():
        """
        Synthetic workload endpoint. `load` (1-10) controls how much CPU-bound
        work is done per request, letting Locust simulate steady-state, ramp,
        and burst profiles by varying request rate AND per-request load.
        """
        payload = request.get_json(silent=True) or {}
        load = payload.get("load", 1)
        try:
            load = int(load)
        except (TypeError, ValueError):
            load = 1
        load = max(1, min(load, 10))

        # CPU-bound busy-work scaled by `load`, standing in for real
        # request processing (e.g. inference, transforms, DB queries).
        iterations = load * 50_000
        acc = 0
        for i in range(iterations):
            acc += (i * i) % 7919

        return jsonify({"status": "processed", "load": load, "checksum": acc})

    return app


def main():
    parser = argparse.ArgumentParser(description="PERM telemetry collector")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    _ensure_csv_header()

    stop_event = threading.Event()
    sampler = threading.Thread(target=sampling_loop, args=(stop_event,), daemon=True)
    sampler.start()

    app = create_app()
    try:
        # threaded=True so /api/process requests can be timed concurrently
        # while the background sampler keeps ticking.
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        stop_event.set()
        sampler.join(timeout=2)


if __name__ == "__main__":
    main()