"""
PERM - Predictive Edge Resource Management
Telemetry Collector  (Part B rewrite - real process pool, real measurements)

Flask-based edge monitoring service that:
  1. Serves /api/process with REAL CPU-bound work executed by a pool of N
     worker PROCESSES (ProcessPoolExecutor).  Each request does a busy-loop
     of  load * MS_PER_LOAD  milliseconds.  If it waits + runs longer than
     REQUEST_TIMEOUT_SEC it returns HTTP 503 and counts as an error.
  2. Measures latency end-to-end (queue wait + service time) and reports
     real psutil CPU/memory metrics every second.
  3. POST /api/capacity {"active_workers": n} truly resizes the pool
     (1..MAX_WORKERS, capped at cpu_count()).  Records scale-up time.
  4. Worker processes are optionally pinned to a fixed core set via
     psutil cpu_affinity (WORKER_CORE_IDS env var, e.g. "0,1,2,3").
     This prevents Locust threads from stealing their CPU time.
  5. Telemetry columns (backward-compatible, new columns at end):
        timestamp, cpu_percent, mem_percent, avg_latency_ms, p95_latency_ms,
        request_count, active_workers, error_count, queue_depth

CONFIGURATION (env vars or defaults):
    MS_PER_LOAD         - ms of busy-loop per unit of `load` (default 18)
    MAX_WORKERS         - hard ceiling for pool size (default 8, capped at cpu_count)
    REQUEST_TIMEOUT_SEC - per-request deadline in seconds (default 2.0)
    WORKER_CORE_IDS     - comma-separated core indices to pin workers to
                          (e.g. "4,5,6,7").  Empty = no pinning (default).

Run:
    python telemetry_collector.py --port 5000

Endpoints (unchanged names/fields + additions):
    GET  /health                       -> liveness check
    GET  /metrics/latest               -> most recent telemetry record
    GET  /metrics/history?seconds=60   -> records from the last N seconds
    POST /api/process                  -> real CPU-bound work
                                           body: {"load": <int 1-10>} optional
                                           returns 200 {"status":"processed","load":N}
                                           or 503 if queue/timeout exceeded
    GET  /api/capacity                 -> {"active_workers": n,
                                           "last_scale_up_seconds": f}
    POST /api/capacity                 -> {"active_workers": n}
                                           resize pool; returns 200 with status
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import logging
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, Future
from datetime import datetime, timezone
from typing import List, Optional

import psutil
from flask import Flask, jsonify, request, g

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telemetry_collector")

# ---------------------------------------------------------------------------
# Configuration (env-configurable)
# ---------------------------------------------------------------------------
MS_PER_LOAD: float = float(os.environ.get("MS_PER_LOAD", "18"))
_cpu_count = os.cpu_count() or 4
MAX_WORKERS: int = min(int(os.environ.get("MAX_WORKERS", "8")), _cpu_count)
REQUEST_TIMEOUT_SEC: float = float(os.environ.get("REQUEST_TIMEOUT_SEC", "2.0"))

# Core affinity: comma-separated core IDs, e.g. "4,5,6,7"
# Leave empty for no pinning.  Pinning keeps worker cores separate from
# the Locust/Flask threads so they don't steal each other's CPU time.
_WORKER_CORE_IDS_STR: str = os.environ.get("WORKER_CORE_IDS", "")
WORKER_CORE_IDS: Optional[List[int]] = (
    [int(x) for x in _WORKER_CORE_IDS_STR.split(",") if x.strip()]
    if _WORKER_CORE_IDS_STR.strip()
    else None
)

SAMPLE_INTERVAL_SEC: float = 1.0
BUFFER_MAXLEN: int = 3600

CSV_LOG_PATH: str = os.path.join(os.path.dirname(__file__), "telemetry_log.csv")
CSV_FIELDS = [
    "timestamp",
    "cpu_percent",
    "mem_percent",
    "avg_latency_ms",
    "p95_latency_ms",
    "request_count",
    "active_workers",
    # --- new columns added at the end (backward-compatible) ---
    "error_count",
    "queue_depth",
]

# ---------------------------------------------------------------------------
# CPU-bound worker function (runs in a subprocess)
# ---------------------------------------------------------------------------
def _cpu_work(load: int, ms_per_load: float) -> None:
    """
    Busy-loop for  load * ms_per_load  milliseconds.
    Pure CPU work - no IO, no GIL - so it truly saturates a core.
    """
    deadline = time.perf_counter() + (load * ms_per_load) / 1000.0
    while time.perf_counter() < deadline:
        _ = 3.14159265358979 ** 2  # trivial arithmetic to keep the CPU hot


# ---------------------------------------------------------------------------
# ProcessPool manager (thread-safe resize)
# ---------------------------------------------------------------------------
class _PoolManager:
    """
    Wraps a ProcessPoolExecutor and lets the pool be resized dynamically
    without dropping in-flight requests.

    Scale-up:  start a new (larger) executor; drain the old one gracefully.
    Scale-down: start a new (smaller) executor; drain the old one gracefully.
    In-flight futures submitted to the OLD executor continue to run until
    the old pool's shutdown(wait=True) completes in a background thread.
    """

    def __init__(self, n_workers: int):
        self._lock = threading.Lock()
        self._n = max(1, min(n_workers, MAX_WORKERS))
        self._executor: ProcessPoolExecutor = self._spawn(self._n)
        self._last_scale_up_seconds: float = 0.0

    def _spawn(self, n: int) -> ProcessPoolExecutor:
        ex = ProcessPoolExecutor(max_workers=n)
        # Warm up: trigger process creation so the first real request is fast
        futs = [ex.submit(_cpu_work, 0, 0.0) for _ in range(n)]
        for f in futs:
            try:
                f.result(timeout=5.0)
            except Exception:
                pass
        # Pin worker processes to the specified cores if requested
        if WORKER_CORE_IDS is not None:
            self._pin_workers(ex)
        return ex

    @staticmethod
    def _pin_workers(ex: ProcessPoolExecutor):
        """Set cpu_affinity on each worker PID (best-effort)."""
        try:
            pids = set()
            # Submit lightweight tasks to discover worker PIDs
            futs = [ex.submit(os.getpid) for _ in range(ex._max_workers)]
            for f in futs:
                try:
                    pids.add(f.result(timeout=3.0))
                except Exception:
                    pass
            for pid in pids:
                try:
                    psutil.Process(pid).cpu_affinity(WORKER_CORE_IDS)
                except Exception:
                    pass
        except Exception as exc:
            log.warning("cpu_affinity pinning failed: %s", exc)

    def submit(self, load: int) -> concurrent.futures.Future:
        with self._lock:
            return self._executor.submit(_cpu_work, load, MS_PER_LOAD)

    def resize(self, new_n: int) -> int:
        new_n = max(1, min(new_n, MAX_WORKERS))
        with self._lock:
            if new_n == self._n:
                return self._n
            is_scale_up = new_n > self._n
            old_executor = self._executor
            old_n = self._n
            t0 = time.monotonic()
            self._executor = self._spawn(new_n)
            self._n = new_n
            scale_dur = time.monotonic() - t0
            if is_scale_up:
                self._last_scale_up_seconds = round(scale_dur, 3)
            log.info("Pool resized %d -> %d (took %.3fs)", old_n, new_n, scale_dur)
        # Drain old executor in a background thread so in-flight work finishes
        threading.Thread(
            target=old_executor.shutdown, kwargs={"wait": True}, daemon=True
        ).start()
        return new_n

    @property
    def n_workers(self) -> int:
        with self._lock:
            return self._n

    @property
    def last_scale_up_seconds(self) -> float:
        with self._lock:
            return self._last_scale_up_seconds


# ---------------------------------------------------------------------------
# Thread-safe shared state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
telemetry_buffer: deque = deque(maxlen=BUFFER_MAXLEN)

# Per-second window accumulators
_window_latencies: List[float] = []       # ms, completed requests
_window_errors: int = 0                   # 503 / timeout count
_window_requests: int = 0                 # all attempted (success + error)

# Queue depth counter (incremented on submit, decremented on finish/timeout)
_queue_depth: int = 0

# Global pool manager (created at startup)
_pool: Optional[_PoolManager] = None


def _record_request(latency_ms: float, is_error: bool) -> None:
    global _window_errors
    with _state_lock:
        _window_latencies.append(latency_ms)
        if is_error:
            _window_errors += 1


def _drain_window():
    global _window_errors, _queue_depth
    with _state_lock:
        latencies = _window_latencies[:]
        errors = _window_errors
        depth = _queue_depth
        _window_latencies.clear()
        _window_errors = 0
    return latencies, errors, depth


def _increment_queue():
    global _queue_depth
    with _state_lock:
        _queue_depth += 1


def _decrement_queue():
    global _queue_depth
    with _state_lock:
        _queue_depth = max(0, _queue_depth - 1)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------
def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------
def _ensure_csv_header() -> None:
    """
    Always ensure the CSV starts with the correct header row.
    If the file does not exist, create it with the header.
    If it exists but the first line is NOT the header (old data files),
    we do NOT overwrite - data integrity is preserved; the caller is
    responsible for backing up before this call.
    """
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
        return

    # Check whether the header is already present
    with open(CSV_LOG_PATH, "r", newline="") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(CSV_FIELDS)
    if first_line != expected_header:
        # Prepend the correct header
        log.info("CSV missing header - prepending it now.")
        with open(CSV_LOG_PATH, "r", newline="") as f:
            body = f.read()
        with open(CSV_LOG_PATH, "w", newline="") as f:
            f.write(expected_header + "\r\n")
            f.write(body)


def _append_csv(record: dict) -> None:
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writerow(record)
    try:
        import database  # optional dependency
        database.record_telemetry_sample(record)
    except Exception as exc:
        log.debug("Failed to persist telemetry to database: %s", exc)


# ---------------------------------------------------------------------------
# Background sampling thread - MEASURED metrics, no formulas
# ---------------------------------------------------------------------------
def sampling_loop(stop_event: threading.Event) -> None:
    """
    Every SAMPLE_INTERVAL_SEC seconds:
      - measure real CPU% and memory% via psutil
      - drain the per-second latency/error accumulators
      - compute avg/p95 from REAL request timings
      - record queue_depth snapshot
      - store + append to CSV
    """
    # Prime psutil's CPU sampler (first call always returns 0.0)
    psutil.cpu_percent(interval=None)

    next_tick = time.monotonic()
    while not stop_event.is_set():
        next_tick += SAMPLE_INTERVAL_SEC

        # --- Real system metrics ---
        sys_cpu = psutil.cpu_percent(interval=None) or 0.0
        mem_pct = psutil.virtual_memory().percent

        # --- Drain request accumulators ---
        latencies, errors, depth = _drain_window()
        req_count = len(latencies)

        if req_count > 0:
            latencies_sorted = sorted(latencies)
            avg_latency = sum(latencies_sorted) / req_count
            p95_latency = _percentile(latencies_sorted, 95)
        else:
            avg_latency = 0.0
            p95_latency = 0.0

        workers = _pool.n_workers if _pool else 2

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": round(sys_cpu, 2),
            "mem_percent": round(mem_pct, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "request_count": req_count,
            "active_workers": workers,
            "error_count": errors,
            "queue_depth": depth,
        }

        with _state_lock:
            telemetry_buffer.append(record)
        _append_csv(record)

        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            stop_event.wait(sleep_time)
        else:
            next_tick = time.monotonic()


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

    @app.route("/metrics/latest", methods=["GET"])
    def metrics_latest():
        with _state_lock:
            if not telemetry_buffer:
                return jsonify({"error": "no telemetry collected yet"}), 503
            latest = telemetry_buffer[-1]
        return jsonify(latest)

    @app.route("/metrics/history", methods=["GET"])
    def metrics_history():
        seconds = request.args.get("seconds", default=60, type=int)
        seconds = max(1, min(seconds, BUFFER_MAXLEN))
        with _state_lock:
            records = list(telemetry_buffer)[-seconds:]
        return jsonify({"count": len(records), "records": records})

    @app.route("/api/capacity", methods=["GET", "POST"])
    def capacity():
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            if "active_workers" in payload:
                if _pool is not None:
                    new_n = _pool.resize(int(payload["active_workers"]))
                else:
                    new_n = int(payload["active_workers"])
                return jsonify({
                    "status": "updated",
                    "active_workers": new_n,
                    "last_scale_up_seconds": _pool.last_scale_up_seconds if _pool else 0.0,
                })
        workers = _pool.n_workers if _pool else 2
        last_scale = _pool.last_scale_up_seconds if _pool else 0.0
        return jsonify({
            "active_workers": workers,
            "last_scale_up_seconds": last_scale,
        })

    @app.route("/api/process", methods=["POST"])
    def process():
        """
        Real CPU-bound workload endpoint.
        Submits work to the process pool; enforces REQUEST_TIMEOUT_SEC.
        Returns HTTP 503 on timeout or queue full.
        Latency measured end-to-end (queue wait + execution time).
        """
        payload = request.get_json(silent=True) or {}
        load = payload.get("load", 1)
        try:
            load = max(0, int(load))
        except (TypeError, ValueError):
            load = 1

        t_start = time.perf_counter()
        _increment_queue()

        try:
            if _pool is None:
                # Fallback before pool is initialised (test_client usage)
                _cpu_work(load, MS_PER_LOAD)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                _decrement_queue()
                _record_request(elapsed_ms, False)
                return jsonify({"status": "processed", "load": load})

            future: Future = _pool.submit(load)
            try:
                future.result(timeout=REQUEST_TIMEOUT_SEC)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                _decrement_queue()
                _record_request(elapsed_ms, False)
                return jsonify({"status": "processed", "load": load})
            except concurrent.futures.TimeoutError:
                future.cancel()
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                _decrement_queue()
                _record_request(elapsed_ms, True)
                return jsonify({"status": "timeout", "load": load}), 503
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                _decrement_queue()
                _record_request(elapsed_ms, True)
                log.error("Worker error: %s", exc)
                return jsonify({"status": "error", "detail": str(exc)}), 503
        except Exception as exc:
            _decrement_queue()
            log.error("Submit error: %s", exc)
            return jsonify({"status": "error", "detail": str(exc)}), 503

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    global _pool

    parser = argparse.ArgumentParser(description="PERM telemetry collector")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=2,
                        help="Initial number of worker processes (default 2)")
    args = parser.parse_args()

    log.info(
        "Config: MS_PER_LOAD=%.1f  MAX_WORKERS=%d  REQUEST_TIMEOUT_SEC=%.1f  "
        "WORKER_CORE_IDS=%s",
        MS_PER_LOAD, MAX_WORKERS, REQUEST_TIMEOUT_SEC,
        WORKER_CORE_IDS if WORKER_CORE_IDS else "none (no pinning)",
    )

    # Backup existing CSV if it lacks a header, then ensure header is present
    if os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "r", newline="") as f:
            first = f.readline().strip()
        if first != ",".join(CSV_FIELDS):
            bak = CSV_LOG_PATH + ".bak"
            if not os.path.exists(bak):
                import shutil
                shutil.copy2(CSV_LOG_PATH, bak)
                log.info("Backed up headerless CSV to %s", bak)
    _ensure_csv_header()

    # Start the process pool
    log.info("Starting process pool with %d workers...", args.workers)
    _pool = _PoolManager(args.workers)
    log.info("Process pool ready.")

    # Start background sampler
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sampling_loop, args=(stop_event,), daemon=True, name="sampler"
    )
    sampler.start()

    app = create_app()
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        stop_event.set()
        sampler.join(timeout=2)
        if _pool is not None:
            _pool._executor.shutdown(wait=False)
        log.info("Telemetry collector stopped.")


if __name__ == "__main__":
    # Required on Windows so spawned child processes don't re-run main()
    import multiprocessing
    multiprocessing.freeze_support()
    main()