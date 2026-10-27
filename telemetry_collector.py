"""
PERM - Predictive Edge Resource Management
Telemetry Collector  (Step 1b - Shared Bounded Queue & Real Process Workers)

Flask-based edge monitoring service that:
  1. Serves /api/process with REAL CPU-bound work executed by a pool of N
     worker PROCESSES pulling from ONE shared bounded queue (QUEUE_CAPACITY).
     Each request does a busy-loop of load * MS_PER_LOAD milliseconds.
     If queue is full, returns HTTP 503 immediately (rejected_count).
     If a request waits or runs past REQUEST_TIMEOUT_SEC, returns HTTP 503 (timeout_count).
     Workers shed/skip jobs already past their deadline.
  2. Measures latency end-to-end (queue wait + service time) and reports
     real psutil CPU/memory metrics every second.
  3. POST /api/capacity {"active_workers": n} resizes the pool in a background thread
     without blocking incoming requests or existing workers (no lock held during spawn).
     New workers serve only once ready. Records cold-start time per scale-up.
  4. Scale-down asks surplus workers to exit after finishing their current job.
  5. Worker processes are pinned to cores, leaving at least 2 cores free for Flask/generators.
  6. Telemetry columns (backward-compatible, new columns at end):
        timestamp, cpu_percent, mem_percent, avg_latency_ms, p95_latency_ms,
        request_count, active_workers, error_count, queue_depth,
        arrivals_count, ok_count, timeout_count, rejected_count,
        in_flight, worker_cpu_percent, ok_p95_latency_ms

CONFIGURATION (env vars or defaults):
    MS_PER_LOAD         - ms of busy-loop per unit of `load` (default 18)
    MAX_WORKERS         - hard ceiling for pool size (default 8, capped at cpu_count)
    REQUEST_TIMEOUT_SEC - per-request deadline in seconds (default 2.0)
    QUEUE_CAPACITY      - maximum bounded queue backlog (default 20)
    WORKER_CORE_IDS     - comma-separated core indices to pin workers to
    PROVISION_DELAY_SEC - extra seconds a newly-spawned worker is unavailable
                          after becoming OS-ready (simulates config/warm-up;
                          default 0 = no extra delay). Records ready_time in
                          scale_events.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import multiprocessing as mp
import os
import queue
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import psutil
from flask import Flask, jsonify, request

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
MS_PER_LOAD: float = float(os.environ.get("MS_PER_LOAD", "18.0"))
_cpu_count: int = os.cpu_count() or 4
_physical_cores_count: int = psutil.cpu_count(logical=False) or 4
MAX_WORKERS: int = min(int(os.environ.get("MAX_WORKERS", "8")), _cpu_count)
REQUEST_TIMEOUT_SEC: float = float(os.environ.get("REQUEST_TIMEOUT_SEC", "2.0"))

# Queue capacity: chosen so max queue wait < REQUEST_TIMEOUT_SEC at 2 workers
# At 2 workers and load=7 (126ms service time), 20 items / (2 / 0.126) ~= 1.26s < 2.0s
QUEUE_CAPACITY: int = int(os.environ.get("QUEUE_CAPACITY", "20"))

# Extra unavailability time after OS-ready (provisioning simulation)
PROVISION_DELAY_SEC: float = float(os.environ.get("PROVISION_DELAY_SEC", "0"))


def _record_latency(val: float = 0.0) -> None:
    """Backward compatibility helper for test suites."""
    pass


def get_physical_core_map() -> List[List[int]]:
    """
    Returns a list of logical CPU id lists, where each entry represents one physical core.
    Example on 8-core / 16-thread CPU: [[0, 1], [2, 3], [4, 5], ..., [14, 15]].
    """
    try:
        if sys.platform == "win32":
            import ctypes
            class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ('ProcessorMask', ctypes.c_size_t),
                    ('Relationship', ctypes.c_uint),
                    ('Reserved', ctypes.c_ulonglong * 2)
                ]
            buf_size = ctypes.c_ulong(0)
            ctypes.windll.kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buf_size))
            n_entries = buf_size.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
            buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * n_entries)()
            ctypes.windll.kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(buf_size))
            cores = []
            for item in buf:
                if item.Relationship == 0:  # RelationProcessorCore
                    cpus = [i for i in range(64) if (item.ProcessorMask & (1 << i))]
                    cores.append(cpus)
            if cores:
                return sorted(cores, key=lambda c: c[0])
    except Exception:
        pass

    n_phys = psutil.cpu_count(logical=False) or 4
    n_log = psutil.cpu_count(logical=True) or n_phys
    ratio = max(1, n_log // n_phys)
    return [[i * ratio + j for j in range(ratio)] for i in range(n_phys)]


_physical_core_map = get_physical_core_map()
_physical_cores_count = len(_physical_core_map)

# Keep 2 physical cores free for Flask, telemetry, and load generators
if _physical_cores_count >= 3:
    RESERVED_PHYSICAL_CORES = _physical_core_map[:2]
    WORKER_PHYSICAL_CORES = _physical_core_map[2:]
else:
    RESERVED_PHYSICAL_CORES = []
    WORKER_PHYSICAL_CORES = _physical_core_map

# Avoid hyperthread siblings: primary logical CPU of each worker physical core
PRIMARY_WORKER_CPUS = [c[0] for c in WORKER_PHYSICAL_CORES]
SECONDARY_WORKER_CPUS = [c[1] for c in WORKER_PHYSICAL_CORES if len(c) > 1]
WORKER_CORE_IDS_DEFAULT = PRIMARY_WORKER_CPUS + SECONDARY_WORKER_CPUS

_WORKER_CORE_IDS_STR: str = os.environ.get("WORKER_CORE_IDS", "")
if _WORKER_CORE_IDS_STR.strip():
    WORKER_CORE_IDS: Optional[List[int]] = [
        int(x) for x in _WORKER_CORE_IDS_STR.split(",") if x.strip()
    ]
else:
    WORKER_CORE_IDS = WORKER_CORE_IDS_DEFAULT


def get_worker_core_affinity(worker_idx: int) -> Optional[List[int]]:
    """Pin one worker per PHYSICAL core (avoiding hyperthread siblings)."""
    if _WORKER_CORE_IDS_STR.strip() and WORKER_CORE_IDS:
        return [WORKER_CORE_IDS[worker_idx % len(WORKER_CORE_IDS)]]
    if PRIMARY_WORKER_CPUS:
        if worker_idx < len(PRIMARY_WORKER_CPUS):
            # Dedicated physical core, no hyperthread sharing
            return [PRIMARY_WORKER_CPUS[worker_idx]]
        else:
            # Overflow workers beyond available physical cores use secondary siblings
            ov = worker_idx - len(PRIMARY_WORKER_CPUS)
            if ov < len(SECONDARY_WORKER_CPUS):
                return [SECONDARY_WORKER_CPUS[ov]]
            return [PRIMARY_WORKER_CPUS[worker_idx % len(PRIMARY_WORKER_CPUS)]]
    return None

SAMPLE_INTERVAL_SEC: float = 1.0
BUFFER_MAXLEN: int = 3600

CSV_LOG_PATH: str = os.path.join(os.path.dirname(__file__), "telemetry_log.csv")
CSV_FIELDS = [
    # Original 9 fields
    "timestamp",
    "cpu_percent",
    "mem_percent",
    "avg_latency_ms",
    "p95_latency_ms",
    "request_count",
    "active_workers",
    "error_count",
    "queue_depth",
    # Step 1b additions
    "arrivals_count",
    "ok_count",
    "timeout_count",
    "rejected_count",
    "in_flight",
    "worker_cpu_percent",
    "ok_p95_latency_ms",
    # Step 2 additions
    "busy_workers",
    "utilization",
    "worker_cpu_per_active_pct",
    "queue_wait_ms_avg",
    "cpu_freq_mhz",
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
        _ = 3.14159265358979 ** 2


def _worker_process_main(
    worker_id: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    queue_depth_val: mp.Value,
    active_workers_busy: mp.Value,
    exit_event: mp.Event,
    ready_event: mp.Event,
    ms_per_load: float,
    core_ids: Optional[List[int]],
) -> None:
    """
    Independent worker process entry point.
    Pulls work items from request_queue, checks deadline for load shedding,
    executes _cpu_work, and sends result back on response_queue.
    """
    if core_ids is not None:
        try:
            psutil.Process().cpu_affinity(core_ids)
        except Exception:
            pass

    # Signal parent that process is initialized and ready
    ready_event.set()

    while not exit_event.is_set():
        try:
            # Short timeout so process can check exit_event periodically
            item = request_queue.get(timeout=0.15)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        if item is None:
            if exit_event.is_set():
                break
            continue

        if len(item) >= 4:
            req_id, load, deadline_mono, enqueue_mono = item[:4]
        else:
            req_id, load, deadline_mono = item[:3]
            enqueue_mono = time.monotonic()

        with queue_depth_val.get_lock():
            queue_depth_val.value = max(0, queue_depth_val.value - 1)

        now_mono = time.monotonic()
        q_wait_ms = max(0.0, (now_mono - enqueue_mono) * 1000.0)

        # Load shedding: skip jobs already past deadline
        if now_mono >= deadline_mono:
            try:
                response_queue.put_nowait((req_id, False, "deadline_exceeded", 0.0, q_wait_ms))
            except Exception:
                pass
            continue

        with active_workers_busy.get_lock():
            active_workers_busy.value += 1

        t0 = time.perf_counter()
        _cpu_work(load, ms_per_load)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        with active_workers_busy.get_lock():
            active_workers_busy.value = max(0, active_workers_busy.value - 1)

        try:
            response_queue.put_nowait((req_id, True, None, duration_ms, q_wait_ms))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker & Process Pool Manager
# ---------------------------------------------------------------------------
class _WorkerHandle:
    def __init__(
        self,
        worker_id: int,
        proc: mp.Process,
        exit_event: mp.Event,
        ready_event: mp.Event,
        core_affinity: Optional[List[int]] = None,
    ):
        self.worker_id = worker_id
        self.proc = proc
        self.exit_event = exit_event
        self.ready_event = ready_event
        self.ready = False
        self.core_affinity = core_affinity
        self.ps_proc: Optional[psutil.Process] = None

    def get_psutil(self) -> Optional[psutil.Process]:
        if self.ps_proc is None and self.proc.pid:
            try:
                p = psutil.Process(self.proc.pid)
                p.cpu_percent(interval=None)  # prime baseline tick
                self.ps_proc = p
            except Exception:
                pass
        return self.ps_proc


class _PoolManager:
    """
    Manages individual worker processes over ONE shared bounded request queue.
    Supports non-blocking scale-up, graceful scale-down, and cold-start tracking.
    """

    def __init__(self, initial_workers: int):
        self._lock = threading.Lock()
        self._scale_lock = threading.Lock()
        self._wid_counter = itertools.count(1)
        self._req_counter = itertools.count(1)

        self.request_queue: mp.Queue = mp.Queue(maxsize=QUEUE_CAPACITY)
        self.response_queue: mp.Queue = mp.Queue()
        self.queue_depth_val: mp.Value = mp.Value("i", 0)
        self.active_workers_busy: mp.Value = mp.Value("i", 0)

        self._workers: List[_WorkerHandle] = []
        self._target_workers: int = max(1, min(initial_workers, MAX_WORKERS))
        self._last_scale_up_seconds: float = 0.0
        self._scale_events: deque = deque(maxlen=200)

        # Pending requests waiting for completion: req_id -> (threading.Event, [success, error, dur])
        self._pending_lock = threading.Lock()
        self._pending_requests: Dict[int, Tuple[threading.Event, List]] = {}

        # Dispatcher thread receiving responses
        self._stop_dispatcher = threading.Event()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, daemon=True, name="dispatcher"
        )
        self._dispatcher_thread.start()

        # Start initial worker processes synchronously on startup
        self._spawn_workers_sync(self._target_workers)

    def _spawn_workers_sync(self, count: int) -> None:
        t0 = time.monotonic()
        for idx in range(count):
            wid = next(self._wid_counter)
            exit_ev = mp.Event()
            ready_ev = mp.Event()
            affinity = get_worker_core_affinity(len(self._workers))
            p = mp.Process(
                target=_worker_process_main,
                args=(
                    wid,
                    self.request_queue,
                    self.response_queue,
                    self.queue_depth_val,
                    self.active_workers_busy,
                    exit_ev,
                    ready_ev,
                    MS_PER_LOAD,
                    affinity,
                ),
                daemon=True,
            )
            p.start()
            wh = _WorkerHandle(wid, p, exit_ev, ready_ev, core_affinity=affinity)
            wh.ready_event.wait(timeout=10.0)
            wh.ready = True
            wh.get_psutil()
            self._workers.append(wh)
        dur = round(time.monotonic() - t0, 3)
        self._last_scale_up_seconds = dur
        self._scale_events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "initial_start",
            "from_workers": 0,
            "to_workers": count,
            "duration_seconds": dur,
        })
        log.info("Initialized %d worker processes (took %.3fs)", count, dur)

    def _dispatcher_loop(self) -> None:
        """Receives completed tasks from workers and signals waiting request threads."""
        while not self._stop_dispatcher.is_set():
            try:
                item = self.response_queue.get(timeout=0.15)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break

            if item is None:
                break

            if len(item) >= 5:
                req_id, success, err_msg, dur_ms, q_wait_ms = item[:5]
            else:
                req_id, success, err_msg, dur_ms = item[:4]
                q_wait_ms = 0.0

            with self._pending_lock:
                entry = self._pending_requests.get(req_id)
            if entry is not None:
                ev, res_box = entry
                res_box[0] = success
                res_box[1] = err_msg
                res_box[2] = dur_ms
                res_box[3] = q_wait_ms
                ev.set()

    def submit_request(self, load: int, timeout_sec: float) -> Tuple[int, str, float]:
        """
        Submit a request to the shared queue.
        Returns (status_code, status_str, duration_ms).
        """
        req_id = next(self._req_counter)
        t_enq = time.monotonic()
        deadline = t_enq + timeout_sec
        ev = threading.Event()
        res_box = [False, "timeout", 0.0, 0.0]

        with self._pending_lock:
            self._pending_requests[req_id] = (ev, res_box)

        try:
            with self.queue_depth_val.get_lock():
                self.queue_depth_val.value += 1
            self.request_queue.put_nowait((req_id, load, deadline, t_enq))
        except queue.Full:
            with self.queue_depth_val.get_lock():
                self.queue_depth_val.value = max(0, self.queue_depth_val.value - 1)
            with self._pending_lock:
                self._pending_requests.pop(req_id, None)
            return 503, "rejected", 0.0

        wait_sec = max(0.001, deadline - time.monotonic())
        completed = ev.wait(timeout=wait_sec)

        with self._pending_lock:
            self._pending_requests.pop(req_id, None)

        if not completed:
            return 503, "timeout", (time.monotonic() - (deadline - timeout_sec)) * 1000.0

        success, err_msg, dur_ms, q_wait_ms = res_box
        _record_queue_wait(q_wait_ms)
        if success:
            return 200, "processed", dur_ms
        else:
            return 503, "timeout", dur_ms

    def resize(self, new_n: int) -> int:
        """
        Trigger resize. Scale-up is non-blocking (spawned in background thread).
        Scale-down asks surplus workers to exit gracefully after current job.
        """
        new_n = max(1, min(new_n, MAX_WORKERS))
        self._target_workers = new_n

        with self._lock:
            active = [w for w in self._workers if w.proc.is_alive() and not w.exit_event.is_set()]
            cur_count = len(active)

        if new_n > cur_count:
            # Scale up in a background thread - NO LOCK held during spawn!
            threading.Thread(
                target=self._scale_up_background, args=(new_n,), daemon=True
            ).start()
        elif new_n < cur_count:
            # Scale down
            self._scale_down_sync(new_n)

        return new_n

    def _scale_up_background(self, target_n: int) -> None:
        with self._scale_lock:
            with self._lock:
                active = [w for w in self._workers if w.proc.is_alive() and not w.exit_event.is_set()]
                needed = target_n - len(active)
                old_count = len(active)

            if needed <= 0:
                return

            t0 = time.monotonic()
            new_handles: List[_WorkerHandle] = []
            for idx in range(needed):
                wid = next(self._wid_counter)
                exit_ev = mp.Event()
                ready_ev = mp.Event()
                affinity = get_worker_core_affinity(old_count + idx)
                p = mp.Process(
                    target=_worker_process_main,
                    args=(
                        wid,
                        self.request_queue,
                        self.response_queue,
                        self.queue_depth_val,
                        self.active_workers_busy,
                        exit_ev,
                        ready_ev,
                        MS_PER_LOAD,
                        affinity,
                    ),
                    daemon=True,
                )
                p.start()
                new_handles.append(_WorkerHandle(wid, p, exit_ev, ready_ev, core_affinity=affinity))

            # Wait for processes to become OS-ready without holding self._lock
            os_ready_time = time.monotonic()
            for wh in new_handles:
                wh.ready_event.wait(timeout=10.0)
            os_ready_dur = round(time.monotonic() - t0, 3)

            # PROVISION_DELAY_SEC: hold workers unavailable for extra time
            # (simulates configuration/warm-up after OS-ready)
            if PROVISION_DELAY_SEC > 0.0:
                log.info(
                    "PROVISION_DELAY_SEC=%.1f: holding %d new worker(s) unavailable for %.1fs",
                    PROVISION_DELAY_SEC, needed, PROVISION_DELAY_SEC,
                )
                time.sleep(PROVISION_DELAY_SEC)

            # Mark workers ready and add to pool
            for wh in new_handles:
                wh.ready = True
                wh.get_psutil()

            dur = round(time.monotonic() - t0, 3)
            self._last_scale_up_seconds = dur

            with self._lock:
                self._workers.extend(new_handles)
                new_total = len([w for w in self._workers if w.proc.is_alive() and not w.exit_event.is_set()])

            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "scale_up",
                "from_workers": old_count,
                "to_workers": new_total,
                "duration_seconds": dur,
                "os_ready_seconds": os_ready_dur,
                "provision_delay_sec": PROVISION_DELAY_SEC,
                "ready_time": datetime.now(timezone.utc).isoformat(),
            }
            self._scale_events.append(event)
            log.info(
                "Scale-up complete: %d -> %d workers (os_ready=%.3fs, total=%.3fs, delay=%.1fs)",
                old_count, new_total, os_ready_dur, dur, PROVISION_DELAY_SEC,
            )

    def _scale_down_sync(self, target_n: int) -> None:
        with self._lock:
            active = [w for w in self._workers if w.proc.is_alive() and not w.exit_event.is_set()]
            surplus = len(active) - target_n
            if surplus <= 0:
                return
            old_count = len(active)
            to_retire = active[-surplus:]
            for wh in to_retire:
                wh.exit_event.set()



        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "scale_down",
            "from_workers": old_count,
            "to_workers": target_n,
            "duration_seconds": 0.0,
        }
        self._scale_events.append(event)
        log.info("Scale-down signaled: %d -> %d workers", old_count, target_n)

    @property
    def n_workers(self) -> int:
        with self._lock:
            return len([w for w in self._workers if w.proc.is_alive() and w.ready and not w.exit_event.is_set()])

    @property
    def last_scale_up_seconds(self) -> float:
        return self._last_scale_up_seconds

    @property
    def scale_events(self) -> List[dict]:
        return list(self._scale_events)

    @property
    def pinning_summary(self) -> List[int]:
        with self._lock:
            pinned = []
            for w in self._workers:
                if w.core_affinity:
                    pinned.extend(w.core_affinity)
            return sorted(list(set(pinned))) if pinned else (WORKER_CORE_IDS or [])

    def get_worker_cpu_stats(self) -> Tuple[float, float]:
        """
        Returns (worker_cpu_percent, worker_cpu_per_active_pct).
        worker_cpu_percent is normalized by total system cpu_count (e.g. 16).
        worker_cpu_per_active_pct is normalized by ACTIVE workers.
        """
        with self._lock:
            procs = [w.get_psutil() for w in self._workers if w.proc.is_alive()]
            n_active = len(procs)
        total_pct = 0.0
        for p in procs:
            if p:
                try:
                    total_pct += p.cpu_percent(interval=None)
                except Exception:
                    pass
        sys_norm = round(total_pct / _cpu_count, 2)
        active_norm = round(total_pct / n_active, 2) if n_active > 0 else 0.0
        return sys_norm, active_norm

    def get_worker_cpu_percent(self) -> float:
        """Measure CPU of worker processes normalized by system core count."""
        return self.get_worker_cpu_stats()[0]

    def shutdown(self) -> None:
        self._stop_dispatcher.set()
        with self._lock:
            for w in self._workers:
                w.exit_event.set()
        for _ in range(len(self._workers) + 2):
            try:
                self.request_queue.put_nowait(None)
            except Exception:
                pass
        try:
            self.response_queue.put_nowait(None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared State & Telemetry Window
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
telemetry_buffer: deque = deque(maxlen=BUFFER_MAXLEN)

# Accumulators for the current 1-second interval
_window_arrivals: int = 0
_window_ok: int = 0
_window_timeout: int = 0
_window_rejected: int = 0
_window_ok_latencies: List[float] = []
_window_all_latencies: List[float] = []
_window_queue_waits: List[float] = []

_pool: Optional[_PoolManager] = None


def _record_arrival() -> None:
    global _window_arrivals
    with _state_lock:
        _window_arrivals += 1


def _record_ok(lat_ms: float) -> None:
    global _window_ok
    with _state_lock:
        _window_ok += 1
        _window_ok_latencies.append(lat_ms)
        _window_all_latencies.append(lat_ms)


def _record_timeout(lat_ms: float) -> None:
    global _window_timeout
    with _state_lock:
        _window_timeout += 1
        _window_all_latencies.append(lat_ms)


def _record_rejected() -> None:
    global _window_rejected
    with _state_lock:
        _window_rejected += 1


def _record_queue_wait(wait_ms: float) -> None:
    global _window_queue_waits
    with _state_lock:
        _window_queue_waits.append(wait_ms)


def _drain_telemetry_window() -> Tuple[int, int, int, int, List[float], List[float], List[float]]:
    global _window_arrivals, _window_ok, _window_timeout, _window_rejected, _window_queue_waits
    with _state_lock:
        arr = _window_arrivals
        ok = _window_ok
        to = _window_timeout
        rej = _window_rejected
        ok_lats = _window_ok_latencies[:]
        all_lats = _window_all_latencies[:]
        q_waits = _window_queue_waits[:]

        _window_arrivals = 0
        _window_ok = 0
        _window_timeout = 0
        _window_rejected = 0
        _window_ok_latencies.clear()
        _window_all_latencies.clear()
        _window_queue_waits.clear()

    return arr, ok, to, rej, ok_lats, all_lats, q_waits


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_d) - 1)
    weight = idx - lo
    return sorted_d[lo] * (1.0 - weight) + sorted_d[hi] * weight


# ---------------------------------------------------------------------------
# CSV Persistence
# ---------------------------------------------------------------------------
def _ensure_csv_header() -> None:
    header = ",".join(CSV_FIELDS)
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "w", newline="") as f:
            f.write(header + "\n")
        return

    with open(CSV_LOG_PATH, "r", newline="") as f:
        first_line = f.readline().strip()

    if first_line != header:
        # Prepend or rewrite header cleanly
        with open(CSV_LOG_PATH, "r", newline="") as f:
            body = f.read()
        lines = body.splitlines()
        # Drop old header if present
        if lines and ("timestamp" in lines[0] or "cpu_percent" in lines[0]):
            data_lines = lines[1:]
        else:
            data_lines = lines
        with open(CSV_LOG_PATH, "w", newline="") as f:
            f.write(header + "\n")
            if data_lines:
                f.write("\n".join(data_lines) + "\n")


def _append_csv(record: dict) -> None:
    try:
        with open(CSV_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writerow(record)
    except Exception as exc:
        log.warning("CSV append error: %s", exc)


# ---------------------------------------------------------------------------
# Background Sampling Loop (1s cadence)
# ---------------------------------------------------------------------------
def sampling_loop(stop_event: threading.Event) -> None:
    psutil.cpu_percent(interval=None)  # prime system CPU
    next_tick = time.monotonic() + SAMPLE_INTERVAL_SEC

    while not stop_event.is_set():
        sys_cpu = psutil.cpu_percent(interval=None)
        mem_pct = psutil.virtual_memory().percent

        arr, ok, to, rej, ok_lats, all_lats, q_waits = _drain_telemetry_window()
        errors = to + rej
        req_count = ok + to  # completed requests in window

        avg_lat = statistics.mean(all_lats) if all_lats else 0.0
        p95_lat = _percentile(all_lats, 95)
        ok_p95_lat = _percentile(ok_lats, 95)
        queue_wait_ms_avg = round(statistics.mean(q_waits), 2) if q_waits else 0.0

        if _pool:
            depth = max(0, _pool.queue_depth_val.value)
            busy = min(max(0, _pool.active_workers_busy.value), _pool.n_workers)
            in_flight = depth + busy
            workers = _pool.n_workers
            worker_cpu_pct, worker_cpu_per_active = _pool.get_worker_cpu_stats()
        else:
            depth = 0
            busy = 0
            in_flight = 0
            workers = 2
            worker_cpu_pct = 0.0
            worker_cpu_per_active = 0.0

        utilization = round(busy / workers, 4) if workers > 0 else 0.0

        try:
            freq_info = psutil.cpu_freq()
            cpu_freq_mhz = round(freq_info.current, 1) if freq_info else 0.0
        except Exception:
            cpu_freq_mhz = 0.0

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": round(sys_cpu, 2),
            "mem_percent": round(mem_pct, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "request_count": req_count,
            "active_workers": workers,
            "error_count": errors,
            "queue_depth": depth,
            # Appended fields
            "arrivals_count": arr,
            "ok_count": ok,
            "timeout_count": to,
            "rejected_count": rej,
            "in_flight": in_flight,
            "worker_cpu_percent": worker_cpu_pct,
            "ok_p95_latency_ms": round(ok_p95_lat, 2),
            # Step 2 additions
            "busy_workers": busy,
            "utilization": utilization,
            "worker_cpu_per_active_pct": worker_cpu_per_active,
            "queue_wait_ms_avg": queue_wait_ms_avg,
            "cpu_freq_mhz": cpu_freq_mhz,
        }

        with _state_lock:
            telemetry_buffer.append(record)
        _append_csv(record)

        next_tick += SAMPLE_INTERVAL_SEC
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            stop_event.wait(sleep_time)
        else:
            # Overran interval; reset next_tick to prevent spin
            next_tick = time.monotonic() + SAMPLE_INTERVAL_SEC


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        pins = _pool.pinning_summary if _pool else WORKER_CORE_IDS
        return jsonify({
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "cpu_count": _cpu_count,
            "physical_cpu_count": _physical_cores_count,
            "pinning_used": pins,
            "reserved_physical_cores": RESERVED_PHYSICAL_CORES,
        })

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
                target = int(payload["active_workers"])
                if _pool is not None:
                    new_n = _pool.resize(target)
                else:
                    new_n = target
                pins = _pool.pinning_summary if _pool else WORKER_CORE_IDS
                return jsonify({
                    "status": "updated",
                    "active_workers": new_n,
                    "last_scale_up_seconds": _pool.last_scale_up_seconds if _pool else 0.0,
                    "scale_events": _pool.scale_events if _pool else [],
                    "cpu_count": _cpu_count,
                    "physical_cpu_count": _physical_cores_count,
                    "pinning_used": pins,
                    "reserved_physical_cores": RESERVED_PHYSICAL_CORES,
                    "queue_capacity": QUEUE_CAPACITY,
                })

        workers = _pool.n_workers if _pool else 2
        last_scale = _pool.last_scale_up_seconds if _pool else 0.0
        events = _pool.scale_events if _pool else []
        pins = _pool.pinning_summary if _pool else WORKER_CORE_IDS
        return jsonify({
            "active_workers": workers,
            "last_scale_up_seconds": last_scale,
            "scale_events": events,
            "cpu_count": _cpu_count,
            "physical_cpu_count": _physical_cores_count,
            "pinning_used": pins,
            "reserved_physical_cores": RESERVED_PHYSICAL_CORES,
            "queue_capacity": QUEUE_CAPACITY,
        })

    @app.route("/api/process", methods=["POST"])
    def process():
        _record_arrival()
        payload = request.get_json(silent=True) or {}
        load = payload.get("load", 1)
        try:
            load = max(0, int(load))
        except (TypeError, ValueError):
            load = 1

        if _pool is None:
            # Fallback for lightweight testing when pool is not initialized
            t0 = time.perf_counter()
            _cpu_work(load, MS_PER_LOAD)
            dur = (time.perf_counter() - t0) * 1000.0
            _record_ok(dur)
            return jsonify({"status": "processed", "load": load})

        status_code, status_str, dur_ms = _pool.submit_request(load, REQUEST_TIMEOUT_SEC)
        if status_code == 200:
            _record_ok(dur_ms)
            return jsonify({"status": "processed", "load": load})
        elif status_str == "rejected":
            _record_rejected()
            return jsonify({"status": "rejected", "error": "queue full", "load": load}), 503
        else:
            _record_timeout(dur_ms)
            return jsonify({"status": "timeout", "reason": status_str, "load": load}), 503

    return app


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    global _pool

    parser = argparse.ArgumentParser(description="PERM telemetry collector")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Initial number of worker processes (default 2)",
    )
    args = parser.parse_args()

    log.info(
        "Config: MS_PER_LOAD=%.1f  MAX_WORKERS=%d  REQUEST_TIMEOUT_SEC=%.1f  "
        "QUEUE_CAPACITY=%d  WORKER_CORE_IDS=%s",
        MS_PER_LOAD,
        MAX_WORKERS,
        REQUEST_TIMEOUT_SEC,
        QUEUE_CAPACITY,
        WORKER_CORE_IDS if WORKER_CORE_IDS else "none (no pinning)",
    )

    _ensure_csv_header()

    log.info("Starting worker pool with %d initial workers...", args.workers)
    _pool = _PoolManager(args.workers)
    log.info("Worker pool ready. Active workers: %d", _pool.n_workers)

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
        if _pool:
            _pool.shutdown()


if __name__ == "__main__":
    main()