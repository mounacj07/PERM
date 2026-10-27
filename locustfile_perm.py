"""
PERM - locustfile_perm.py
=========================
Open-loop canonical traffic generator for PERM testbed evaluation and live demos.
Works in headless mode and with the Locust Web UI.

Open-Loop Architecture:
  - Exactly ONE Locust user acts as the open-loop dispatcher.
  - Request dispatch times are precomputed from the seeded Poisson rate curve
    imported from burst_test.py (ensuring identical target curves).
  - Each request is spawned in an independent greenlet (gevent.spawn) at its
    scheduled arrival time, completely unblocked by earlier response latencies.
  - Requests are dispatched via the Locust user's HTTP client (self.client),
    so Locust records request counts, latencies, percentiles, and errors.
  - Automatically pins the Locust process to logical CPU cores [0, 1, 2, 3].
  - The run automatically stops itself when the profile duration completes.

Supported Profiles (via env PERM_PROFILE):
  step, ramp, spike, double_burst, sawtooth, random_walk, random_mix

Usage (Headless):
  locust -f locustfile_perm.py --headless -u 1 -r 1 -t 125s --host http://127.0.0.1:5000 --csv data/runs_locust/step_seed42

Usage (Web UI):
  locust -f locustfile_perm.py --host http://127.0.0.1:5000
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import gevent
import gevent.lock
from locust import HttpUser, events, task
from locust.clients import LocustHttpAdapter

# Import canonical profile definitions from burst_test
import burst_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("locustfile_perm")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE_LOG_PATH = os.path.join(PROJECT_ROOT, "phase_log.csv")
PHASE_LOG_FIELDS = ["timestamp", "phase", "cycle", "target_users", "elapsed_sec"]

# Pin process to logical CPU cores [0-3]
try:
    import psutil
    p = psutil.Process()
    p.cpu_affinity([0, 1, 2, 3])
    log.info("Pinned Locust process (PID %d) to logical cores [0, 1, 2, 3]", p.pid)
except Exception as exc:
    log.warning("Could not set CPU affinity for Locust process: %s", exc)


def _ensure_phase_log_header():
    if not os.path.exists(PHASE_LOG_PATH):
        with open(PHASE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PHASE_LOG_FIELDS)
            writer.writeheader()


def _log_phase(phase_name: str, cycle: int, target_users: int, elapsed_sec: float):
    _ensure_phase_log_header()
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    try:
        with open(PHASE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PHASE_LOG_FIELDS)
            writer.writerow({
                "timestamp": ts_str,
                "phase": phase_name,
                "cycle": cycle,
                "target_users": target_users,
                "elapsed_sec": round(elapsed_sec, 2),
            })
    except Exception as exc:
        log.warning("Failed writing to phase_log.csv: %s", exc)


_DISPATCHER_SEMAPHORE = gevent.lock.Semaphore(1)


class PermOpenLoopUser(HttpUser):
    """
    Open-loop Locust user that dispatches Poisson-scheduled requests.
    If multiple users are spawned in the Web UI, only one becomes the active dispatcher.
    """

    def on_start(self):
        # Configure large connection pool to prevent socket exhaustion during bursts
        adapter = LocustHttpAdapter(pool_manager=None, pool_connections=250, pool_maxsize=250, max_retries=0)
        self.client.mount("http://", adapter)
        self.client.mount("https://", adapter)

    @task
    def dispatch_traffic(self):
        # Ensure only a single dispatcher runs across greenlets/users
        if not _DISPATCHER_SEMAPHORE.acquire(blocking=False):
            gevent.sleep(2.0)
            return

        try:
            profile = os.environ.get("PERM_PROFILE", "step")
            seed = int(os.environ.get("PERM_SEED", "42"))
            load = int(os.environ.get("PERM_LOAD", "7"))
            timeout = float(os.environ.get("PERM_TIMEOUT", "3.0"))

            offsets, burst_start_offset, total_dur = burst_test.generate_schedule(profile, seed)
            log.info(
                "Starting Locust open-loop run: profile=%s, seed=%d, total_requests=%d, duration=%.1fs, burst_offset=%.1fs",
                profile,
                seed,
                len(offsets),
                total_dur,
                burst_start_offset,
            )

            t0_mono = time.monotonic()
            burst_start_mono = t0_mono + burst_start_offset
            burst_signaled = False

            _log_phase("profile_start", 1, 1, 0.0)

            def _fire_request():
                try:
                    self.client.post("/api/process", json={"load": load}, timeout=timeout, name="POST /api/process")
                except Exception:
                    pass

            for offset in offsets:
                target_mono = t0_mono + offset
                now = time.monotonic()

                # Check burst start signal
                if not burst_signaled and now >= burst_start_mono:
                    burst_signaled = True
                    phase_name = "ramp" if profile == "ramp" else "burst"
                    log.info("LOCUST BURST STARTED at offset %.2fs", now - t0_mono)
                    _log_phase(phase_name, 1, 1, now - t0_mono)

                sleep_wait = target_mono - now
                if sleep_wait > 0.0005:
                    gevent.sleep(sleep_wait)

                # Re-check burst signal after sleep
                if not burst_signaled and time.monotonic() >= burst_start_mono:
                    burst_signaled = True
                    phase_name = "ramp" if profile == "ramp" else "burst"
                    log.info("LOCUST BURST STARTED at offset %.2fs", time.monotonic() - t0_mono)
                    _log_phase(phase_name, 1, 1, time.monotonic() - t0_mono)

                gevent.spawn(_fire_request)

            # Wait for remaining duration of profile
            run_end_mono = t0_mono + total_dur
            remain = run_end_mono - time.monotonic()
            if remain > 0:
                gevent.sleep(remain)

            elapsed_total = time.monotonic() - t0_mono
            _log_phase("profile_end", 1, 1, elapsed_total)
            log.info("Profile %s (seed %d) completed in %.2fs. Waiting for in-flight requests...", profile, seed, elapsed_total)

            # Grace period for in-flight requests to complete
            gevent.sleep(2.0)

            log.info("Locust run finished. Stopping runner.")
            if self.environment.runner:
                if getattr(self.environment.parsed_options, "headless", False):
                    self.environment.runner.quit()
                else:
                    self.environment.runner.stop()

        finally:
            _DISPATCHER_SEMAPHORE.release()
