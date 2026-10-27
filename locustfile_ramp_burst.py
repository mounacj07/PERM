"""
PERM - locustfile_ramp_burst.py
===============================
Thin wrapper locustfile for the 'ramp_burst' precursor workload profile.
Created because burst_test.py and locustfile_perm.py are strictly immutable boundaries.
Loads the Poisson schedule from ramp_burst.py and dispatches requests via PermRampBurstUser.
"""

from __future__ import annotations

import logging
import os
import time

import gevent
from locust import HttpUser, task
from locust.clients import LocustHttpAdapter

import ramp_burst
from locustfile_perm import _DISPATCHER_SEMAPHORE, _log_phase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("locustfile_ramp_burst")

# Pin process to logical CPU cores [0-3]
try:
    import psutil
    p = psutil.Process()
    p.cpu_affinity([0, 1, 2, 3])
except Exception as exc:
    pass


class PermRampBurstUser(HttpUser):
    """
    Open-loop Locust user that dispatches Poisson-scheduled requests for ramp_burst.
    """

    def on_start(self):
        adapter = LocustHttpAdapter(pool_manager=None, pool_connections=250, pool_maxsize=250, max_retries=0)
        self.client.mount("http://", adapter)
        self.client.mount("https://", adapter)

    @task
    def dispatch_traffic(self):
        if not _DISPATCHER_SEMAPHORE.acquire(blocking=False):
            gevent.sleep(2.0)
            return

        try:
            profile = "ramp_burst"
            seed = int(os.environ.get("PERM_SEED", "42"))
            load = int(os.environ.get("PERM_LOAD", "7"))
            timeout = float(os.environ.get("PERM_TIMEOUT", "3.0"))

            offsets, burst_start_offset, total_dur = ramp_burst.generate_schedule(profile, seed)
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

                if not burst_signaled and now >= burst_start_mono:
                    burst_signaled = True
                    log.info("LOCUST PRECURSOR RAMP STARTED at offset %.2fs", now - t0_mono)
                    _log_phase("ramp", 1, 1, now - t0_mono)

                sleep_wait = target_mono - now
                if sleep_wait > 0.0005:
                    gevent.sleep(sleep_wait)

                if not burst_signaled and time.monotonic() >= burst_start_mono:
                    burst_signaled = True
                    log.info("LOCUST PRECURSOR RAMP STARTED at offset %.2fs", time.monotonic() - t0_mono)
                    _log_phase("ramp", 1, 1, time.monotonic() - t0_mono)

                gevent.spawn(_fire_request)

            run_end_mono = t0_mono + total_dur
            remain = run_end_mono - time.monotonic()
            if remain > 0:
                gevent.sleep(remain)

            elapsed_total = time.monotonic() - t0_mono
            _log_phase("profile_end", 1, 1, elapsed_total)
            log.info("Profile %s (seed %d) completed in %.2fs. Waiting for in-flight requests...", profile, seed, elapsed_total)
            gevent.sleep(2.0)

            if self.environment.runner:
                if getattr(self.environment.parsed_options, "headless", False):
                    self.environment.runner.quit()
                else:
                    self.environment.runner.stop()

        finally:
            _DISPATCHER_SEMAPHORE.release()
