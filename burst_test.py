"""
PERM - burst_test.py
====================
Open-loop traffic generator for PERM testbed evaluation.

Sends HTTP POST /api/process requests at target arrival rates following a Poisson
process (exponential inter-arrival intervals) regardless of response latency,
using a large concurrent worker pool.

Profiles:
  1. step:
     - 30s calm @ 4 req/s
     - 60s burst step @ 40 req/s (burst start recorded)
     - 30s calm @ 4 req/s
     Total duration = 120s
  2. ramp:
     - 30s calm @ 4 req/s
     - 30s linear ramp (4 -> 40 req/s) (burst start recorded)
     - 45s hold @ 40 req/s
     - 30s calm @ 4 req/s
     Total duration = 135s

Usage:
    python burst_test.py [--profile {step,ramp}] [--seed 42] [--target-url http://127.0.0.1:5000/api/process] [--load 7]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("burst_test")

DEFAULT_TARGET_URL = "http://127.0.0.1:5000/api/process"
DEFAULT_LOAD = 7
MAX_WORKERS_THREADPOOL = 250


def generate_schedule(profile: str, seed: int) -> Tuple[List[float], float, float]:
    """
    Generate schedule of request dispatch offsets (in seconds from t0) using a Poisson process.
    Returns:
        (dispatch_offsets, burst_start_offset, total_duration)
    """
    rng = random.Random(seed)
    offsets: List[float] = []

    if profile == "step":
        stages = [
            # (duration_sec, start_rate, end_rate, is_burst_start)
            (30.0, 4.0, 4.0, False),
            (60.0, 40.0, 40.0, True),
            (30.0, 4.0, 4.0, False),
        ]
    elif profile == "ramp":
        stages = [
            (30.0, 4.0, 4.0, False),
            (30.0, 4.0, 40.0, True),
            (45.0, 40.0, 40.0, False),
            (30.0, 4.0, 4.0, False),
        ]
    elif profile == "spike":
        # 4 -> 40 req/s for 15 s -> 4
        stages = [
            (30.0, 4.0, 4.0, False),
            (15.0, 40.0, 40.0, True),
            (30.0, 4.0, 4.0, False),
        ]
    elif profile == "double_burst":
        # Two 30s bursts separated by 30s calm
        stages = [
            (20.0, 4.0, 4.0, False),
            (30.0, 40.0, 40.0, True),
            (30.0, 4.0, 4.0, False),
            (30.0, 40.0, 40.0, False),
            (20.0, 4.0, 4.0, False),
        ]
    elif profile == "sawtooth":
        # Up and down ramps
        stages = [
            (20.0, 4.0, 4.0, False),
            (20.0, 4.0, 40.0, True),
            (20.0, 40.0, 4.0, False),
            (20.0, 4.0, 40.0, False),
            (20.0, 40.0, 4.0, False),
            (20.0, 4.0, 4.0, False),
        ]
    elif profile == "random_walk":
        # Seeded rate changes every 10-20 s between 3 and 45 req/s
        stages = [(20.0, 4.0, 4.0, False)]
        rem_dur = 100.0  # total 120s
        first_burst_flag = True
        cur_rate = 4.0
        while rem_dur > 0:
            dur = min(rem_dur, round(rng.uniform(10.0, 20.0), 1))
            next_rate = round(rng.uniform(3.0, 45.0), 1)
            is_burst = False
            if first_burst_flag and (next_rate >= 15.0 or rem_dur < 80.0):
                is_burst = True
                first_burst_flag = False
            stages.append((dur, cur_rate, next_rate, is_burst))
            cur_rate = next_rate
            rem_dur -= dur
    else:
        raise ValueError(f"Unknown profile: {profile}")

    current_t = 0.0
    stage_start_t = 0.0
    burst_start_offset = 30.0
    total_duration = sum(d for d, _, _, _ in stages)

    for dur, r_start, r_end, is_burst in stages:
        stage_end = stage_start_t + dur
        if is_burst:
            burst_start_offset = stage_start_t

        current_t = max(current_t, stage_start_t)
        while current_t < stage_end:
            # Linear interpolation of rate within stage
            if dur > 0:
                fraction = min(1.0, max(0.0, (current_t - stage_start_t) / dur))
            else:
                fraction = 1.0
            current_rate = r_start + fraction * (r_end - r_start)
            current_rate = max(0.1, current_rate)

            # Poisson arrival interval ~ Exponential(lambda)
            dt = rng.expovariate(current_rate)
            current_t += dt
            if current_t < stage_end:
                offsets.append(current_t)

        stage_start_t = stage_end

    offsets.sort()
    return offsets, burst_start_offset, total_duration


def run_burst(
    target_url: str = DEFAULT_TARGET_URL,
    profile: str = "step",
    seed: int = 42,
    load: int = DEFAULT_LOAD,
    timeout_sec: float = 4.0,
    max_workers: int = MAX_WORKERS_THREADPOOL,
    on_burst_start: Optional[Callable[[datetime, float], None]] = None,
) -> Dict:
    """
    Execute open-loop traffic burst.
    Returns summary dictionary with timing, request counts, and status codes.
    """
    offsets, burst_start_offset, total_dur = generate_schedule(profile, seed)
    log.info(
        "Starting burst run: profile=%s seed=%d total_reqs=%d duration=%.1fs burst_offset=%.1fs",
        profile,
        seed,
        len(offsets),
        total_dur,
        burst_start_offset,
    )

    t0_mono = time.monotonic()
    t0_utc = datetime.now(timezone.utc)

    burst_start_mono = t0_mono + burst_start_offset
    burst_start_utc = datetime.fromtimestamp(
        t0_utc.timestamp() + burst_start_offset, tz=timezone.utc
    )

    burst_signaled = False
    lock = threading.Lock()
    results: List[Tuple[int, float]] = []  # (status_code, latency_ms)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def _dispatch_one():
        t_req = time.perf_counter()
        try:
            r = session.post(
                target_url,
                json={"load": load},
                timeout=timeout_sec,
            )
            lat = (time.perf_counter() - t_req) * 1000.0
            with lock:
                results.append((r.status_code, lat))
        except Exception:
            lat = (time.perf_counter() - t_req) * 1000.0
            with lock:
                results.append((-1, lat))

    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    for offset in offsets:
        target_time = t0_mono + offset
        now = time.monotonic()

        # Signal burst start hook once reached
        if not burst_signaled and now >= burst_start_mono:
            burst_signaled = True
            log.info("BURST STARTED at %s (offset %.2fs)", burst_start_utc.isoformat(), burst_start_offset)
            if on_burst_start:
                try:
                    on_burst_start(burst_start_utc, burst_start_mono)
                except Exception as exc:
                    log.warning("on_burst_start callback failed: %s", exc)

        sleep_wait = target_time - now
        if sleep_wait > 0.0005:
            time.sleep(sleep_wait)

        # Re-check burst trigger after sleep
        if not burst_signaled and time.monotonic() >= burst_start_mono:
            burst_signaled = True
            log.info("BURST STARTED at %s (offset %.2fs)", burst_start_utc.isoformat(), burst_start_offset)
            if on_burst_start:
                try:
                    on_burst_start(burst_start_utc, burst_start_mono)
                except Exception as exc:
                    log.warning("on_burst_start callback failed: %s", exc)

        futures.append(pool.submit(_dispatch_one))

    # Wait for remaining scheduled requests
    run_end_mono = t0_mono + total_dur
    remain = run_end_mono - time.monotonic()
    if remain > 0:
        time.sleep(remain)

    pool.shutdown(wait=True)
    session.close()

    elapsed = time.monotonic() - t0_mono
    status_codes = [c for c, _ in results]
    latencies = [l for _, l in results]

    n_ok = status_codes.count(200)
    n_rej = status_codes.count(503)
    n_err = status_codes.count(-1)

    return {
        "profile": profile,
        "seed": seed,
        "target_url": target_url,
        "start_time_utc": t0_utc.isoformat(),
        "burst_start_utc": burst_start_utc.isoformat(),
        "burst_start_mono": burst_start_mono,
        "duration_sec": round(elapsed, 2),
        "total_requests": len(results),
        "ok_count": n_ok,
        "rejected_count": n_rej,
        "error_count": n_err,
        "goodput_req_s": round(n_ok / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="PERM open-loop Poisson traffic generator")
    parser.add_argument(
        "--profile",
        choices=["step", "ramp", "spike", "double_burst", "sawtooth", "random_walk"],
        default="step",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-url", type=str, default=DEFAULT_TARGET_URL)
    parser.add_argument("--load", type=int, default=DEFAULT_LOAD)
    args = parser.parse_args()

    summary = run_burst(
        target_url=args.target_url,
        profile=args.profile,
        seed=args.seed,
        load=args.load,
    )
    print("\n" + "=" * 60)
    print("BURST RUN COMPLETED")
    print(json.dumps(summary, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
