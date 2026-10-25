"""
PERM - calibrate_testbed.py
============================
Finds the saturation point (req/s where p95 latency exceeds 300 ms OR
error rate exceeds 1 %) for worker counts 1, 2, 4, 8 by ramping request
rate in-process (no Locust needed).

Usage:
    python calibrate_testbed.py [--port 5000] [--load 7]

The telemetry collector must already be running on --port (default 5000).
The script calls /api/capacity to resize the pool, then sends concurrent
HTTP POST /api/process at increasing rates and measures real latency.

Results are written to  testbed_calibration.json  and printed as a table.

Tuning goal documented here
---------------------------
MS_PER_LOAD = 18 ms/unit  (default in telemetry_collector.py)

With load=8 (mid-burst):
  Each request does 8 * 18 = 144 ms of CPU work.
  Saturation for N workers ~ N * (1000 / 144) ~= N * 6.9 req/s.

  N=1  ->  ~7   req/s saturation
  N=2  ->  ~14  req/s saturation
  N=4  ->  ~28  req/s saturation
  N=8  ->  ~55  req/s saturation

Locust burst stage: 90 users x 1/(0.2-1.0)s wait ~= 90-450 req/s *attempted*,
but with load=7..10 and 2 workers -> heavy queuing/503s expected,
and with 6-8 workers -> latency drops below 300ms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import requests

CALIBRATION_OUTPUT = os.path.join(os.path.dirname(__file__), "testbed_calibration.json")

# Ramp settings
RAMP_START_RPS = 1
RAMP_MAX_RPS = 120         # upper bound - stopped when saturation found
RAMP_STEP_RPS = 2          # increase per step
RAMP_STEP_DURATION_SEC = 5 # seconds per ramp step
MIN_REQUESTS_PER_STEP = 10  # skip step if too few samples

SATURATION_P95_MS = 300.0   # p95 latency threshold
SATURATION_ERROR_PCT = 1.0  # error rate threshold (%)

DEFAULT_LOAD = 7            # load value sent to /api/process
DEFAULT_WORKER_COUNTS = [1, 2, 4, 8]


def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _set_workers(base_url: str, n: int) -> bool:
    try:
        r = requests.post(f"{base_url}/api/capacity", json={"active_workers": n}, timeout=5.0)
        return r.status_code == 200
    except Exception as exc:
        print(f"  [!] Failed to set workers to {n}: {exc}")
        return False


def _wait_for_pool(base_url: str, expected_n: int, timeout: float = 15.0) -> bool:
    """Poll /api/capacity until active_workers matches expected_n or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/api/capacity", timeout=2.0)
            if r.status_code == 200 and r.json().get("active_workers") == expected_n:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _send_one(base_url: str, load: int) -> Tuple[float, bool]:
    """Send one /api/process request. Returns (latency_ms, is_error)."""
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base_url}/api/process", json={"load": load}, timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, r.status_code != 200
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, True


def _measure_at_rate(
    base_url: str,
    rps: float,
    duration_sec: float,
    load: int,
) -> Dict:
    """
    Send `rps` requests/second for `duration_sec` seconds using a thread pool.
    Returns dict with p95_ms, avg_ms, error_pct, actual_rps, n_requests.
    """
    interval = 1.0 / rps if rps > 0 else 1.0
    n_target = max(MIN_REQUESTS_PER_STEP, int(rps * duration_sec))
    parallelism = max(1, min(int(rps) + 4, 64))  # enough threads to sustain rate

    latencies: List[float] = []
    errors = 0
    t_start = time.monotonic()

    def _worker(i: int):
        # Stagger requests evenly across the duration window
        target_t = t_start + i * interval
        now = time.monotonic()
        if target_t > now:
            time.sleep(target_t - now)
        return _send_one(base_url, load)

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futs = [pool.submit(_worker, i) for i in range(n_target)]
        for fut in as_completed(futs):
            lat_ms, is_err = fut.result()
            latencies.append(lat_ms)
            if is_err:
                errors += 1

    elapsed = time.monotonic() - t_start
    actual_rps = len(latencies) / elapsed if elapsed > 0 else 0.0
    latencies.sort()
    p95 = _percentile(latencies, 95)
    avg = statistics.mean(latencies) if latencies else 0.0
    err_pct = 100.0 * errors / len(latencies) if latencies else 0.0

    return {
        "target_rps": rps,
        "actual_rps": round(actual_rps, 2),
        "n_requests": len(latencies),
        "p95_latency_ms": round(p95, 2),
        "avg_latency_ms": round(avg, 2),
        "error_count": errors,
        "error_pct": round(err_pct, 2),
    }


def calibrate_workers(
    base_url: str,
    n_workers: int,
    load: int,
) -> Dict:
    """Ramp rate until saturation for a fixed number of workers."""
    print(f"\n  [workers={n_workers}] Setting pool size...")
    _set_workers(base_url, n_workers)
    time.sleep(2.0)  # let pool settle
    _wait_for_pool(base_url, n_workers)

    saturation_rps: Optional[float] = None
    ramp_table: List[Dict] = []

    rps = RAMP_START_RPS
    while rps <= RAMP_MAX_RPS:
        print(f"    rate={rps:.0f} rps ... ", end="", flush=True)
        result = _measure_at_rate(base_url, rps, RAMP_STEP_DURATION_SEC, load)
        ramp_table.append(result)
        print(
            f"p95={result['p95_latency_ms']:.1f}ms  "
            f"avg={result['avg_latency_ms']:.1f}ms  "
            f"err={result['error_pct']:.1f}%"
        )

        saturated = (
            result["p95_latency_ms"] > SATURATION_P95_MS
            or result["error_pct"] > SATURATION_ERROR_PCT
        )
        if saturated:
            saturation_rps = result["actual_rps"]
            # Also record the last "good" rps
            if len(ramp_table) >= 2:
                prev = ramp_table[-2]
                saturation_rps_good = prev["actual_rps"]
            else:
                saturation_rps_good = 0.0
            print(
                f"  [workers={n_workers}] SATURATED at ~{saturation_rps:.1f} req/s "
                f"(last good: {saturation_rps_good:.1f} req/s)"
            )
            break

        rps += RAMP_STEP_RPS

    if saturation_rps is None:
        saturation_rps = float(rps)
        print(f"  [workers={n_workers}] Did not saturate up to {RAMP_MAX_RPS} req/s")

    return {
        "n_workers": n_workers,
        "load": load,
        "ms_per_load": float(os.environ.get("MS_PER_LOAD", "18")),
        "saturation_rps": round(saturation_rps, 2),
        "saturation_p95_threshold_ms": SATURATION_P95_MS,
        "saturation_error_threshold_pct": SATURATION_ERROR_PCT,
        "ramp_table": ramp_table,
    }


def main():
    parser = argparse.ArgumentParser(description="PERM testbed calibration tool")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--load", type=int, default=DEFAULT_LOAD,
                        help="load value to send per request (default 7)")
    parser.add_argument("--workers", type=str,
                        default=",".join(str(w) for w in DEFAULT_WORKER_COUNTS),
                        help="comma-separated worker counts to test (default 1,2,4,8)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    worker_counts = [int(w) for w in args.workers.split(",") if w.strip()]

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=3.0)
        assert r.status_code == 200
        print(f"[[OK]] Collector healthy at {base_url}")
    except Exception as exc:
        print(f"[[FAIL]] Cannot reach collector at {base_url}: {exc}")
        sys.exit(1)

    print(f"\nCalibration parameters:")
    print(f"  load value        = {args.load}")
    print(f"  MS_PER_LOAD       = {os.environ.get('MS_PER_LOAD', '18')} ms")
    print(f"  saturation p95 >=  {SATURATION_P95_MS} ms  OR  error% >= {SATURATION_ERROR_PCT}%")
    print(f"  worker counts     = {worker_counts}")
    print(f"  ramp step         = {RAMP_STEP_RPS} rps / {RAMP_STEP_DURATION_SEC}s")

    results = []
    for n in worker_counts:
        r = calibrate_workers(base_url, n, args.load)
        results.append(r)

    output = {
        "calibration_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "load": args.load,
        "ms_per_load": float(os.environ.get("MS_PER_LOAD", "18")),
        "saturation_p95_threshold_ms": SATURATION_P95_MS,
        "saturation_error_threshold_pct": SATURATION_ERROR_PCT,
        "results": results,
    }

    with open(CALIBRATION_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[[OK]] Results saved to {CALIBRATION_OUTPUT}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Workers':>8}  {'Sat. req/s':>12}  {'MS_PER_LOAD':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['n_workers']:>8}  {r['saturation_rps']:>12.1f}  "
            f"{r['ms_per_load']:>12.1f}"
        )
    print("=" * 60)
    print("\nInterpretation:")
    print("  2 workers should saturate during Locust burst (90 users, load 7-10)")
    print("  6-8 workers should handle that burst without p95 > 300ms")


if __name__ == "__main__":
    main()
