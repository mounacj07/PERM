"""
PERM – acceptance_run.py
=========================
Runs the 60-second fast-demo acceptance test:
  - PASS 1: capacity fixed at 2 workers  (PERM_AUTO_SCALE=0)
  - PASS 2: PERM auto-scaling on          (PERM_AUTO_SCALE=1)

Saves per-second telemetry rows to step1_scaling_proof.csv.

Requirements:
  • All PERM services running: python run_all.py
  • This script is separate from Locust; it drives load internally using
    threads and reads /metrics/history from the collector.

Usage:
    python acceptance_run.py [--duration 60] [--workers-fixed 2]
                             [--collector http://127.0.0.1:5000]
                             [--remediation http://127.0.0.1:5003]
                             [--concurrency 30] [--load 8]

The script:
  1. Sets capacity to --workers-fixed and disables autoscaling.
  2. Fires continuous load for --duration seconds.
  3. Samples /metrics/history every second.
  4. Restores auto-scaling and repeats.
  5. Writes all rows (with a 'pass' column: 'fixed'/'auto') to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "step1_scaling_proof.csv")

CSV_FIELDS = [
    "pass",
    "elapsed_sec",
    "timestamp",
    "active_workers",
    "cpu_percent",
    "mem_percent",
    "avg_latency_ms",
    "p95_latency_ms",
    "request_count",
    "error_count",
    "queue_depth",
]


def _wait_healthy(url: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _set_capacity(collector_url: str, n: int):
    try:
        requests.post(f"{collector_url}/api/capacity",
                      json={"active_workers": n}, timeout=3.0)
    except Exception as exc:
        print(f"  [!] Could not set capacity: {exc}")


def _set_remediation_mode(remediation_url: str, mode: str):
    """mode: 'AUTO' or 'MANUAL' (MANUAL = no autoscale)."""
    try:
        requests.post(f"{remediation_url}/remediation/mode",
                      json={"mode": mode}, timeout=3.0)
    except Exception:
        pass


def _fire_requests(collector_url: str, stop_event: threading.Event,
                   load: int, concurrency: int):
    """Fire continuous requests until stop_event is set."""
    def _one():
        while not stop_event.is_set():
            try:
                requests.post(f"{collector_url}/api/process",
                              json={"load": load}, timeout=4.0)
            except Exception:
                pass

    threads = [threading.Thread(target=_one, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    return threads


def _collect_telemetry(collector_url: str, duration_sec: int,
                       pass_name: str) -> List[dict]:
    """Poll /metrics/history every second for duration_sec seconds."""
    rows = []
    t0 = time.monotonic()
    last_ts = None

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= duration_sec:
            break

        try:
            r = requests.get(f"{collector_url}/metrics/history?seconds=2",
                             timeout=2.0)
            if r.status_code == 200:
                records = r.json().get("records", [])
                for rec in records:
                    ts = rec.get("timestamp", "")
                    if ts and ts != last_ts:
                        last_ts = ts
                        row = {"pass": pass_name,
                               "elapsed_sec": round(elapsed, 1)}
                        for field in CSV_FIELDS[2:]:
                            row[field] = rec.get(field, "")
                        rows.append(row)
        except Exception:
            pass

        time.sleep(1.0)

    return rows


def run_pass(
    collector_url: str,
    remediation_url: str,
    pass_name: str,
    fixed_workers: int,
    auto_scale: bool,
    duration_sec: int,
    load: int,
    concurrency: int,
) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"PASS: {pass_name}")
    print(f"  workers={'AUTO' if auto_scale else fixed_workers}  "
          f"duration={duration_sec}s  load={load}  concurrency={concurrency}")
    print("=" * 60)

    # Configure mode
    if auto_scale:
        _set_remediation_mode(remediation_url, "AUTO")
        print("  [*] Autoscale: ON (PERM will manage workers)")
    else:
        _set_remediation_mode(remediation_url, "MANUAL")  # freeze autoscale
        _set_capacity(collector_url, fixed_workers)
        print(f"  [*] Autoscale: OFF, fixed at {fixed_workers} workers")

    time.sleep(2.0)  # settle

    stop_event = threading.Event()
    threads = _fire_requests(collector_url, stop_event, load, concurrency)

    rows = _collect_telemetry(collector_url, duration_sec, pass_name)

    stop_event.set()
    for t in threads:
        t.join(timeout=5.0)

    print(f"  [✓] Collected {len(rows)} telemetry rows")
    if rows:
        p95_vals = [r.get("p95_latency_ms", 0) for r in rows
                    if isinstance(r.get("p95_latency_ms"), (int, float))]
        err_vals = [r.get("error_count", 0) for r in rows
                    if isinstance(r.get("error_count"), (int, float))]
        cpu_vals = [r.get("cpu_percent", 0) for r in rows
                    if isinstance(r.get("cpu_percent"), (int, float))]
        print(f"  avg p95={sum(p95_vals)/len(p95_vals):.1f}ms  "
              f"total errors={sum(err_vals)}  "
              f"avg cpu={sum(cpu_vals)/len(cpu_vals):.1f}%" if p95_vals else "")

    return rows


def main():
    parser = argparse.ArgumentParser(description="PERM Acceptance Run")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--workers-fixed", type=int, default=2)
    parser.add_argument("--collector", default="http://127.0.0.1:5000")
    parser.add_argument("--remediation", default="http://127.0.0.1:5003")
    parser.add_argument("--concurrency", type=int, default=30,
                        help="Concurrent request senders (default 30)")
    parser.add_argument("--load", type=int, default=8,
                        help="load value per request (default 8)")
    args = parser.parse_args()

    # Health checks
    print("Checking services…")
    if not _wait_healthy(args.collector, timeout=10.0):
        print(f"[✗] Collector not healthy at {args.collector}")
        sys.exit(1)
    print(f"  [✓] Collector at {args.collector}")

    try:
        r = requests.get(f"{args.remediation}/health", timeout=3.0)
        remediation_ok = r.status_code == 200
    except Exception:
        remediation_ok = False

    if not remediation_ok:
        print(f"  [!] Remediation service not available at {args.remediation}")
        print("      Running with MANUAL-only mode (no auto-scaling).")

    all_rows = []

    # Pass 1: Fixed capacity
    rows1 = run_pass(
        collector_url=args.collector,
        remediation_url=args.remediation,
        pass_name="fixed_2workers",
        fixed_workers=args.workers_fixed,
        auto_scale=False,
        duration_sec=args.duration,
        load=args.load,
        concurrency=args.concurrency,
    )
    all_rows.extend(rows1)

    # Brief cooldown between passes
    print("\n[*] Cooldown 10s before pass 2…")
    _set_capacity(args.collector, 2)  # reset to baseline
    time.sleep(10)

    # Pass 2: Auto-scaling
    rows2 = run_pass(
        collector_url=args.collector,
        remediation_url=args.remediation,
        pass_name="auto_scaling",
        fixed_workers=args.workers_fixed,
        auto_scale=remediation_ok,
        duration_sec=args.duration,
        load=args.load,
        concurrency=args.concurrency,
    )
    all_rows.extend(rows2)

    # Write CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n[✓] Results saved to {OUTPUT_PATH}")

    # Summary table
    def _summarize(rows: List[dict], label: str):
        p95s = [r["p95_latency_ms"] for r in rows
                if isinstance(r.get("p95_latency_ms"), (int, float))]
        errs = [r["error_count"] for r in rows
                if isinstance(r.get("error_count"), (int, float))]
        cpus = [r["cpu_percent"] for r in rows
                if isinstance(r.get("cpu_percent"), (int, float))]
        ws   = [r["active_workers"] for r in rows
                if isinstance(r.get("active_workers"), (int, float))]
        print(f"\n  {label}:")
        print(f"    p95 avg={sum(p95s)/len(p95s):.1f}ms  max={max(p95s):.1f}ms" if p95s else "")
        print(f"    errors total={sum(errs)}" if errs else "")
        print(f"    cpu avg={sum(cpus)/len(cpus):.1f}%" if cpus else "")
        print(f"    workers seen={sorted(set(ws))}" if ws else "")

    print("\n" + "="*60)
    print("ACCEPTANCE SUMMARY")
    print("="*60)
    _summarize(rows1, "Pass 1 – Fixed 2 workers")
    _summarize(rows2, "Pass 2 – Auto-scaling")
    print("="*60)


if __name__ == "__main__":
    main()
