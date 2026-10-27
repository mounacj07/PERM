"""
measure_idle_baseline.py
========================
F3 Idle baseline measurement:
Logs 60 s of system CPU and the top 10 processes by CPU consumption
with none of our benchmark processes running.
"""

import time
import psutil
import numpy as np
from collections import defaultdict


def main():
    print("Starting 60s idle baseline measurement (60 samples)...")
    sys_samples = []
    proc_samples = defaultdict(list)

    # Prime psutil baseline
    psutil.cpu_percent(None)
    for p in psutil.process_iter(["pid", "name"]):
        try:
            p.cpu_percent(None)
        except Exception:
            pass

    t0 = time.time()
    for tick in range(60):
        time.sleep(1.0)
        c = psutil.cpu_percent(None)
        sys_samples.append(c)

        for p in psutil.process_iter(["pid", "name"]):
            try:
                pc = p.cpu_percent(None)
                if pc > 0.0:
                    proc_samples[(p.info["pid"], p.info["name"])].append(pc)
            except Exception:
                pass

    elapsed = time.time() - t0
    sys_arr = np.array(sys_samples)

    print("\n" + "=" * 70)
    print("60-SECOND IDLE SYSTEM CPU BASELINE")
    print("=" * 70)
    print(f"Elapsed time:      {elapsed:.1f}s")
    print(f"Samples recorded:  {len(sys_arr)}")
    print(f"Mean System CPU:   {np.mean(sys_arr):.2f}%")
    print(f"Median System CPU: {np.median(sys_arr):.2f}%")
    print(f"Min System CPU:    {np.min(sys_arr):.2f}%")
    print(f"Max System CPU:    {np.max(sys_arr):.2f}%")
    print(f"p90 System CPU:    {np.percentile(sys_arr, 90):.2f}%")
    print(f"p95 System CPU:    {np.percentile(sys_arr, 95):.2f}%")
    print(f"p99 System CPU:    {np.percentile(sys_arr, 99):.2f}%")

    # Ranking
    ranked = sorted(
        [(name, pid, np.sum(vals) / 60.0) for (pid, name), vals in proc_samples.items()],
        key=lambda x: x[2],
        reverse=True,
    )

    print("\n" + "=" * 85)
    print("TOP 10 PROCESSES BY CPU CONSUMPTION OVER 60 SECONDS")
    print("=" * 85)
    header = f"{'Rank':<4} | {'Process Name':<28} | {'PID':<6} | {'Mean Core %':<12} | {'Machine %':<12} | {'Threads':<10}"
    print(header)
    print("-" * 85)

    for i, (name, pid, avg_core) in enumerate(ranked[:10], 1):
        mach_pct = avg_core / 16.0
        th_eq = avg_core / 100.0
        print(f"{i:<4} | {name:<28} | {pid:<6} | {avg_core:>10.2f}% | {mach_pct:>10.2f}% | {th_eq:>9.2f}")


if __name__ == "__main__":
    main()
