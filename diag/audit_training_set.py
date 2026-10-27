"""
diag/audit_training_set.py
==========================
READ-ONLY audit of the ST-AGBDT h=20 training dataset and configuration.
Evaluates:
  a) Fraction of calm windows (arrivals < 2x baseline for last 10s) whose target is a burst (> 2x baseline).
  b) Mean and 85th percentile of target for calm windows vs calm windows from steady profiles only.
  c) Feature dependency on elapsed time, short history padding, cycle position.
  d) Burst onset offsets per profile and seed across training runs.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_and_eval_final import get_run_splits, DATA_DIR, load_capacity_per_worker


def audit_training_set():
    print("=" * 80)
    print("PART 3: TRAINING-SET AUDIT (ST-AGBDT h=20 MODEL)")
    print("=" * 80)

    training_runs, calib_runs, test_runs, excl_runs = get_run_splits()
    print(f"Total training runs ({len(training_runs)}): {training_runs}")
    print(f"Calibration runs ({len(calib_runs)}): {calib_runs}")
    print(f"Held-out test runs ({len(test_runs)}): {test_runs}")
    print(f"Excluded runs ({len(excl_runs)}): {excl_runs}")

    h = 20
    baseline_rate = 4.0
    calm_threshold = 2.0 * baseline_rate  # 8.0 req/s
    burst_threshold = 2.0 * baseline_rate  # 8.0 req/s

    # Containers for calm windows
    calm_targets_all: List[float] = []
    calm_raw_target_at_th: List[float] = []
    calm_targets_steady_only: List[float] = []

    # Check onset offsets per profile and seed
    burst_onsets: Dict[str, float] = {}

    for run_id in training_runs:
        csv_path = os.path.join(DATA_DIR, f"{run_id}.csv")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        arr = np.array([float(r.get("arrivals_count", 0.0)) for r in rows], dtype=np.float64)
        T = len(arr)

        # 3s smoothed arrivals (matching train_and_eval_final.py)
        smoothed_arr = np.empty_like(arr)
        for i in range(T):
            smoothed_arr[i] = np.mean(arr[max(0, i - 2) : i + 1])

        # Find first index where smoothed_arr > burst_threshold (burst onset)
        onset_idx = None
        for i in range(T):
            if smoothed_arr[i] >= burst_threshold:
                onset_idx = i
                break
        burst_onsets[run_id] = float(onset_idx) if onset_idx is not None else -1.0

        is_steady = "flat" in run_id or "steady" in run_id

        # Valid indices t in [10, T - h) per train_and_eval_final.py line 166
        for t in range(10, T - h):
            # Check if calm in the last 10s: arrivals < 2x baseline (8.0 req/s) for all tau in [t-9, t]
            window_10s = arr[t - 9 : t + 1]
            if np.all(window_10s < calm_threshold):
                # Target in training: max(smoothed_arr[t+1 : t+h+1])
                fut_3s = smoothed_arr[t + 1 : t + h + 1]
                target_train = float(np.max(fut_3s))
                # Raw arrival target at t+h
                raw_at_th = float(arr[t + h])

                calm_targets_all.append(target_train)
                calm_raw_target_at_th.append(raw_at_th)

                if is_steady:
                    calm_targets_steady_only.append(target_train)

    calm_all = np.array(calm_targets_all)
    calm_th = np.array(calm_raw_target_at_th)
    calm_steady = np.array(calm_targets_steady_only) if calm_targets_steady_only else np.array([])

    print("\n--- a) Fraction of Calm Windows Whose Target is a Burst (> 2x baseline = 8 req/s) ---")
    print(f"Total calm windows (t in [10, T-20] with last 10s arrivals < 8 req/s): {len(calm_all)}")
    burst_targets_max = np.sum(calm_all > burst_threshold)
    frac_max = burst_targets_max / len(calm_all) if len(calm_all) > 0 else 0.0
    print(f"Using training target max(smoothed[t+1..t+20]):")
    print(f"  Count with target > 8.0 req/s: {burst_targets_max} / {len(calm_all)}")
    print(f"  Fraction: {frac_max:.4f} ({frac_max * 100:.2f}%)")

    burst_targets_th = np.sum(calm_th > burst_threshold)
    frac_th = burst_targets_th / len(calm_th) if len(calm_th) > 0 else 0.0
    print(f"Using point target arrivals[t+20]:")
    print(f"  Count with target > 8.0 req/s: {burst_targets_th} / {len(calm_th)}")
    print(f"  Fraction: {frac_th:.4f} ({frac_th * 100:.2f}%)")

    print("\n--- b) Target Distributions for Calm Windows ---")
    print(f"All calm windows in training set:")
    print(f"  Mean target (forward max) : {calm_all.mean():.4f} req/s")
    print(f"  85th percentile           : {np.percentile(calm_all, 85):.4f} req/s")
    print(f"  Median (50th percentile)  : {np.median(calm_all):.4f} req/s")
    print(f"  Min / Max                 : {calm_all.min():.4f} / {calm_all.max():.4f} req/s")
    print(f"Point target at t+h (arrivals[t+20]):")
    print(f"  Mean target at t+h        : {calm_th.mean():.4f} req/s")
    print(f"  85th percentile at t+h    : {np.percentile(calm_th, 85):.4f} req/s")
    print(f"Calm windows from steady profiles only:")
    if len(calm_steady) > 0:
        print(f"  Count: {len(calm_steady)}")
        print(f"  Mean target: {calm_steady.mean():.4f} req/s")
        print(f"  85th percentile: {np.percentile(calm_steady, 85):.4f} req/s")
    else:
        print("  NOTE: No dedicated 'steady' or 'flat' runs exist in data/runs_locust training set!")
        print("  All runs in training set are burst/dynamic profiles (step, ramp, spike, double_burst, random_walk, random_mix)!")

    print("\n--- c) Feature Dependency on Time & Short History Handling ---")
    print("Does any feature depend on:")
    print("  1. Elapsed time since run start? NO elapsed time feature is included.")
    print("     Features are strictly: ['arr_curr'] + [arr_{mean,max,std,slope,accel,ewma}_{5,10,30,60}].")
    print("  2. Padding when history is shorter than longest window (60s)?")
    print("     No zero-padding, edge-padding, or NaN is used.")
    print("     Code in stagbdt/features.py lines 64-65 simply takes: win = arr[max(0, n - w):]")
    print("     So for n < w, win has length n, computing statistics over truncated window.")
    print("  3. Cycle position? No cyclic features or timestamp features exist.")

    print("\n--- d) Burst Onset Offsets Per Profile and Seed in Training Runs ---")
    print("Run ID                           | Onset Offset (sec) | Profile Type")
    print("---------------------------------+--------------------+------------------")
    for r_id, onset in sorted(burst_onsets.items()):
        prof = r_id.split("_seed")[0] if "_seed" in r_id else r_id
        print(f"{r_id:32s} | {onset:18.1f} | {prof}")

    # Summarize fixed onset profiles
    print("\nOnset offset pattern:")
    for prof in ["step", "ramp", "spike", "double_burst", "random_walk", "random_mix"]:
        prof_onsets = [burst_onsets[r] for r in burst_onsets if r.startswith(prof)]
        if prof_onsets:
            print(f"  Profile '{prof}': onsets = {prof_onsets}")


if __name__ == "__main__":
    audit_training_set()
