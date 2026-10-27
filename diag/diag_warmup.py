"""
diag/diag_warmup.py
===================
Offline diagnostic for ST-AGBDT h=20 forecaster warm-up behavior.
Simulates constant 4 req/s arrivals with Poisson noise (seed 44), 120s long.
Evaluates extract_features_single and h=20 predict_one for t = 3..120.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stagbdt.features import extract_features_single, FEATURE_NAMES, WINDOWS
from stagbdt.model import STAGBDT

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "stagbdt_h20_lean.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "stagbdt_final_config.json")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_warmup_results.csv")


def inspect_short_history_handling():
    """Analyze and print how extract_features_single handles short history."""
    print("=" * 70)
    print("ANALYSIS: How extract_features_single handles short histories (< longest window)")
    print("=" * 70)
    code_quote = """
    From stagbdt/features.py (lines 62-90):

    n = len(arr)
    for w in WINDOWS:
        win = arr[max(0, n - w):]
        win_len = len(win)

        mean_val = float(np.mean(win))
        max_val = float(np.max(win))
        std_val = float(np.std(win)) if win_len > 1 else 0.0
        slope_val = float((win[-1] - win[0]) / max(1.0, float(win_len - 1)))

        half_w = max(1, win_len // 2)
        s_recent = float((win[-1] - win[-half_w]) / float(half_w))
        s_older = float((win[-half_w] - win[0]) / float(half_w))
        accel_val = float((s_recent - s_older) / float(half_w))

        # EWMA over window max(0, n - 2*w) to n
        ewma_win = arr[max(0, n - 2 * w):]
        alpha_ewma = 2.0 / (w + 1.0)
        ewma = float(ewma_win[0])
        for v in ewma_win[1:]:
            ewma = alpha_ewma * float(v) + (1.0 - alpha_ewma) * ewma
    """
    print(code_quote)
    print("Finding:")
    print("1. Truncated slice (NO padding): When n < w, win = arr[max(0, n - w):] = arr[0:n].")
    print("   No zero-padding, edge-padding, or NaN is used. Instead, it computes statistics")
    print("   over whatever short history exists (e.g. for w=60 at t=5, win has length 5).")
    print("2. Slope & Acceleration normalization: Divided by max(1.0, win_len - 1) and half_w,")
    print("   which scales by the actual available sample count rather than full window w.")
    print("3. EWMA initialization: Initializes with ewma_win[0] (which is arr[0]), then loops.")
    print("=" * 70)


def run_warmup_diag(seed: int = 44, use_default_rng: bool = True) -> Tuple[int, int, float, float]:
    # Load configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    conformal_margin = float(config.get("conformal_margins", {}).get("h20", 5.0763))
    effective_capacity = float(config.get("effective_capacity_per_worker", 5.523))
    capacity_per_worker = float(config.get("capacity_per_worker", 7.89))
    target_utilization = float(config.get("target_utilization", 0.70))

    # Load h=20 model
    model = STAGBDT.load_json(MODEL_PATH)

    # Generate constant 4 req/s with Poisson noise for 120s
    if use_default_rng:
        rng = np.random.default_rng(seed)
        arrivals = rng.poisson(lam=4.0, size=120).astype(np.float64)
    else:
        rng_rs = np.random.RandomState(seed)
        arrivals = rng_rs.poisson(lam=4.0, size=120).astype(np.float64)

    rows = []
    first_t_ge_3 = None
    max_rec_workers = 2
    max_raw_pred = -1.0
    max_upper = -1.0

    print(f"\nEvaluating h=20 predictions for constant 4 req/s (seed={seed}, default_rng={use_default_rng})...")
    print(f"Arrivals sample (first 10s): {list(arrivals[:10])}")
    print(f"Arrivals mean: {arrivals.mean():.2f}, std: {arrivals.std():.2f}, min: {arrivals.min()}, max: {arrivals.max()}")

    for t in range(3, 121):
        hist = arrivals[:t]
        feats = extract_features_single(hist)
        raw_pred = float(model.predict_one(feats))
        upper_bound = raw_pred + conformal_margin
        rec_workers = int(np.clip(math.ceil(upper_bound / effective_capacity), 2, 8))

        if rec_workers >= 3 and first_t_ge_3 is None:
            first_t_ge_3 = t
        if rec_workers > max_rec_workers:
            max_rec_workers = rec_workers
        if raw_pred > max_raw_pred:
            max_raw_pred = raw_pred
        if upper_bound > max_upper:
            max_upper = upper_bound

        row = {
            "t": t,
            "arrivals_t": hist[-1],
            "raw_forecast": round(raw_pred, 4),
            "conformal_margin": round(conformal_margin, 4),
            "upper_bound": round(upper_bound, 4),
            "recommended_workers": rec_workers,
        }
        # Include feature values
        for idx, feat_name in enumerate(FEATURE_NAMES):
            row[feat_name] = round(float(feats[idx]), 4)
        rows.append(row)

    # Save to CSV
    if rows:
        fieldnames = ["t", "arrivals_t", "raw_forecast", "conformal_margin", "upper_bound", "recommended_workers"] + FEATURE_NAMES
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} per-t evaluations to {OUTPUT_CSV}")

    print("\n" + "=" * 70)
    print("PART 1 RESULTS SUMMARY:")
    print(f"  First t with recommended_workers >= 3 : {first_t_ge_3} (at t={first_t_ge_3}s)")
    print(f"  Maximum recommended_workers over run  : {max_rec_workers} workers")
    print(f"  Maximum raw forecast                  : {max_raw_pred:.4f} req/s")
    print(f"  Maximum upper bound                   : {max_upper:.4f} req/s")
    print(f"  Conformal margin                      : {conformal_margin:.4f} req/s")
    print(f"  Effective capacity per worker         : {effective_capacity:.3f} req/s (cap_per_worker={capacity_per_worker} * {target_utilization})")
    print(f"  Threshold for 3 workers               : > {2 * effective_capacity:.3f} req/s upper bound (raw > {2 * effective_capacity - conformal_margin:.4f} req/s)")
    print("=" * 70)

    # Print sample steps around warm-up and steady state
    print("\nSample trace steps around t=3..15, 30, 60, 90, 120:")
    print("   t | arr[t] | raw_fc | margin | upper | rec_w | arr_mean_10 | arr_max_10 | arr_mean_60")
    print("-----+--------+--------+--------+-------+-------+-------------+------------+------------")
    for r in rows:
        if r["t"] in [3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 60, 90, 120]:
            print(f" {r['t']:3d} | {r['arrivals_t']:6.1f} | {r['raw_forecast']:6.2f} | {r['conformal_margin']:6.2f} | {r['upper_bound']:5.2f} |   {r['recommended_workers']:d}   | {r['arr_mean_10']:11.2f} | {r['arr_max_10']:10.2f} | {r['arr_mean_60']:10.2f}")

    return first_t_ge_3, max_rec_workers, max_raw_pred, max_upper


if __name__ == "__main__":
    inspect_short_history_handling()
    print("\n--- Running with numpy.random.default_rng(44) ---")
    run_warmup_diag(seed=44, use_default_rng=True)
    print("\n--- Running with numpy.random.RandomState(44) for comparison ---")
    run_warmup_diag(seed=44, use_default_rng=False)
