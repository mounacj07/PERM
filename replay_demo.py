"""
replay_demo.py
==============
Streams a held-out run second by second through the online feature extraction
and ST-AGBDT model. Outputs forecast, upper bound, recommended workers vs
actual future demand requirement.

Saves CSV and PNG visualization for evaluation and auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stagbdt.features import (
    FEATURE_NAMES,
    StreamingArrivalFeatureExtractor,
    extract_features_single,
    extract_features_sequence,
)
from stagbdt.model import STAGBDT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("replay_demo")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "models", "stagbdt_final_config.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "replay")


def replay_run(
    run_id: str,
    horizon: int = 20,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Stream a recorded run second by second.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}. Run train_and_eval_final.py first.")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    h_key = f"h{horizon}"
    if h_key not in config["conformal_margins"]:
        raise ValueError(f"Horizon h={horizon} not found in config conformal_margins")

    margin = float(config["conformal_margins"][h_key])
    eff_cap = float(config["effective_capacity_per_worker"])
    model_file_rel = config["model_files"][h_key]
    model_file = os.path.join(PROJECT_ROOT, model_file_rel)

    log.info("Loading ST-AGBDT model from %s...", model_file_rel)
    model = STAGBDT.load_json(model_file)

    csv_path = os.path.join(DATA_DIR, f"{run_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    T = len(raw_rows)
    arr = np.array([float(r.get("arrivals_count", 0.0)) for r in raw_rows], dtype=np.float64)

    # Compute ground truth smoothed arrivals for target evaluation
    smoothed_arr = np.empty_like(arr)
    for i in range(T):
        smoothed_arr[i] = np.mean(arr[max(0, i - 2) : i + 1])

    # Streaming state
    extractor = StreamingArrivalFeatureExtractor(max_buffer_len=150)

    replay_records: List[Dict[str, Any]] = []

    log.info("Streaming run '%s' (%d seconds, horizon=%ds)...", run_id, T, horizon)

    for t in range(T):
        curr_arr = arr[t]
        feats = extractor.push(curr_arr)

        # Valid prediction window: t >= 10 and future target within trace t + horizon < T
        if 10 <= t < (T - horizon):
            # 1. Feature inference via pure numpy model
            raw_forecast = float(model.predict_one(feats))
            upper_bound = raw_forecast + margin

            # 2. Recommended workers
            rec_workers = int(np.clip(math.ceil(upper_bound / eff_cap), 2, 8))
            rec_workers_raw = int(np.clip(math.ceil(raw_forecast / eff_cap), 2, 8))

            # 3. Ground truth target: max 3s smoothed arrivals over next horizon seconds
            future_window = smoothed_arr[t + 1 : t + horizon + 1]
            actual_future_max = float(np.max(future_window))
            actual_workers_needed = int(np.clip(math.ceil(actual_future_max / eff_cap), 2, 8))

            is_miss = rec_workers < actual_workers_needed
            extra_workers = max(0, rec_workers - actual_workers_needed)

            replay_records.append({
                "second": t,
                "current_arrivals": round(curr_arr, 2),
                "current_smoothed": round(float(smoothed_arr[t]), 2),
                "raw_forecast": round(raw_forecast, 3),
                "conformal_margin": round(margin, 3),
                "upper_bound": round(upper_bound, 3),
                "recommended_workers": rec_workers,
                "recommended_workers_raw": rec_workers_raw,
                "actual_future_max_smoothed": round(actual_future_max, 3),
                "actual_workers_needed": actual_workers_needed,
                "is_miss": bool(is_miss),
                "extra_workers": extra_workers,
            })

    # Summary statistics
    n_pts = len(replay_records)
    misses = sum(1 for r in replay_records if r["is_miss"])
    miss_pct = (misses / n_pts * 100.0) if n_pts > 0 else 0.0
    mean_extra = float(np.mean([r["extra_workers"] for r in replay_records])) if n_pts > 0 else 0.0
    diffs = [abs(r["raw_forecast"] - r["actual_future_max_smoothed"]) for r in replay_records]
    mae = float(np.mean(diffs)) if diffs else 0.0

    log.info("Replay summary for %s (h=%ds):", run_id, horizon)
    log.info("  Valid time steps: %d", n_pts)
    log.info("  Forecast MAE: %.3f req/s", mae)
    log.info("  Workers needed miss rate: %d / %d (%.2f%%)", misses, n_pts, miss_pct)
    log.info("  Mean extra workers: %.3f", mean_extra)

    # Save CSV
    csv_out_path = os.path.join(output_dir, f"replay_{run_id}.csv")
    png_out_path = os.path.join(output_dir, f"replay_{run_id}.png")

    if save_artifacts and replay_records:
        with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(replay_records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(replay_records)
        log.info("Saved replay CSV to %s", csv_out_path)

        # Plot
        seconds = [r["second"] for r in replay_records]
        arr_vals = [r["current_arrivals"] for r in replay_records]
        smooth_vals = [r["current_smoothed"] for r in replay_records]
        raw_fc = [r["raw_forecast"] for r in replay_records]
        upper_fc = [r["upper_bound"] for r in replay_records]
        actual_dem = [r["actual_future_max_smoothed"] for r in replay_records]

        rec_w = [r["recommended_workers"] for r in replay_records]
        act_w = [r["actual_workers_needed"] for r in replay_records]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})

        # Panel 1: Demand & Forecast
        ax1.plot(seconds, arr_vals, color="#cccccc", label="1s Arrivals", linewidth=1.0, alpha=0.8)
        ax1.plot(seconds, smooth_vals, color="#7f7f7f", label="3s Smoothed", linewidth=1.2)
        ax1.plot(seconds, actual_dem, color="#d62728", label=f"Actual Future Max Demand (h={horizon}s)", linewidth=2.0)
        ax1.plot(seconds, raw_fc, color="#1f77b4", linestyle="--", label="ST-AGBDT Raw (alpha=0.85)", linewidth=1.8)
        ax1.plot(seconds, upper_fc, color="#2ca02c", label=f"ST-AGBDT + 90% Conformal (+{margin:.2f} rps)", linewidth=2.2)
        ax1.fill_between(seconds, raw_fc, upper_fc, color="#2ca02c", alpha=0.15, label="Conformal Margin Buffer")

        ax1.set_title(f"Streaming Forecaster Replay: {run_id} (Horizon h = {horizon}s)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Arrival Rate (req/s)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Panel 2: Workers Allocation vs Requirement
        ax2.step(seconds, act_w, where="mid", color="#d62728", label="Actual Workers Needed", linewidth=2.0)
        ax2.step(seconds, rec_w, where="mid", color="#2ca02c", label="Recommended Workers (90% Conformal)", linewidth=2.2)

        # Shade violations
        sec_arr = np.array(seconds)
        rec_arr = np.array(rec_w)
        act_arr = np.array(act_w)
        ax2.fill_between(sec_arr, rec_arr, act_arr, where=(rec_arr < act_arr), color="#d62728", alpha=0.4, step="mid", label="Under-Provisioning Breach")
        ax2.fill_between(sec_arr, rec_arr, act_arr, where=(rec_arr >= act_arr), color="#2ca02c", alpha=0.15, step="mid", label="Safe Operational Headroom")

        ax2.set_xlabel("Replay Time (seconds)", fontsize=11)
        ax2.set_ylabel("Workers (N)", fontsize=11)
        ax2.set_ylim(1, 9)
        ax2.set_yticks([2, 4, 6, 8])
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(png_out_path, dpi=200, bbox_inches="tight")
        plt.close()
        log.info("Saved replay plot to %s", png_out_path)

    return {
        "run_id": run_id,
        "horizon": horizon,
        "valid_steps": n_pts,
        "mae": round(mae, 4),
        "misses": misses,
        "miss_pct": round(miss_pct, 4),
        "mean_extra_workers": round(mean_extra, 4),
        "csv_path": csv_out_path,
        "png_path": png_out_path,
        "records": replay_records,
    }


def main():
    parser = argparse.ArgumentParser(description="ST-AGBDT Streaming Replay Demo")
    parser.add_argument("--run", default="random_mix_seed20", help="Run ID to stream (e.g. random_mix_seed20, spike_seed44)")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon (10, 20, or 30)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save CSV and PNG")
    args = parser.parse_args()

    replay_run(args.run, horizon=args.horizon, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
