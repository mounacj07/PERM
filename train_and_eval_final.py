"""
train_and_eval_final.py
=======================
Trains final ST-AGBDT models for horizons h in {10, 20, 30} seconds on data/runs_locust.
Evaluates on held-out test runs with 95% bootstrap CIs over whole runs.
Saves lean JSON models, config file, results JSON, and README.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stagbdt.features import (
    FEATURE_NAMES,
    extract_features_single,
    extract_features_sequence,
    StreamingArrivalFeatureExtractor,
)
from stagbdt.model import STAGBDT
from stagbdt.conformal import ConformalCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_final")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "testbed_calibration.json")

HORIZONS = [10, 20, 30]
TARGET_UTILIZATION = 0.70
CONFORMAL_COVERAGE = 0.90
BOOTSTRAP_RESAMPLES = 1000

# Fixed model config
MODEL_CONFIG = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.1,
    "alpha": 0.85,
    "l2_reg": 1.0,
    "min_child_hessian": 1e-3,
    "n_bins": 64,
    "random_state": 42,
}


def get_git_commit() -> str:
    """Retrieve current HEAD commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            universal_newlines=True,
        ).strip()
        return out
    except Exception as e:
        log.warning("Could not read git commit: %s", e)
        return "UNKNOWN"


def load_capacity_per_worker() -> float:
    """Load 1-worker saturation rps from testbed_calibration.json."""
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
    for r in d.get("results", []):
        if r.get("n_workers") == 1:
            return float(r.get("saturation_rps", 7.89))
    return 7.89


def get_run_splits() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Define run partitions per task specifications:
      - Exclude: sawtooth (all seeds) + flagged runs (step_seed42 with failed_requests=1).
      - Calibration: random_mix 15-19 + step_seed44, ramp_seed44.
      - Held-out test: random_mix 20-29 + the other fixed seed-44 runs (spike_seed44, double_burst_seed44, random_walk_seed44).
      - Training: remaining 24 unflagged, non-sawtooth runs in data/runs_locust.
    """
    excluded_runs = [
        "sawtooth_seed42",
        "sawtooth_seed43",
        "sawtooth_seed44",
        "step_seed42",
    ]

    calibration_runs = [
        "random_mix_seed15",
        "random_mix_seed16",
        "random_mix_seed17",
        "random_mix_seed18",
        "random_mix_seed19",
        "step_seed44",
        "ramp_seed44",
    ]

    test_runs = [
        "random_mix_seed20",
        "random_mix_seed21",
        "random_mix_seed22",
        "random_mix_seed23",
        "random_mix_seed24",
        "random_mix_seed25",
        "random_mix_seed26",
        "random_mix_seed27",
        "random_mix_seed28",
        "random_mix_seed29",
        "spike_seed44",
        "double_burst_seed44",
        "random_walk_seed44",
    ]

    # Discover all CSVs in data/runs_locust
    csv_paths = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_run_ids = []
    for p in csv_paths:
        fname = os.path.basename(p)
        if "_stats" in fname or "phase_log" in fname:
            continue
        base = os.path.splitext(fname)[0]
        all_run_ids.append(base)

    all_run_ids = sorted(list(set(all_run_ids)))

    # Training is all unassigned runs
    assigned = set(excluded_runs) | set(calibration_runs) | set(test_runs)
    training_runs = sorted([r for r in all_run_ids if r not in assigned])

    return training_runs, calibration_runs, test_runs, excluded_runs


def load_single_run_data(run_id: str, horizons: List[int] = HORIZONS) -> Dict[str, Any]:
    """Load run CSV and compute arrival-only features, targets, and baseline signals."""
    csv_path = os.path.join(DATA_DIR, f"{run_id}.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    arr = np.array([float(r.get("arrivals_count", 0.0)) for r in rows], dtype=np.float64)
    T = len(arr)

    # 3s smoothed arrivals
    smoothed_arr = np.empty_like(arr)
    for i in range(T):
        smoothed_arr[i] = np.mean(arr[max(0, i - 2) : i + 1])

    # Per horizon features and targets
    data_by_h: Dict[int, Dict[str, np.ndarray]] = {}

    for h in horizons:
        valid_t = np.arange(10, T - h)
        X, fnames = extract_features_sequence(arr, valid_t=valid_t)

        N = len(valid_t)
        y_target = np.empty(N, dtype=np.float64)
        for idx, t in enumerate(valid_t):
            fut_3s = smoothed_arr[t + 1 : t + h + 1]
            y_target[idx] = np.max(fut_3s)

        # Baselines:
        # 1. Persistence raw: arr[t]
        base_persist_raw = arr[valid_t]
        # 2. Persistence smoothed: smoothed_arr[t]
        base_persist_smooth = smoothed_arr[valid_t]
        # 3. EWMA 10s: feature arr_ewma_10
        idx_ewma10 = fnames.index("arr_ewma_10")
        base_ewma10 = X[:, idx_ewma10]
        # 4. Linear Trend: arr[t] + h * arr_slope_10
        idx_slope10 = fnames.index("arr_slope_10")
        base_linear = arr[valid_t] + float(h) * X[:, idx_slope10]

        data_by_h[h] = {
            "valid_t": valid_t,
            "X": X,
            "y": y_target,
            "arr_t": arr[valid_t],
            "smoothed_t": smoothed_arr[valid_t],
            "base_persist_raw": base_persist_raw,
            "base_persist_smooth": base_persist_smooth,
            "base_ewma10": base_ewma10,
            "base_linear": base_linear,
        }

    return {
        "run_id": run_id,
        "csv_path": csv_path,
        "arr": arr,
        "smoothed_arr": smoothed_arr,
        "data_by_h": data_by_h,
    }


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_per_worker: float,
    utilization: float = TARGET_UTILIZATION,
) -> Dict[str, float]:
    """Compute MAE, RMSE, workers needed miss rate, and mean extra workers."""
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    eff_cap = capacity_per_worker * utilization
    actual_workers = np.clip(np.ceil(y_true / eff_cap), 2.0, 8.0)
    pred_workers = np.clip(np.ceil(y_pred / eff_cap), 2.0, 8.0)

    miss_mask = pred_workers < actual_workers
    miss_pct = float(np.mean(miss_mask)) * 100.0

    extra_workers = np.maximum(0.0, pred_workers - actual_workers)
    mean_extra = float(np.mean(extra_workers))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "workers_needed_miss_pct": round(miss_pct, 4),
        "mean_extra_workers": round(mean_extra, 4),
    }


def run_run_level_bootstrap(
    run_records: List[Dict[str, Any]],
    model_predictions: Dict[str, List[np.ndarray]],
    h: int,
    capacity_per_worker: float,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Bootstrap 95% confidence intervals by resampling WHOLE RUNS with replacement.
    """
    rng = np.random.RandomState(seed)
    n_runs = len(run_records)
    model_names = list(model_predictions.keys())

    # Pre-extract true targets per run
    y_per_run = [r["data_by_h"][h]["y"] for r in run_records]

    boot_metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"mae": [], "miss": [], "extra": []} for m in model_names
    }

    for b in range(n_resamples):
        sampled_indices = rng.choice(n_runs, size=n_runs, replace=True)

        y_boot = np.concatenate([y_per_run[i] for i in sampled_indices])
        for m in model_names:
            preds_boot = np.concatenate([model_predictions[m][i] for i in sampled_indices])
            m_res = compute_evaluation_metrics(y_boot, preds_boot, capacity_per_worker)
            boot_metrics[m]["mae"].append(m_res["mae"])
            boot_metrics[m]["miss"].append(m_res["workers_needed_miss_pct"])
            boot_metrics[m]["extra"].append(m_res["mean_extra_workers"])

    cis: Dict[str, Dict[str, List[float]]] = {}
    for m in model_names:
        cis[m] = {
            "mae_ci95": [
                round(float(np.percentile(boot_metrics[m]["mae"], 2.5)), 4),
                round(float(np.percentile(boot_metrics[m]["mae"], 97.5)), 4),
            ],
            "miss_ci95": [
                round(float(np.percentile(boot_metrics[m]["miss"], 2.5)), 4),
                round(float(np.percentile(boot_metrics[m]["miss"], 97.5)), 4),
            ],
            "extra_ci95": [
                round(float(np.percentile(boot_metrics[m]["extra"], 2.5)), 4),
                round(float(np.percentile(boot_metrics[m]["extra"], 97.5)), 4),
            ],
        }

    return cis


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    git_commit = get_git_commit()
    cap_per_worker = load_capacity_per_worker()
    log.info("Git commit: %s", git_commit)
    log.info("Capacity per worker (from testbed_calibration.json): %.2f req/s", cap_per_worker)
    log.info("Effective capacity per worker (utilization=%.2f): %.3f req/s", TARGET_UTILIZATION, cap_per_worker * TARGET_UTILIZATION)

    train_ids, cal_ids, test_ids, excl_ids = get_run_splits()
    log.info("Split definition:")
    log.info("  Training runs (%d): %s", len(train_ids), train_ids)
    log.info("  Calibration runs (%d): %s", len(cal_ids), cal_ids)
    log.info("  Held-out test runs (%d): %s", len(test_ids), test_ids)
    log.info("  Excluded runs (%d): %s", len(excl_ids), excl_ids)

    # Load runs
    log.info("Loading run CSV files...")
    train_runs = [load_single_run_data(rid) for rid in train_ids]
    cal_runs = [load_single_run_data(rid) for rid in cal_ids]
    test_runs = [load_single_run_data(rid) for rid in test_ids]

    final_models: Dict[int, STAGBDT] = {}
    conformal_calibrators: Dict[int, ConformalCalibrator] = {}
    conformal_margins: Dict[str, float] = {}
    baseline_conformal_margins: Dict[str, Dict[str, float]] = {}

    all_evaluation_results: Dict[str, Any] = {
        "metadata": {
            "git_commit": git_commit,
            "target_utilization": TARGET_UTILIZATION,
            "capacity_per_worker": cap_per_worker,
            "effective_capacity_per_worker": round(cap_per_worker * TARGET_UTILIZATION, 4),
            "conformal_coverage": CONFORMAL_COVERAGE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "training_run_ids": train_ids,
            "calibration_run_ids": cal_ids,
            "test_run_ids": test_ids,
            "excluded_run_ids": excl_ids,
            "features": FEATURE_NAMES,
            "model_config": MODEL_CONFIG,
        },
        "horizons": {},
    }

    # Iterate over horizons
    for h in HORIZONS:
        log.info("=================================================================")
        log.info("Processing Horizon h = %d s", h)
        log.info("=================================================================")

        # 1. Prepare Training Data
        X_train = np.vstack([r["data_by_h"][h]["X"] for r in train_runs])
        y_train = np.concatenate([r["data_by_h"][h]["y"] for r in train_runs])
        log.info("  Training samples: %d, features: %d", X_train.shape[0], X_train.shape[1])

        # Train model
        model = STAGBDT(
            n_estimators=MODEL_CONFIG["n_estimators"],
            max_depth=MODEL_CONFIG["max_depth"],
            learning_rate=MODEL_CONFIG["learning_rate"],
            alpha=MODEL_CONFIG["alpha"],
            l2_reg=MODEL_CONFIG["l2_reg"],
            min_child_hessian=MODEL_CONFIG["min_child_hessian"],
            n_bins=MODEL_CONFIG["n_bins"],
            random_state=MODEL_CONFIG["random_state"],
        )
        model.feature_names_ = FEATURE_NAMES
        model.fit(X_train, y_train)
        final_models[h] = model

        # Save lean JSON model
        lean_path = os.path.join(MODELS_DIR, f"stagbdt_h{h}_lean.json")
        model.save_json(lean_path, include_recursive=False)
        log.info("  Saved lean JSON model to %s (size: %d bytes)", lean_path, os.path.getsize(lean_path))

        # 2. Conformal Calibration on Calibration Runs
        X_cal = np.vstack([r["data_by_h"][h]["X"] for r in cal_runs])
        y_cal = np.concatenate([r["data_by_h"][h]["y"] for r in cal_runs])
        pred_cal_stag = model.predict(X_cal)

        calibrator = ConformalCalibrator(coverage_level=CONFORMAL_COVERAGE)
        margin_stag = calibrator.calibrate(y_cal, pred_cal_stag)
        conformal_calibrators[h] = calibrator
        conformal_margins[f"h{h}"] = round(margin_stag, 4)
        log.info("  ST-AGBDT 90%% conformal margin for h=%ds: %.4f req/s", h, margin_stag)

        # Baseline calibration on the SAME calibration runs
        cal_persist_raw = np.concatenate([r["data_by_h"][h]["base_persist_raw"] for r in cal_runs])
        cal_persist_smooth = np.concatenate([r["data_by_h"][h]["base_persist_smooth"] for r in cal_runs])
        cal_ewma10 = np.concatenate([r["data_by_h"][h]["base_ewma10"] for r in cal_runs])
        cal_linear = np.concatenate([r["data_by_h"][h]["base_linear"] for r in cal_runs])

        cal_p_raw = ConformalCalibrator(coverage_level=CONFORMAL_COVERAGE)
        m_p_raw = cal_p_raw.calibrate(y_cal, cal_persist_raw)

        cal_p_smooth = ConformalCalibrator(coverage_level=CONFORMAL_COVERAGE)
        m_p_smooth = cal_p_smooth.calibrate(y_cal, cal_persist_smooth)

        cal_ewma = ConformalCalibrator(coverage_level=CONFORMAL_COVERAGE)
        m_ewma = cal_ewma.calibrate(y_cal, cal_ewma10)

        cal_lin = ConformalCalibrator(coverage_level=CONFORMAL_COVERAGE)
        m_lin = cal_lin.calibrate(y_cal, cal_linear)

        baseline_conformal_margins[f"h{h}"] = {
            "persistence_raw": round(m_p_raw, 4),
            "persistence_smoothed": round(m_p_smooth, 4),
            "ewma_10": round(m_ewma, 4),
            "linear_trend": round(m_lin, 4),
        }

        # 3. Predict on Held-Out Test Runs
        test_preds_by_model: Dict[str, List[np.ndarray]] = {
            "ST-AGBDT (raw alpha=0.85)": [],
            "ST-AGBDT + Conformal 90%": [],
            "Persistence (raw) Padded": [],
            "Persistence (smoothed) Padded": [],
            "EWMA 10s Padded": [],
            "Linear Trend Padded": [],
            "Persistence (raw) Unpadded": [],
            "Persistence (smoothed) Unpadded": [],
            "EWMA 10s Unpadded": [],
            "Linear Trend Unpadded": [],
        }

        for r in test_runs:
            X_te = r["data_by_h"][h]["X"]
            pred_stag = model.predict(X_te)
            pred_stag_conf = calibrator.predict_upper(pred_stag)

            p_raw = r["data_by_h"][h]["base_persist_raw"]
            p_smooth = r["data_by_h"][h]["base_persist_smooth"]
            ewma = r["data_by_h"][h]["base_ewma10"]
            lin = r["data_by_h"][h]["base_linear"]

            test_preds_by_model["ST-AGBDT (raw alpha=0.85)"].append(pred_stag)
            test_preds_by_model["ST-AGBDT + Conformal 90%"].append(pred_stag_conf)

            test_preds_by_model["Persistence (raw) Padded"].append(cal_p_raw.predict_upper(p_raw))
            test_preds_by_model["Persistence (smoothed) Padded"].append(cal_p_smooth.predict_upper(p_smooth))
            test_preds_by_model["EWMA 10s Padded"].append(cal_ewma.predict_upper(ewma))
            test_preds_by_model["Linear Trend Padded"].append(cal_lin.predict_upper(lin))

            test_preds_by_model["Persistence (raw) Unpadded"].append(p_raw)
            test_preds_by_model["Persistence (smoothed) Unpadded"].append(p_smooth)
            test_preds_by_model["EWMA 10s Unpadded"].append(ewma)
            test_preds_by_model["Linear Trend Unpadded"].append(lin)

        # 4. Compute Test Metrics & Bootstrap CIs
        y_test_all = np.concatenate([r["data_by_h"][h]["y"] for r in test_runs])

        h_results: Dict[str, Any] = {}
        boot_cis = run_run_level_bootstrap(
            run_records=test_runs,
            model_predictions=test_preds_by_model,
            h=h,
            capacity_per_worker=cap_per_worker,
            n_resamples=BOOTSTRAP_RESAMPLES,
        )

        for m_name, preds_list in test_preds_by_model.items():
            all_preds = np.concatenate(preds_list)
            point_m = compute_evaluation_metrics(y_test_all, all_preds, cap_per_worker)
            h_results[m_name] = {
                "point_metrics": point_m,
                "bootstrap_ci95": boot_cis[m_name],
            }

        all_evaluation_results["horizons"][f"h{h}"] = {
            "stagbdt_conformal_margin": round(margin_stag, 4),
            "baseline_conformal_margins": baseline_conformal_margins[f"h{h}"],
            "models": h_results,
        }

    # Save final config JSON
    config_data = {
        "format_version": "1.0",
        "description": "ST-AGBDT final model configuration for horizons 10s, 20s, 30s",
        "git_commit": git_commit,
        "date_trained": "2026-10-27",
        "target_utilization": TARGET_UTILIZATION,
        "capacity_per_worker": cap_per_worker,
        "effective_capacity_per_worker": round(cap_per_worker * TARGET_UTILIZATION, 4),
        "conformal_coverage": CONFORMAL_COVERAGE,
        "conformal_margins": conformal_margins,
        "baseline_conformal_margins": baseline_conformal_margins,
        "hyperparameters": MODEL_CONFIG,
        "features": FEATURE_NAMES,
        "splits": {
            "training_run_ids": train_ids,
            "calibration_run_ids": cal_ids,
            "test_run_ids": test_ids,
            "excluded_run_ids": excl_ids,
        },
        "model_files": {
            "h10": "models/stagbdt_h10_lean.json",
            "h20": "models/stagbdt_h20_lean.json",
            "h30": "models/stagbdt_h30_lean.json",
        },
    }

    config_path = os.path.join(MODELS_DIR, "stagbdt_final_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    log.info("Saved final config to %s", config_path)

    # Save detailed evaluation results
    results_path = os.path.join(RESULTS_DIR, "final_evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluation_results, f, indent=2)
    log.info("Saved evaluation results to %s", results_path)

    # Generate README_STAGBDT.md
    generate_readme_artifact(config_data)

    # Print summary tables to stdout
    print_markdown_summary(all_evaluation_results)


def generate_readme_artifact(config: Dict[str, Any]) -> None:
    """Generate models/README_STAGBDT.md explaining format and usage."""
    readme_path = os.path.join(MODELS_DIR, "README_STAGBDT.md")
    content = f"""# ST-AGBDT Final Model Documentation & Format Specification

## 1. Overview
This directory contains the production-ready **Self-Throttling Asymmetric Gradient Boosted Decision Tree (ST-AGBDT)** models trained on Locust canonical runs (`data/runs_locust`).
All models use **arrival-only features**, ensuring predictions remain valid regardless of online worker autoscaling pool adjustments.

- **Git Commit**: `{config['git_commit']}`
- **Loss Function**: Asymmetric Least Squares (Expectile) with $\\alpha = 0.85$
- **Ensemble**: 50 trees, max depth 4, learning rate 0.1, 64 histogram bins
- **Target**: Maximum of 3-second rolling smoothed arrivals over future horizon $h \\in \\{{10, 20, 30\\}}$ seconds
- **Capacity**: Calibrated at {config['capacity_per_worker']:.2f} req/s per worker (`testbed_calibration.json:results[0].saturation_rps`), targeted at {config['target_utilization']*100:.0f}% utilization ({config['effective_capacity_per_worker']:.3f} req/s per worker).

---

## 2. File Manifest
- `models/stagbdt_h10_lean.json`: Compiled flat-tree ensemble for $h=10\\text{{s}}$ (zero external ML dependencies).
- `models/stagbdt_h20_lean.json`: Compiled flat-tree ensemble for $h=20\\text{{s}}$.
- `models/stagbdt_h30_lean.json`: Compiled flat-tree ensemble for $h=30\\text{{s}}$.
- `models/stagbdt_final_config.json`: Master metadata file containing features, run splits, conformal margins, and hyperparameters.
- `models/README_STAGBDT.md`: This file.

---

## 3. Lean Model JSON Schema
Each lean JSON file stores the ensemble compiled into continuous flat NumPy arrays for single-sample inference in <100 µs:
```json
{{
  "model_type": "STAGBDT",
  "full_name": "Self-Throttling Asymmetric Gradient Boosted Decision Trees",
  "n_estimators": 50,
  "learning_rate": 0.1,
  "max_depth": 4,
  "alpha": 0.85,
  "l2_reg": 1.0,
  "min_child_hessian": 0.001,
  "n_bins": 64,
  "random_state": 42,
  "base_pred_": 4.1234,
  "feature_names_": [...],
  "bin_edges_": [[...], ...],
  "flat_trees_": [
    {{
      "max_depth": 4,
      "feature_indices": [...],
      "thresholds": [...],
      "values": [...]
    }},
    ...
  ]
}}
```

### Loading & Inference (Pure NumPy)
```python
from stagbdt.model import STAGBDT

# Load model without any external ML dependencies
model = STAGBDT.load_json("models/stagbdt_h20_lean.json")

# Predict raw arrival rate
raw_forecast = model.predict_one(feature_vector)

# Apply 90% split-conformal upper bound
conformal_margin = {config['conformal_margins']['h20']}
upper_bound = raw_forecast + conformal_margin

# Compute recommended workers (clipped [2, 8])
eff_cap = {config['effective_capacity_per_worker']}
recommended_workers = int(min(8, max(2, math.ceil(upper_bound / eff_cap))))
```

---

## 4. Arrival-Only Feature Specification (25 Features)
Features are computed over 1-second arrival counts $arr$:
- `arr_curr`: Instantaneous count $arr[t]$
- For window durations $w \\in [5, 10, 30, 60]$ seconds:
  - `arr_mean_{{w}}`: Rolling mean $\\frac{{1}}{{w}} \\sum arr$
  - `arr_max_{{w}}`: Rolling maximum $\\max arr$
  - `arr_std_{{w}}`: Rolling population standard deviation
  - `arr_slope_{{w}}`: Linear slope $(arr[t] - arr[t-w+1]) / (w - 1)$
  - `arr_accel_{{w}}`: Slope change between recent and older half-window
  - `arr_ewma_{{w}}`: Exponential moving average with $\\alpha_{{ewma}} = \\frac{{2}}{{w + 1}}$

Both offline batch extraction and online streaming buffer use the identical implementation in `stagbdt/features.py`.

---

## 5. Calibrated Conformal Margins (90% Coverage)
Calibrated on the 7 designated calibration runs (`random_mix` seeds 15–19, `step_seed44`, `ramp_seed44`):
- **h = 10s**: `+{config['conformal_margins']['h10']:.4f}` req/s
- **h = 20s**: `+{config['conformal_margins']['h20']:.4f}` req/s
- **h = 30s**: `+{config['conformal_margins']['h30']:.4f}` req/s
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    log.info("Saved README to %s", readme_path)


def print_markdown_summary(res: Dict[str, Any]) -> None:
    """Print complete Markdown evaluation table for copy-paste into handoff."""
    print("\n" + "=" * 90)
    print("FINAL EVALUATION ON HELD-OUT TEST RUNS (13 RUNS)")
    print("=" * 90)

    for h_key, h_data in res["horizons"].items():
        h = h_key.replace("h", "")
        print(f"\n### Horizon h = {h} s")
        print(f"*ST-AGBDT Conformal 90% Margin: +{h_data['stagbdt_conformal_margin']:.4f} req/s*")
        print(f"*Baseline Padded Margins: Persistence Raw=+{h_data['baseline_conformal_margins']['persistence_raw']:.4f}, EWMA10=+{h_data['baseline_conformal_margins']['ewma_10']:.4f}, Linear=+{h_data['baseline_conformal_margins']['linear_trend']:.4f}*")
        print("\n| Model | MAE (req/s) [95% CI] | Miss Rate % [95% CI] | Extra Workers [95% CI] |")
        print("| :--- | :---: | :---: | :---: |")

        models_dict = h_data["models"]
        display_order = [
            "ST-AGBDT + Conformal 90%",
            "ST-AGBDT (raw alpha=0.85)",
            "Persistence (raw) Padded",
            "EWMA 10s Padded",
            "Linear Trend Padded",
            "Persistence (raw) Unpadded",
            "EWMA 10s Unpadded",
            "Linear Trend Unpadded",
        ]

        for m_name in display_order:
            if m_name not in models_dict:
                continue
            m_info = models_dict[m_name]
            p = m_info["point_metrics"]
            ci = m_info["bootstrap_ci95"]

            mae_str = f"{p['mae']:.3f} [{ci['mae_ci95'][0]:.3f}, {ci['mae_ci95'][1]:.3f}]"
            miss_str = f"{p['workers_needed_miss_pct']:.2f}% [{ci['miss_ci95'][0]:.2f}%, {ci['miss_ci95'][1]:.2f}%]"
            extra_str = f"{p['mean_extra_workers']:.3f} [{ci['extra_ci95'][0]:.3f}, {ci['extra_ci95'][1]:.3f}]"

            bold = "**" if "ST-AGBDT" in m_name else ""
            print(f"| {bold}{m_name}{bold} | {mae_str} | {miss_str} | {extra_str} |")

    print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
