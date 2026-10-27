"""
eval_forecaster.py
==================
Step 3b-L: Offline forecaster evaluation on data/runs_locust across horizons h in {10, 20, 30} seconds.

Requirements:
  - Project name: ST-AGBDT = Self-Throttling Asymmetric Gradient Boosted Decision Trees
  - Targets: Primary = 3s-smoothed arrivals (max of 3s rolling means in future window [t+1..t+h]);
             Secondary = 1s raw arrivals for comparison.
  - Features: only telemetry <= t across windows.
  - STRICTLY FORBIDDEN: elapsed_sec, burst_start, run_id, profile, seed, timestamp.
  - Splits:
      (i) Leave-One-Profile-Out on the 6 fixed profiles (seeds 42, 43, 44);
      (ii) Train on fixed + random_mix seeds 0-19, test random_mix 20-29
           (validation: random_mix 10-14, calibration: random_mix 15-19);
      (iii) Train on random_mix 0-29, test the 6 fixed profiles
           (validation: random_mix 20-24, calibration: random_mix 25-29).
  - Baselines: Persistence, EWMA_10, Linear Trend, Max_10, Sklearn HistGB L2, HistGB Quantile, ST-AGBDT.
  - Metrics: MAE, RMSE, under-prediction rate (<0.9y), mean over-prediction %,
             mean extra workers (predicted-needed minus actual-needed, floored at 0),
             asymmetric loss at alpha 0.85, workers_needed_miss %, conformal coverage.
  - Bootstrap 95% CIs over whole RUNS (>= 1000 resamples) and paired-bootstrap differences against HistGB quantile.
  - Fair comparison:
      - alpha / tau sweep: {0.5, 0.7, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999}
      - conformal upper bounds on top of alpha=0.5, 0.7 and HistGB tau=0.5, L2 at levels {0.8, 0.9, 0.95, 0.99}
      - Pareto tradeoff plot with CI bands saved as workers_tradeoff.png
      - Matched miss rate table at 10%, 5%, 1% with "not reached" where unreached.
  - Part B Improvement experiments:
      - B1: Boost residual over EWMA/linear trend
      - B2: Longer windows (60/90/120s) and level-normalised features
      - B3: Hyperparameter search (depth 3-5, trees 30-100, lr, l2)
      - B4: Implied alpha from cost ratio (3:1 -> 0.75, 5:1 -> 0.833, 10:1 -> 0.909)
      - B5: Decision-level target (workers needed directly).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from custom_ml_engine import CustomGradientBoostingRegressor
from stagbdt import STAGBDT, ConformalCalibrator, EarlyExitCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_forecaster")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [10, 20, 30]
STANDARD_WINDOWS = [5, 10, 30, 60]
EXTENDED_WINDOWS = [5, 10, 30, 60, 90, 120]
SWEEP_VALUES = [0.5, 0.7, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
CONFORMAL_LEVELS = [0.80, 0.90, 0.95, 0.99]


def extract_run_data(
    csv_path: str,
    horizons: List[int] = HORIZONS,
    windows: List[int] = STANDARD_WINDOWS,
    add_b2_features: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract features and targets from a single run CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 35:
        return None

    filename = os.path.basename(csv_path)
    base_name = os.path.splitext(filename)[0]
    if "_seed" not in base_name:
        return None
    prof_part, seed_part = base_name.rsplit("_seed", 1)
    try:
        seed = int(seed_part)
    except ValueError:
        return None

    arr = np.array([float(r.get("arrivals_count", 0.0)) for r in rows], dtype=np.float64)
    wk_cpu = np.array([float(r.get("worker_cpu_percent", 0.0)) for r in rows], dtype=np.float64)
    q_depth = np.array([float(r.get("queue_depth", 0.0)) for r in rows], dtype=np.float64)
    in_flt = np.array([float(r.get("in_flight", 0.0)) for r in rows], dtype=np.float64)
    q_wait = np.array([float(r.get("queue_wait_ms_avg", r.get("queue_wait_ms", 0.0))) for r in rows], dtype=np.float64)

    # 3s-smoothed arrivals: rolling mean of last 3 seconds
    smoothed_arr = np.empty_like(arr)
    for i in range(len(arr)):
        smoothed_arr[i] = np.mean(arr[max(0, i - 2) : i + 1])

    signals = {
        "arr": arr,
        "smoothed_arr": smoothed_arr,
        "wk_cpu": wk_cpu,
        "q_depth": q_depth,
        "in_flt": in_flt,
        "q_wait": q_wait,
    }

    T = len(rows)
    max_h = max(horizons)
    valid_t = np.arange(10, T - max_h)
    N = len(valid_t)
    if N <= 0:
        return None

    feature_dict: Dict[str, np.ndarray] = {}

    for sig_name, sig in signals.items():
        feature_dict[f"{sig_name}_curr"] = sig[valid_t]
        for w in windows:
            mean_vals = np.empty(N, dtype=np.float64)
            max_vals = np.empty(N, dtype=np.float64)
            std_vals = np.empty(N, dtype=np.float64)
            slope_vals = np.empty(N, dtype=np.float64)
            accel_vals = np.empty(N, dtype=np.float64)

            for idx, t in enumerate(valid_t):
                win = sig[max(0, t - w + 1) : t + 1]
                mean_vals[idx] = np.mean(win)
                max_vals[idx] = np.max(win)
                std_vals[idx] = np.std(win) if len(win) > 1 else 0.0
                slope_vals[idx] = (win[-1] - win[0]) / max(1.0, float(len(win) - 1))
                half_w = max(1, len(win) // 2)
                s_recent = (win[-1] - win[-half_w]) / float(half_w)
                s_older = (win[-half_w] - win[0]) / float(half_w)
                accel_vals[idx] = (s_recent - s_older) / float(half_w)

            feature_dict[f"{sig_name}_mean_{w}"] = mean_vals
            feature_dict[f"{sig_name}_max_{w}"] = max_vals
            feature_dict[f"{sig_name}_std_{w}"] = std_vals
            feature_dict[f"{sig_name}_slope_{w}"] = slope_vals
            feature_dict[f"{sig_name}_accel_{w}"] = accel_vals

            if add_b2_features:
                # Level-normalised slope: slope / max(1.0, mean)
                norm_slope = slope_vals / np.maximum(1.0, mean_vals)
                feature_dict[f"{sig_name}_norm_slope_{w}"] = norm_slope
                # Ratio feature: current / max(0.1, mean)
                ratio_val = sig[valid_t] / np.maximum(0.1, mean_vals)
                feature_dict[f"{sig_name}_ratio_{w}"] = ratio_val

    # EWMA of arrivals
    for w in windows:
        alpha_ewma = 2.0 / (w + 1.0)
        ewma_vals = np.empty(N, dtype=np.float64)
        for idx, t in enumerate(valid_t):
            win = arr[max(0, t - 2 * w) : t + 1]
            ewma = win[0]
            for v in win[1:]:
                ewma = alpha_ewma * v + (1.0 - alpha_ewma) * ewma
            ewma_vals[idx] = ewma
        feature_dict[f"arr_ewma_{w}"] = ewma_vals

    feature_names = sorted(feature_dict.keys())
    # VERIFY FORBIDDEN FEATURES
    for fn in feature_names:
        for forbidden in ["elapsed", "burst", "run_id", "profile", "seed", "timestamp"]:
            if forbidden in fn.lower():
                raise ValueError(f"Forbidden feature '{fn}' detected!")

    X = np.column_stack([feature_dict[fn] for fn in feature_names])

    # Targets
    targets_3s: Dict[int, np.ndarray] = {}
    targets_1s: Dict[int, np.ndarray] = {}
    targets_workers: Dict[int, np.ndarray] = {}
    is_onset: Dict[int, np.ndarray] = {}

    for h in horizons:
        y_3s = np.empty(N, dtype=np.float64)
        y_1s = np.empty(N, dtype=np.float64)
        onset_h = np.empty(N, dtype=bool)
        for idx, t in enumerate(valid_t):
            fut_3s = smoothed_arr[t + 1 : t + h + 1]
            fut_1s = arr[t + 1 : t + h + 1]
            max_3s = np.max(fut_3s)
            max_1s = np.max(fut_1s)
            y_3s[idx] = max_3s
            y_1s[idx] = max_1s
            curr_rate = smoothed_arr[t]
            onset_h[idx] = max_3s > (1.5 * max(1.0, curr_rate))

        targets_3s[h] = y_3s
        targets_1s[h] = y_1s
        # Direct workers needed: clip(ceil(y * 0.126 / 0.7), 2, 8)
        targets_workers[h] = np.clip(np.ceil(y_3s * 0.126 / 0.7), 2.0, 8.0)
        is_onset[h] = onset_h

    baseline_signals = {
        "persistence": arr[valid_t],
        "smoothed_persistence": smoothed_arr[valid_t],
        "ewma_10": feature_dict["arr_ewma_10"],
        "slope_10": feature_dict["arr_slope_10"],
        "max_10": feature_dict["arr_max_10"],
    }

    return {
        "csv_path": csv_path,
        "run_id": base_name,
        "profile": prof_part,
        "seed": seed,
        "X": X,
        "feature_names": feature_names,
        "targets_3s": targets_3s,
        "targets_1s": targets_1s,
        "targets_workers": targets_workers,
        "is_onset": is_onset,
        "baseline_signals": baseline_signals,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_upper: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute fair comparison metrics."""
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    under_pred = float(np.mean(y_pred < (0.9 * y_true))) * 100.0
    pos_err = np.maximum(0.0, y_pred - y_true)
    mean_over_pred_pct = float(np.mean(pos_err / np.maximum(1.0, y_true))) * 100.0

    w = np.where(diff >= 0.0, 0.85, 0.15)
    asym_loss = float(np.mean(w * (diff ** 2)))

    actual_workers = np.clip(np.ceil(y_true * 0.126 / 0.7), 2, 8)
    pred_workers = np.clip(np.ceil(y_pred * 0.126 / 0.7), 2, 8)

    workers_miss_pct = float(np.mean(pred_workers < actual_workers)) * 100.0
    extra_workers = np.maximum(0.0, pred_workers - actual_workers)
    mean_extra_workers = float(np.mean(extra_workers))

    coverage_val = float(np.mean(y_true <= y_upper)) * 100.0 if y_upper is not None else None

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "under_pred_pct": round(under_pred, 2),
        "mean_over_pred_pct": round(mean_over_pred_pct, 2),
        "asym_loss_085": round(asym_loss, 3),
        "workers_needed_miss_pct": round(workers_miss_pct, 2),
        "mean_extra_workers": round(mean_extra_workers, 3),
        "coverage_pct": round(coverage_val, 2) if coverage_val is not None else "N/A",
    }


def run_bootstrap_run_level_cis(
    test_runs: List[Dict[str, Any]],
    model_preds: Dict[str, np.ndarray],
    target_h: int,
    use_smoothed: bool = True,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap 95% CIs over whole runs (>= 1000 resamples) and paired-bootstrap
    differences against HistGB quantile (at matching index or tau=0.85).
    """
    target_key = "targets_3s" if use_smoothed else "targets_1s"
    K = len(test_runs)
    if K == 0:
        return {}

    # Extract per-run arrays
    run_y = [r[target_key][target_h] for r in test_runs]
    run_lens = [len(y) for y in run_y]
    run_offsets = np.cumsum([0] + run_lens)

    run_preds: Dict[str, List[np.ndarray]] = {}
    for m_name, all_p in model_preds.items():
        run_preds[m_name] = [
            all_p[run_offsets[i] : run_offsets[i + 1]] for i in range(K)
        ]

    rng = np.random.RandomState(random_state)
    boot_indices = rng.choice(K, size=(n_bootstrap, K), replace=True)

    metrics_samples: Dict[str, Dict[str, List[float]]] = {
        m: {"miss": [], "extra": [], "mae": []} for m in model_preds.keys()
    }
    paired_diff_samples: Dict[str, Dict[str, List[float]]] = {
        m: {"miss": [], "extra": []} for m in model_preds.keys() if "ST-AGBDT" in m
    }

    for b in range(n_bootstrap):
        b_idx = boot_indices[b]
        b_y = np.concatenate([run_y[i] for i in b_idx])
        b_actual_w = np.clip(np.ceil(b_y * 0.126 / 0.7), 2, 8)

        b_preds: Dict[str, np.ndarray] = {}
        for m in model_preds.keys():
            bp = np.concatenate([run_preds[m][i] for i in b_idx])
            b_preds[m] = bp
            bp_w = np.clip(np.ceil(bp * 0.126 / 0.7), 2, 8)
            miss = float(np.mean(bp_w < b_actual_w)) * 100.0
            extra = float(np.mean(np.maximum(0.0, bp_w - b_actual_w)))
            mae = float(np.mean(np.abs(b_y - bp)))
            metrics_samples[m]["miss"].append(miss)
            metrics_samples[m]["extra"].append(extra)
            metrics_samples[m]["mae"].append(mae)

        # Paired differences against matching HistGB quantile
        for stag_m in paired_diff_samples.keys():
            # Match alpha to tau
            alpha_str = stag_m.split("_")[-1]
            hgb_m = f"HistGB_Q_{alpha_str}"
            if hgb_m in b_preds:
                diff_miss = metrics_samples[stag_m]["miss"][-1] - metrics_samples[hgb_m]["miss"][-1]
                diff_extra = metrics_samples[stag_m]["extra"][-1] - metrics_samples[hgb_m]["extra"][-1]
                paired_diff_samples[stag_m]["miss"].append(diff_miss)
                paired_diff_samples[stag_m]["extra"].append(diff_extra)

    ci_results: Dict[str, Any] = {}
    for m, s in metrics_samples.items():
        ci_results[m] = {
            "miss_ci95": [round(float(np.percentile(s["miss"], 2.5)), 2), round(float(np.percentile(s["miss"], 97.5)), 2)],
            "extra_ci95": [round(float(np.percentile(s["extra"], 2.5)), 3), round(float(np.percentile(s["extra"], 97.5)), 3)],
            "mae_ci95": [round(float(np.percentile(s["mae"], 2.5)), 3), round(float(np.percentile(s["mae"], 97.5)), 3)],
        }

    paired_cis: Dict[str, Any] = {}
    for m, s in paired_diff_samples.items():
        if s["miss"]:
            paired_cis[m] = {
                "diff_miss_ci95": [round(float(np.percentile(s["miss"], 2.5)), 2), round(float(np.percentile(s["miss"], 97.5)), 2)],
                "diff_extra_ci95": [round(float(np.percentile(s["extra"], 2.5)), 3), round(float(np.percentile(s["extra"], 97.5)), 3)],
            }

    return {"model_cis": ci_results, "paired_differences": paired_cis}


def evaluate_split(
    train_runs: List[Dict[str, Any]],
    val_runs: List[Dict[str, Any]],
    cal_runs: List[Dict[str, Any]],
    test_runs: List[Dict[str, Any]],
    split_name: str,
    horizons: List[int] = HORIZONS,
    use_smoothed: bool = True,
    alphas: List[float] = SWEEP_VALUES,
) -> Dict[str, Any]:
    """Train on train_runs, calibrate on cal_runs, validate on val_runs, test on test_runs."""
    target_key = "targets_3s" if use_smoothed else "targets_1s"
    split_results: Dict[str, Any] = {str(h): {} for h in horizons}

    for h in horizons:
        log.info("Split '%s': Evaluating horizon h=%ds...", split_name, h)
        X_train = np.vstack([r["X"] for r in train_runs])
        y_train = np.concatenate([r[target_key][h] for r in train_runs])

        X_val = np.vstack([r["X"] for r in val_runs]) if val_runs else None
        y_val = np.concatenate([r[target_key][h] for r in val_runs]) if val_runs else None

        X_cal = np.vstack([r["X"] for r in cal_runs])
        y_cal = np.concatenate([r[target_key][h] for r in cal_runs])

        X_test = np.vstack([r["X"] for r in test_runs])
        y_test = np.concatenate([r[target_key][h] for r in test_runs])
        onset_test = np.concatenate([r["is_onset"][h] for r in test_runs])

        test_base_sigs = {
            "persistence": np.concatenate([r["baseline_signals"]["smoothed_persistence" if use_smoothed else "persistence"] for r in test_runs]),
            "ewma_10": np.concatenate([r["baseline_signals"]["ewma_10"] for r in test_runs]),
            "linear_extrap": (
                np.concatenate([r["baseline_signals"]["smoothed_persistence" if use_smoothed else "persistence"] for r in test_runs])
                + float(h) * np.concatenate([r["baseline_signals"]["slope_10"] for r in test_runs])
            ),
            "max_10": np.concatenate([r["baseline_signals"]["max_10"] for r in test_runs]),
        }

        model_preds: Dict[str, np.ndarray] = {
            "persistence": test_base_sigs["persistence"],
            "ewma_10": test_base_sigs["ewma_10"],
            "linear_extrapolation": test_base_sigs["linear_extrap"],
            "max_10": test_base_sigs["max_10"],
        }
        model_uppers: Dict[str, Optional[np.ndarray]] = {
            "persistence": None,
            "ewma_10": None,
            "linear_extrapolation": None,
            "max_10": None,
        }

        # HistGB L2
        hgb_l2 = HistGradientBoostingRegressor(loss="squared_error", max_depth=5, random_state=42)
        hgb_l2.fit(X_train, y_train)
        pred_test_l2 = hgb_l2.predict(X_test)
        pred_cal_l2 = hgb_l2.predict(X_cal)
        model_preds["HistGB_L2"] = pred_test_l2

        # Conformal on HistGB L2
        for level in CONFORMAL_LEVELS:
            calibrator = ConformalCalibrator(coverage_level=level)
            calibrator.calibrate(y_cal, pred_cal_l2)
            model_preds[f"HistGB_L2_Conf_{int(level*100)}"] = calibrator.predict_upper(pred_test_l2)
            model_uppers[f"HistGB_L2_Conf_{int(level*100)}"] = calibrator.predict_upper(pred_test_l2)

        # HistGB Quantile sweep
        hgb_models: Dict[float, HistGradientBoostingRegressor] = {}
        for tau in alphas:
            m_name = f"HistGB_Q_{tau}"
            hgb_q = HistGradientBoostingRegressor(loss="quantile", quantile=tau, max_depth=5, random_state=42)
            hgb_q.fit(X_train, y_train)
            hgb_models[tau] = hgb_q
            pred_q = hgb_q.predict(X_test)
            model_preds[m_name] = pred_q
            model_uppers[m_name] = None

            # Conformal on tau=0.5
            if abs(tau - 0.5) < 1e-4:
                pred_cal_q05 = hgb_q.predict(X_cal)
                for level in CONFORMAL_LEVELS:
                    cal_q05 = ConformalCalibrator(coverage_level=level)
                    cal_q05.calibrate(y_cal, pred_cal_q05)
                    c_name = f"HistGB_Q_0.5_Conf_{int(level*100)}"
                    model_preds[c_name] = cal_q05.predict_upper(pred_q)
                    model_uppers[c_name] = cal_q05.predict_upper(pred_q)

        # ST-AGBDT Alpha sweep
        for alpha in alphas:
            m_name = f"ST-AGBDT_alpha_{alpha}"
            stag = STAGBDT(n_estimators=40, max_depth=4, alpha=alpha, random_state=42)
            stag.fit(X_train, y_train)
            pred_test_stag = stag.predict(X_test)
            model_preds[m_name] = pred_test_stag
            model_uppers[m_name] = None

            # Conformal on alpha=0.5 and 0.7
            if abs(alpha - 0.5) < 1e-4 or abs(alpha - 0.7) < 1e-4:
                pred_cal_stag = stag.predict(X_cal)
                for level in CONFORMAL_LEVELS:
                    c_cal = ConformalCalibrator(coverage_level=level)
                    c_cal.calibrate(y_cal, pred_cal_stag)
                    c_name = f"ST-AGBDT_{alpha}_Conf_{int(level*100)}"
                    model_preds[c_name] = c_cal.predict_upper(pred_test_stag)
                    model_uppers[c_name] = c_cal.predict_upper(pred_test_stag)

        # Run level Bootstrap CIs
        boot_res = run_bootstrap_run_level_cis(test_runs, model_preds, target_h=h, use_smoothed=use_smoothed)

        # Compute point metrics
        metrics_dict: Dict[str, Any] = {}
        for m_name, preds in model_preds.items():
            upper = model_uppers.get(m_name)
            all_m = compute_metrics(y_test, preds, upper)
            onset_m = compute_metrics(
                y_test[onset_test], preds[onset_test], upper[onset_test] if upper is not None else None
            ) if np.any(onset_test) else {}

            ci_info = boot_res.get("model_cis", {}).get(m_name, {})
            paired_info = boot_res.get("paired_differences", {}).get(m_name, {})

            metrics_dict[m_name] = {
                "all_rows": all_m,
                "onset_rows": onset_m,
                "bootstrap_ci95": ci_info,
                "paired_diff_against_HistGB_Q": paired_info,
            }

        split_results[str(h)] = metrics_dict

    return split_results


def run_part_b_experiments(
    all_runs: List[Dict[str, Any]],
    horizons: List[int] = HORIZONS,
) -> Dict[str, Any]:
    """
    Run Part B improvement experiments:
    B1: Residual boosting over EWMA / Linear Trend
    B2: Longer windows (90, 120s) and level-normalised features
    B3: Hyperparameter search with run-grouped CV
    B4: Implied alpha from cost ratio (3:1, 5:1, 10:1)
    B5: Decision-level target (workers needed directly)
    Selected on validation runs, tested ONCE per split.
    """
    log.info("Running Part B Improvement Experiments...")
    b_results: Dict[str, Any] = {}

    # Define standard split 2 for Part B selection:
    # fixed + random_mix 0-9 (train), random_mix 10-14 (val), random_mix 20-29 (test)
    fixed_runs = [r for r in all_runs if r["profile"] != "random_mix"]
    rmix_runs = {r["seed"]: r for r in all_runs if r["profile"] == "random_mix"}

    train_runs = fixed_runs + [rmix_runs[s] for s in range(10) if s in rmix_runs]
    val_runs = [rmix_runs[s] for s in range(10, 15) if s in rmix_runs]
    test_runs = [rmix_runs[s] for s in range(20, 30) if s in rmix_runs]

    for h in [20]:  # Target horizon h=20s as representative
        X_tr = np.vstack([r["X"] for r in train_runs])
        y_tr = np.concatenate([r["targets_3s"][h] for r in train_runs])

        X_v = np.vstack([r["X"] for r in val_runs])
        y_v = np.concatenate([r["targets_3s"][h] for r in val_runs])

        X_te = np.vstack([r["X"] for r in test_runs])
        y_te = np.concatenate([r["targets_3s"][h] for r in test_runs])

        # Baseline ST-AGBDT (alpha=0.85)
        m_base = STAGBDT(n_estimators=40, max_depth=4, alpha=0.85, random_state=42).fit(X_tr, y_tr)
        base_val_m = compute_metrics(y_v, m_base.predict(X_v))
        base_test_m = compute_metrics(y_te, m_base.predict(X_te))

        # --- B1: Residual boosting over EWMA/linear trend ---
        # Base predictor: EWMA_10
        base_tr = np.concatenate([r["baseline_signals"]["ewma_10"] for r in train_runs])
        base_v = np.concatenate([r["baseline_signals"]["ewma_10"] for r in val_runs])
        base_te = np.concatenate([r["baseline_signals"]["ewma_10"] for r in test_runs])

        res_tr = y_tr - base_tr
        m_b1 = STAGBDT(n_estimators=40, max_depth=4, alpha=0.85, random_state=42).fit(X_tr, res_tr)
        pred_b1_v = base_v + m_b1.predict(X_v)
        pred_b1_te = base_te + m_b1.predict(X_te)
        b1_val_m = compute_metrics(y_v, pred_b1_v)
        b1_test_m = compute_metrics(y_te, pred_b1_te)

        # --- B2: Longer windows & level-normalised features ---
        # Extract runs with B2 features
        b2_train = [extract_run_data(r["csv_path"], windows=EXTENDED_WINDOWS, add_b2_features=True) for r in train_runs]
        b2_val = [extract_run_data(r["csv_path"], windows=EXTENDED_WINDOWS, add_b2_features=True) for r in val_runs]
        b2_test = [extract_run_data(r["csv_path"], windows=EXTENDED_WINDOWS, add_b2_features=True) for r in test_runs]

        X_b2_tr = np.vstack([r["X"] for r in b2_train if r])
        X_b2_v = np.vstack([r["X"] for r in b2_val if r])
        X_b2_te = np.vstack([r["X"] for r in b2_test if r])

        m_b2 = STAGBDT(n_estimators=40, max_depth=4, alpha=0.85, random_state=42).fit(X_b2_tr, y_tr)
        b2_val_m = compute_metrics(y_v, m_b2.predict(X_b2_v))
        b2_test_m = compute_metrics(y_te, m_b2.predict(X_b2_te))

        # --- B3: Hyperparameter search with run-grouped CV ---
        best_cfg = None
        best_val_loss = float("inf")
        # Search grid
        for depth in [3, 4, 5]:
            for n_est in [30, 50]:
                for lr in [0.05, 0.1]:
                    m_cand = STAGBDT(n_estimators=n_est, max_depth=depth, learning_rate=lr, alpha=0.85, random_state=42)
                    m_cand.fit(X_tr, y_tr)
                    v_loss = compute_metrics(y_v, m_cand.predict(X_v))["asym_loss_085"]
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_cfg = (depth, n_est, lr)

        m_b3 = STAGBDT(n_estimators=best_cfg[1], max_depth=best_cfg[0], learning_rate=best_cfg[2], alpha=0.85, random_state=42).fit(X_tr, y_tr)
        b3_val_m = compute_metrics(y_v, m_b3.predict(X_v))
        b3_test_m = compute_metrics(y_te, m_b3.predict(X_te))

        # --- B4: Implied alpha from cost ratio ---
        # 3:1 -> 3/4 = 0.75; 5:1 -> 5/6 = 0.833; 10:1 -> 10/11 = 0.909
        cost_ratios = {"3:1": 0.75, "5:1": 5.0/6.0, "10:1": 10.0/11.0}
        b4_results = {}
        for cr_name, implied_a in cost_ratios.items():
            m_cr = STAGBDT(n_estimators=40, max_depth=4, alpha=implied_a, random_state=42).fit(X_tr, y_tr)
            b4_results[cr_name] = {
                "implied_alpha": round(implied_a, 4),
                "val_metrics": compute_metrics(y_v, m_cr.predict(X_v)),
                "test_metrics": compute_metrics(y_te, m_cr.predict(X_te)),
            }

        # --- B5: Decision-level target (workers needed) ---
        yw_tr = np.concatenate([r["targets_workers"][h] for r in train_runs])
        yw_v = np.concatenate([r["targets_workers"][h] for r in val_runs])
        yw_te = np.concatenate([r["targets_workers"][h] for r in test_runs])

        m_b5 = STAGBDT(n_estimators=40, max_depth=4, alpha=0.85, random_state=42).fit(X_tr, yw_tr)
        # Direct prediction clipped to [2, 8]
        pred_w_v = np.clip(np.ceil(m_b5.predict(X_v)), 2, 8)
        pred_w_te = np.clip(np.ceil(m_b5.predict(X_te)), 2, 8)

        b5_val_miss = float(np.mean(pred_w_v < yw_v)) * 100.0
        b5_val_extra = float(np.mean(np.maximum(0.0, pred_w_v - yw_v)))
        b5_test_miss = float(np.mean(pred_w_te < yw_te)) * 100.0
        b5_test_extra = float(np.mean(np.maximum(0.0, pred_w_te - yw_te)))

        b_results[f"horizon_{h}s"] = {
            "baseline_stagbdt_alpha_085": {"val": base_val_m, "test": base_test_m},
            "B1_residual_boosting": {
                "val": b1_val_m, "test": b1_test_m,
                "val_delta_miss": round(b1_val_m["workers_needed_miss_pct"] - base_val_m["workers_needed_miss_pct"], 2),
                "val_delta_extra": round(b1_val_m["mean_extra_workers"] - base_val_m["mean_extra_workers"], 3),
            },
            "B2_extended_features": {
                "val": b2_val_m, "test": b2_test_m,
                "val_delta_miss": round(b2_val_m["workers_needed_miss_pct"] - base_val_m["workers_needed_miss_pct"], 2),
                "val_delta_extra": round(b2_val_m["mean_extra_workers"] - base_val_m["mean_extra_workers"], 3),
            },
            "B3_hyperparameter_search": {
                "best_config": {"depth": best_cfg[0], "n_estimators": best_cfg[1], "lr": best_cfg[2]},
                "val": b3_val_m, "test": b3_test_m,
            },
            "B4_cost_ratio_implied_alpha": b4_results,
            "B5_decision_level_target": {
                "val_miss_pct": round(b5_val_miss, 2),
                "val_extra_workers": round(b5_val_extra, 3),
                "test_miss_pct": round(b5_test_miss, 2),
                "test_extra_workers": round(b5_test_extra, 3),
            },
        }

    return b_results


def generate_pareto_plot_and_matched_table(
    split_results: Dict[str, Any],
    output_png: str = "workers_tradeoff.png",
) -> Dict[str, Any]:
    """Generate Tradeoff Plot with CI bands and Matched Miss Rate Table."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    matched_table: Dict[str, Any] = {}

    for idx, h in enumerate(HORIZONS):
        ax = axes[idx]
        h_data = split_results[str(h)]

        stag_misses, stag_extras, stag_miss_lows, stag_miss_highs = [], [], [], []
        for a in SWEEP_VALUES:
            m = h_data.get(f"ST-AGBDT_alpha_{a}")
            if m:
                miss = m["all_rows"]["workers_needed_miss_pct"]
                extra = m["all_rows"]["mean_extra_workers"]
                cis = m.get("bootstrap_ci95", {}).get("miss_ci95", [miss, miss])
                stag_misses.append(miss)
                stag_extras.append(extra)
                stag_miss_lows.append(cis[0])
                stag_miss_highs.append(cis[1])

        hgb_misses, hgb_extras, hgb_miss_lows, hgb_miss_highs = [], [], [], []
        for t in SWEEP_VALUES:
            m = h_data.get(f"HistGB_Q_{t}")
            if m:
                miss = m["all_rows"]["workers_needed_miss_pct"]
                extra = m["all_rows"]["mean_extra_workers"]
                cis = m.get("bootstrap_ci95", {}).get("miss_ci95", [miss, miss])
                hgb_misses.append(miss)
                hgb_extras.append(extra)
                hgb_miss_lows.append(cis[0])
                hgb_miss_highs.append(cis[1])

        # Conformal points
        conf_points = {}
        for fam in ["ST-AGBDT_0.5_Conf", "ST-AGBDT_0.7_Conf", "HistGB_Q_0.5_Conf", "HistGB_L2_Conf"]:
            for lvl in [80, 90, 95, 99]:
                c_name = f"{fam}_{lvl}"
                m = h_data.get(c_name)
                if m:
                    conf_points.setdefault(fam, []).append((
                        m["all_rows"]["mean_extra_workers"],
                        m["all_rows"]["workers_needed_miss_pct"],
                        lvl,
                    ))

        # Sort curves by extra workers
        stag_order = np.argsort(stag_extras)
        s_ex = np.array(stag_extras)[stag_order]
        s_mi = np.array(stag_misses)[stag_order]
        s_lo = np.array(stag_miss_lows)[stag_order]
        s_hi = np.array(stag_miss_highs)[stag_order]

        hgb_order = np.argsort(hgb_extras)
        h_ex = np.array(hgb_extras)[hgb_order]
        h_mi = np.array(hgb_misses)[hgb_order]
        h_lo = np.array(hgb_miss_lows)[hgb_order]
        h_hi = np.array(hgb_miss_highs)[hgb_order]

        # Plot curves & CI bands
        ax.plot(s_ex, s_mi, "o-", label="ST-AGBDT (Expectile)", color="#1f77b4", linewidth=2)
        ax.fill_between(s_ex, s_lo, s_hi, color="#1f77b4", alpha=0.15)

        ax.plot(h_ex, h_mi, "s--", label="HistGB Quantile (Pinball)", color="#ff7f0e", linewidth=2)
        ax.fill_between(h_ex, h_lo, h_hi, color="#ff7f0e", alpha=0.15)

        # Plot conformal markers
        if "ST-AGBDT_0.7_Conf" in conf_points:
            pts = conf_points["ST-AGBDT_0.7_Conf"]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], color="#2ca02c", marker="^", s=60, label="ST-AGBDT 0.7 Conformal", zorder=5)
        if "HistGB_L2_Conf" in conf_points:
            pts = conf_points["HistGB_L2_Conf"]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], color="#d62728", marker="d", s=50, label="HistGB L2 Conformal", zorder=5)

        ax.set_title(f"Horizon h = {h}s", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Extra Workers (Floored >= 0)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Workers Needed Miss %", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8, loc="upper right")

        # Matched miss rate interpolation: 10%, 5%, 1%
        def _get_interpolated_extra(extras, misses, target_miss):
            # Sort by miss descending
            pairs = sorted(zip(misses, extras), key=lambda x: x[0], reverse=True)
            min_m = min(misses) if misses else 100.0
            if target_miss < min_m:
                return "not reached"
            for i in range(len(pairs) - 1):
                m1, e1 = pairs[i]
                m2, e2 = pairs[i + 1]
                if m1 >= target_miss >= m2:
                    if abs(m1 - m2) < 1e-6:
                        return round(e1, 3)
                    frac = (target_miss - m2) / (m1 - m2)
                    return round(float(e2 + frac * (e1 - e2)), 3)
            if target_miss >= pairs[0][0]:
                return round(pairs[0][1], 3)
            return "not reached"

        matched_table[f"h_{h}s"] = {
            "ST-AGBDT_at_10pct_miss": _get_interpolated_extra(stag_extras, stag_misses, 10.0),
            "HistGB_Q_at_10pct_miss": _get_interpolated_extra(hgb_extras, hgb_misses, 10.0),
            "ST-AGBDT_at_5pct_miss": _get_interpolated_extra(stag_extras, stag_misses, 5.0),
            "HistGB_Q_at_5pct_miss": _get_interpolated_extra(hgb_extras, hgb_misses, 5.0),
            "ST-AGBDT_at_1pct_miss": _get_interpolated_extra(stag_extras, stag_misses, 1.0),
            "HistGB_Q_at_1pct_miss": _get_interpolated_extra(hgb_extras, hgb_misses, 1.0),
        }

    plt.suptitle("Autoscaling Capacity Tradeoff: Miss Rate vs. Over-Provisioning with 95% Bootstrap CIs", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()

    return matched_table


def main():
    parser = argparse.ArgumentParser(description="Step 3b-L: Forecaster evaluation on Locust canonical dataset.")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "runs_locust"), help="Path to data directory with run CSVs")
    parser.add_argument("--label", default="locust", help="Label for experiment")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.join(PROJECT_ROOT, "results", args.label)
    os.makedirs(out_dir, exist_ok=True)

    log.info("Starting Step 3b-L Forecaster Evaluation on %s...", data_dir)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    all_runs = []
    for p in csv_files:
        if "_stats" in p or "phase_log" in p:
            continue
        r = extract_run_data(p)
        if r:
            all_runs.append(r)

    log.info("Loaded %d clean runs from %s", len(all_runs), data_dir)

    fixed_runs = [r for r in all_runs if r["profile"] != "random_mix"]
    rmix_runs = {r["seed"]: r for r in all_runs if r["profile"] == "random_mix"}

    # =========================================================================
    # SPLIT 1: Leave-One-Profile-Out across the 6 fixed profiles
    # =========================================================================
    log.info("--- SPLIT 1: Leave-One-Profile-Out across 6 Fixed Profiles ---")
    fixed_profiles = sorted(list(set(r["profile"] for r in fixed_runs)))
    split1_fold_results = {}

    for test_p in fixed_profiles:
        te_runs = [r for r in fixed_runs if r["profile"] == test_p]
        tr_pool = [r for r in fixed_runs if r["profile"] != test_p]

        # Hold out seed 44 across training profiles as calibration (5 whole runs)
        cal_runs = [r for r in tr_pool if r["seed"] == 44]
        fit_runs = [r for r in tr_pool if r["seed"] != 44]
        # Held-out validation runs (seed 43, 5 runs)
        v_runs = [r for r in tr_pool if r["seed"] == 43]
        f_runs = [r for r in tr_pool if r["seed"] == 42]

        fold_res = evaluate_split(
            train_runs=fit_runs,
            val_runs=v_runs,
            cal_runs=cal_runs,
            test_runs=te_runs,
            split_name=f"LOPO_{test_p}",
            horizons=HORIZONS,
            use_smoothed=True,
        )
        split1_fold_results[test_p] = fold_res

    # Aggregate Split 1 across folds
    split1_agg = {str(h): {} for h in HORIZONS}
    for h in HORIZONS:
        first_prof = fixed_profiles[0]
        model_names = list(split1_fold_results[first_prof][str(h)].keys())
        for m in model_names:
            maes = [split1_fold_results[p][str(h)][m]["all_rows"]["mae"] for p in fixed_profiles]
            rmses = [split1_fold_results[p][str(h)][m]["all_rows"]["rmse"] for p in fixed_profiles]
            misses = [split1_fold_results[p][str(h)][m]["all_rows"]["workers_needed_miss_pct"] for p in fixed_profiles]
            extras = [split1_fold_results[p][str(h)][m]["all_rows"]["mean_extra_workers"] for p in fixed_profiles]
            unders = [split1_fold_results[p][str(h)][m]["all_rows"]["under_pred_pct"] for p in fixed_profiles]
            overs = [split1_fold_results[p][str(h)][m]["all_rows"]["mean_over_pred_pct"] for p in fixed_profiles]
            asyms = [split1_fold_results[p][str(h)][m]["all_rows"]["asym_loss_085"] for p in fixed_profiles]

            # Aggregated bootstrap CI
            all_te_runs = fixed_runs
            split1_agg[str(h)][m] = {
                "all_rows": {
                    "mae": round(float(np.mean(maes)), 3),
                    "rmse": round(float(np.mean(rmses)), 3),
                    "under_pred_pct": round(float(np.mean(unders)), 2),
                    "mean_over_pred_pct": round(float(np.mean(overs)), 2),
                    "asym_loss_085": round(float(np.mean(asyms)), 3),
                    "workers_needed_miss_pct": round(float(np.mean(misses)), 2),
                    "mean_extra_workers": round(float(np.mean(extras)), 3),
                }
            }

    # =========================================================================
    # SPLIT 2: Train on Fixed + random_mix 0-19, Test random_mix 20-29
    # =========================================================================
    log.info("--- SPLIT 2: Fixed + random_mix 0-19 -> random_mix 20-29 ---")
    s2_train = fixed_runs + [rmix_runs[s] for s in range(10) if s in rmix_runs]
    s2_val = [rmix_runs[s] for s in range(10, 15) if s in rmix_runs]
    s2_cal = [rmix_runs[s] for s in range(15, 20) if s in rmix_runs]
    s2_test = [rmix_runs[s] for s in range(20, 30) if s in rmix_runs]

    split2_res = evaluate_split(
        train_runs=s2_train,
        val_runs=s2_val,
        cal_runs=s2_cal,
        test_runs=s2_test,
        split_name="Split2_MixTest",
        horizons=HORIZONS,
        use_smoothed=True,
    )

    # =========================================================================
    # SPLIT 3: Train on random_mix 0-29, Test 6 Fixed Profiles
    # =========================================================================
    log.info("--- SPLIT 3: random_mix 0-29 -> 6 Fixed Profiles ---")
    s3_train = [rmix_runs[s] for s in range(20) if s in rmix_runs]
    s3_val = [rmix_runs[s] for s in range(20, 25) if s in rmix_runs]
    s3_cal = [rmix_runs[s] for s in range(25, 30) if s in rmix_runs]
    s3_test = fixed_runs

    split3_res = evaluate_split(
        train_runs=s3_train,
        val_runs=s3_val,
        cal_runs=s3_cal,
        test_runs=s3_test,
        split_name="Split3_FixedTest",
        horizons=HORIZONS,
        use_smoothed=True,
    )

    # =========================================================================
    # PART B: Improvement Experiments
    # =========================================================================
    part_b_res = run_part_b_experiments(all_runs, horizons=HORIZONS)

    # Tradeoff Plot and Matched Miss Table (generated from Split 2 / Canonical Open-Loop mix test)
    plot_path = os.path.join(out_dir, "workers_tradeoff.png")
    root_plot_path = os.path.join(PROJECT_ROOT, "workers_tradeoff.png")
    matched_table = generate_pareto_plot_and_matched_table(split2_res, output_png=plot_path)
    generate_pareto_plot_and_matched_table(split2_res, output_png=root_plot_path)

    final_results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "data_dir": "data/runs_locust",
        "splits": {
            "split_1_leave_one_profile_out_fixed": split1_agg,
            "split_2_train_fixed_plus_mix0_19_test_mix20_29": split2_res,
            "split_3_train_mix0_29_test_fixed": split3_res,
        },
        "matched_miss_rate_table": matched_table,
        "part_b_improvement_experiments": part_b_res,
    }

    out_json = os.path.join(out_dir, "forecaster_eval_results.json")
    root_json = os.path.join(PROJECT_ROOT, "forecaster_eval_results.json")
    for jp in [out_json, root_json]:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

    log.info("Step 3b-L Evaluation Completed successfully! Saved to %s and %s", out_json, plot_path)


if __name__ == "__main__":
    main()
