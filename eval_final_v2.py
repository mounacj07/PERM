"""
eval_final_v2.py
================
Step 3 additions:
  - Peak-hold baselines (30s and 60s), raw and conformal 90%.
  - Paired-bootstrap differences for RAW MAE (whole-run resampling).
  - Paired-bootstrap differences for extra workers at matched operating points
    (interpolated over conformal levels 0.50 - 0.99).
  - Empirical coverage of ST-AGBDT conformal 90% upper bound (overall and per profile).
Saves results/final_evaluation_results_v2.json and outputs regenerated tables.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stagbdt.features import (
    FEATURE_NAMES,
    extract_features_sequence,
)
from stagbdt.model import STAGBDT
from stagbdt.conformal import ConformalCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_v2")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CONFIG_FILE = os.path.join(MODELS_DIR, "stagbdt_final_config.json")
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "testbed_calibration.json")

HORIZONS = [10, 20, 30]
TARGET_UTILIZATION = 0.70
CONFORMAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
BOOTSTRAP_RESAMPLES = 1000


def load_capacity_per_worker() -> float:
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        d = json.load(f)
    for r in d.get("results", []):
        if r.get("n_workers") == 1:
            return float(r.get("saturation_rps", 7.89))
    return 7.89


def load_single_run(run_id: str, horizons: List[int] = HORIZONS) -> Dict[str, Any]:
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

    # Extract profile name
    prof = run_id.rsplit("_seed", 1)[0]

    data_by_h: Dict[int, Dict[str, np.ndarray]] = {}
    for h in horizons:
        valid_t = np.arange(10, T - h)
        X, fnames = extract_features_sequence(arr, valid_t=valid_t)
        N = len(valid_t)

        y_target = np.empty(N, dtype=np.float64)
        for idx, t in enumerate(valid_t):
            fut_3s = smoothed_arr[t + 1 : t + h + 1]
            y_target[idx] = np.max(fut_3s)

        idx_ewma10 = fnames.index("arr_ewma_10")
        idx_slope10 = fnames.index("arr_slope_10")
        idx_max30 = fnames.index("arr_max_30")
        idx_max60 = fnames.index("arr_max_60")

        data_by_h[h] = {
            "valid_t": valid_t,
            "X": X,
            "y": y_target,
            "persistence": arr[valid_t],
            "ewma_10": X[:, idx_ewma10],
            "linear_trend": arr[valid_t] + float(h) * X[:, idx_slope10],
            "peak_hold_30s": X[:, idx_max30],
            "peak_hold_60s": X[:, idx_max60],
        }

    return {
        "run_id": run_id,
        "profile": prof,
        "arr": arr,
        "smoothed_arr": smoothed_arr,
        "data_by_h": data_by_h,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_per_worker: float,
    utilization: float = TARGET_UTILIZATION,
) -> Dict[str, float]:
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    eff_cap = capacity_per_worker * utilization
    actual_workers = np.clip(np.ceil(y_true / eff_cap), 2.0, 8.0)
    pred_workers = np.clip(np.ceil(y_pred / eff_cap), 2.0, 8.0)

    miss_pct = float(np.mean(pred_workers < actual_workers)) * 100.0
    extra_workers = np.maximum(0.0, pred_workers - actual_workers)
    mean_extra = float(np.mean(extra_workers))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "workers_needed_miss_pct": round(miss_pct, 4),
        "mean_extra_workers": round(mean_extra, 4),
    }


def interpolate_frontier(
    misses: List[float],
    extras: List[float],
    target_miss: float,
) -> Optional[float]:
    """
    Given (miss, extra) pairs along a conformal frontier,
    interpolate the extra workers required to reach target_miss.
    Returns None if target_miss is lower than the lowest miss reached.
    """
    # Sort pairs by miss descending
    pairs = sorted(zip(misses, extras), key=lambda x: x[0], reverse=True)
    min_miss = min(misses) if misses else 100.0
    if target_miss < min_miss - 1e-5:
        return None  # Unreached

    # If target_miss >= maximum miss on curve, return extra of max miss
    if target_miss >= pairs[0][0]:
        return float(pairs[0][1])

    for i in range(len(pairs) - 1):
        m1, e1 = pairs[i]
        m2, e2 = pairs[i + 1]
        if m1 >= target_miss >= m2:
            if abs(m1 - m2) < 1e-6:
                return float(e1)
            frac = (target_miss - m2) / (m1 - m2)
            return float(e2 + frac * (e1 - e2))

    return float(pairs[-1][1])


def main():
    cap_per_worker = load_capacity_per_worker()
    eff_cap = cap_per_worker * TARGET_UTILIZATION
    log.info("Capacity per worker: %.2f rps, Effective: %.3f rps", cap_per_worker, eff_cap)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    cal_ids = config["splits"]["calibration_run_ids"]
    test_ids = config["splits"]["test_run_ids"]

    log.info("Loading %d calibration runs and %d test runs...", len(cal_ids), len(test_ids))
    cal_runs = [load_single_run(rid) for rid in cal_ids]
    test_runs = [load_single_run(rid) for rid in test_ids]

    results_v2: Dict[str, Any] = {
        "metadata": {
            "description": "PERM Step 3 additions: peak-hold baselines, paired differences, matched frontiers, empirical coverage",
            "capacity_per_worker": cap_per_worker,
            "target_utilization": TARGET_UTILIZATION,
            "effective_capacity_per_worker": round(eff_cap, 4),
            "conformal_levels": CONFORMAL_LEVELS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "test_run_ids": test_ids,
            "calibration_run_ids": cal_ids,
        },
        "horizons": {},
    }

    base_methods = [
        "Persistence",
        "EWMA 10s",
        "Linear Trend",
        "Peak-Hold 30s",
        "Peak-Hold 60s",
    ]
    all_methods = ["ST-AGBDT"] + base_methods

    for h in HORIZONS:
        log.info("==================================================================")
        log.info("Evaluating Horizon h = %d s", h)
        log.info("==================================================================")

        # Load ST-AGBDT model
        model_file = os.path.join(PROJECT_ROOT, config["model_files"][f"h{h}"])
        model = STAGBDT.load_json(model_file)

        # 1. Calibration phase for all methods across all conformal levels
        y_cal = np.concatenate([r["data_by_h"][h]["y"] for r in cal_runs])

        cal_preds: Dict[str, np.ndarray] = {
            "ST-AGBDT": model.predict(np.vstack([r["data_by_h"][h]["X"] for r in cal_runs])),
            "Persistence": np.concatenate([r["data_by_h"][h]["persistence"] for r in cal_runs]),
            "EWMA 10s": np.concatenate([r["data_by_h"][h]["ewma_10"] for r in cal_runs]),
            "Linear Trend": np.concatenate([r["data_by_h"][h]["linear_trend"] for r in cal_runs]),
            "Peak-Hold 30s": np.concatenate([r["data_by_h"][h]["peak_hold_30s"] for r in cal_runs]),
            "Peak-Hold 60s": np.concatenate([r["data_by_h"][h]["peak_hold_60s"] for r in cal_runs]),
        }

        # Calibrate margins across all levels
        conformal_margins: Dict[str, Dict[float, float]] = {m: {} for m in all_methods}
        for m in all_methods:
            s_cal = y_cal - cal_preds[m]
            n_cal = len(s_cal)
            sorted_s = np.sort(s_cal)
            for lvl in CONFORMAL_LEVELS:
                q_idx = int(np.ceil((n_cal + 1) * lvl)) - 1
                q_idx = min(n_cal - 1, max(0, q_idx))
                conformal_margins[m][lvl] = float(sorted_s[q_idx])

        # 2. Test phase on held-out runs
        test_data_by_run: List[Dict[str, Any]] = []
        for r in test_runs:
            d_h = r["data_by_h"][h]
            X_te = d_h["X"]
            preds = {
                "ST-AGBDT": model.predict(X_te),
                "Persistence": d_h["persistence"],
                "EWMA 10s": d_h["ewma_10"],
                "Linear Trend": d_h["linear_trend"],
                "Peak-Hold 30s": d_h["peak_hold_30s"],
                "Peak-Hold 60s": d_h["peak_hold_60s"],
            }
            test_data_by_run.append({
                "run_id": r["run_id"],
                "profile": r["profile"],
                "y": d_h["y"],
                "raw_preds": preds,
            })

        y_test_all = np.concatenate([td["y"] for td in test_data_by_run])

        # Point metrics at 90% conformal and raw
        point_evaluations: Dict[str, Any] = {}
        for m in all_methods:
            raw_p = np.concatenate([td["raw_preds"][m] for td in test_data_by_run])
            pad_90_p = raw_p + conformal_margins[m][0.90]

            point_evaluations[f"{m} (raw)"] = compute_metrics(y_test_all, raw_p, cap_per_worker)
            point_evaluations[f"{m} Padded 90%"] = compute_metrics(y_test_all, pad_90_p, cap_per_worker)

        # 3. Empirical coverage of ST-AGBDT 90% conformal upper bound
        stag_pred_all = np.concatenate([td["raw_preds"]["ST-AGBDT"] for td in test_data_by_run])
        stag_upper_all = stag_pred_all + conformal_margins["ST-AGBDT"][0.90]
        overall_coverage = float(np.mean(y_test_all <= stag_upper_all)) * 100.0

        profile_coverage: Dict[str, float] = {}
        unique_profs = sorted(list(set(td["profile"] for td in test_data_by_run)))
        for prof in unique_profs:
            y_prof = np.concatenate([td["y"] for td in test_data_by_run if td["profile"] == prof])
            p_prof = np.concatenate([td["raw_preds"]["ST-AGBDT"] for td in test_data_by_run if td["profile"] == prof])
            cov_prof = float(np.mean(y_prof <= (p_prof + conformal_margins["ST-AGBDT"][0.90]))) * 100.0
            profile_coverage[prof] = round(cov_prof, 2)

        # 4. Bootstrap Resampling (B = 1000 whole runs)
        rng = np.random.RandomState(42)
        n_test = len(test_data_by_run)

        # Store boots
        boot_raw_mae: Dict[str, List[float]] = {m: [] for m in all_methods}
        boot_raw_diff_mae: Dict[str, List[float]] = {m: [] for m in base_methods}

        # Frontier evaluation per method per bootstrap resample:
        # For each method, at each resample, compute the (miss, extra) curve over all CONFORMAL_LEVELS
        target_operating_misses = [10.0, 5.0, 1.0]
        # Also include the point achieved by ST-AGBDT 90% conformal
        stag_90_miss = point_evaluations["ST-AGBDT Padded 90%"]["workers_needed_miss_pct"]
        target_operating_misses.append(round(stag_90_miss, 2))
        target_operating_misses = sorted(list(set(target_operating_misses)), reverse=True)

        boot_matched_extra: Dict[str, Dict[float, List[Optional[float]]]] = {
            m: {om: [] for om in target_operating_misses} for m in all_methods
        }
        boot_matched_diff_extra: Dict[str, Dict[float, List[Optional[float]]]] = {
            bm: {om: [] for om in target_operating_misses} for bm in base_methods
        }

        for b in range(BOOTSTRAP_RESAMPLES):
            sampled_idx = rng.choice(n_test, size=n_test, replace=True)
            y_b = np.concatenate([test_data_by_run[i]["y"] for i in sampled_idx])

            # Raw predictions
            raw_b = {
                m: np.concatenate([test_data_by_run[i]["raw_preds"][m] for i in sampled_idx])
                for m in all_methods
            }

            # 1. Raw MAE
            mae_stag = float(np.mean(np.abs(y_b - raw_b["ST-AGBDT"])))
            boot_raw_mae["ST-AGBDT"].append(mae_stag)

            for bm in base_methods:
                mae_bm = float(np.mean(np.abs(y_b - raw_b[bm])))
                boot_raw_mae[bm].append(mae_bm)
                # Diff = ST-AGBDT - Baseline (negative means ST-AGBDT has lower MAE)
                boot_raw_diff_mae[bm].append(mae_stag - mae_bm)

            # 2. Conformal frontiers
            frontiers_b: Dict[str, Tuple[List[float], List[float]]] = {}
            for m in all_methods:
                m_misses, m_extras = [], []
                for lvl in CONFORMAL_LEVELS:
                    p_lvl = raw_b[m] + conformal_margins[m][lvl]
                    metrics_lvl = compute_metrics(y_b, p_lvl, cap_per_worker)
                    m_misses.append(metrics_lvl["workers_needed_miss_pct"])
                    m_extras.append(metrics_lvl["mean_extra_workers"])
                frontiers_b[m] = (m_misses, m_extras)

            # Interpolate extra workers at target operating misses
            for om in target_operating_misses:
                extra_stag = interpolate_frontier(frontiers_b["ST-AGBDT"][0], frontiers_b["ST-AGBDT"][1], om)
                boot_matched_extra["ST-AGBDT"][om].append(extra_stag)

                for bm in base_methods:
                    extra_bm = interpolate_frontier(frontiers_b[bm][0], frontiers_b[bm][1], om)
                    boot_matched_extra[bm][om].append(extra_bm)

                    if extra_stag is not None and extra_bm is not None:
                        # Diff = ST-AGBDT - Baseline (negative means ST-AGBDT uses fewer extra workers)
                        boot_matched_diff_extra[bm][om].append(extra_stag - extra_bm)
                    else:
                        boot_matched_diff_extra[bm][om].append(None)

        # Compile paired differences summary
        paired_raw_mae_summary: Dict[str, Any] = {}
        for bm in base_methods:
            diffs = boot_raw_diff_mae[bm]
            paired_raw_mae_summary[bm] = {
                "mean_diff": round(float(np.mean(diffs)), 4),
                "ci95": [
                    round(float(np.percentile(diffs, 2.5)), 4),
                    round(float(np.percentile(diffs, 97.5)), 4),
                ],
                "zero_in_ci": bool(np.percentile(diffs, 2.5) <= 0.0 <= np.percentile(diffs, 97.5)),
            }

        matched_frontier_summary: Dict[str, Any] = {}
        for om in target_operating_misses:
            om_key = f"{om:.2f}pct_miss"
            matched_frontier_summary[om_key] = {"target_miss_pct": om, "methods": {}}

            for m in all_methods:
                valid_extras = [v for v in boot_matched_extra[m][om] if v is not None]
                if valid_extras:
                    matched_frontier_summary[om_key]["methods"][m] = {
                        "extra_workers_mean": round(float(np.mean(valid_extras)), 4),
                        "extra_workers_ci95": [
                            round(float(np.percentile(valid_extras, 2.5)), 4),
                            round(float(np.percentile(valid_extras, 97.5)), 4),
                        ],
                        "reach_rate_pct": round(len(valid_extras) / BOOTSTRAP_RESAMPLES * 100.0, 1),
                    }
                else:
                    matched_frontier_summary[om_key]["methods"][m] = {
                        "status": "not reached",
                        "reach_rate_pct": 0.0,
                    }

            # Paired differences vs ST-AGBDT at this operating point
            matched_frontier_summary[om_key]["paired_diff_extra_vs_stagbdt"] = {}
            for bm in base_methods:
                valid_diffs = [v for v in boot_matched_diff_extra[bm][om] if v is not None]
                if valid_diffs:
                    matched_frontier_summary[om_key]["paired_diff_extra_vs_stagbdt"][bm] = {
                        "mean_diff_extra": round(float(np.mean(valid_diffs)), 4),
                        "ci95": [
                            round(float(np.percentile(valid_diffs, 2.5)), 4),
                            round(float(np.percentile(valid_diffs, 97.5)), 4),
                        ],
                        "zero_in_ci": bool(np.percentile(valid_diffs, 2.5) <= 0.0 <= np.percentile(valid_diffs, 97.5)),
                    }
                else:
                    matched_frontier_summary[om_key]["paired_diff_extra_vs_stagbdt"][bm] = {
                        "status": "not reached",
                    }

        results_v2["horizons"][f"h{h}"] = {
            "conformal_margins_90pct": {m: round(conformal_margins[m][0.90], 4) for m in all_methods},
            "point_evaluations": point_evaluations,
            "empirical_coverage_stagbdt_90pct": {
                "overall_pct": round(overall_coverage, 2),
                "by_profile_pct": profile_coverage,
            },
            "paired_bootstrap_raw_mae_diff": paired_raw_mae_summary,
            "matched_operating_frontiers": matched_frontier_summary,
        }

    # Save to results/final_evaluation_results_v2.json
    out_file = os.path.join(RESULTS_DIR, "final_evaluation_results_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_v2, f, indent=2)
    log.info("Saved results_v2 to %s", out_file)

    # Print summary tables
    print_v2_summary(results_v2)


def print_v2_summary(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("REGENERATED EVALUATION TABLES WITH PEAK-HOLD BASELINES & PAIRED DIFFERENCES")
    print("=" * 100)

    for h_key in ["h10", "h20", "h30"]:
        h_data = res["horizons"][h_key]
        h = h_key.replace("h", "")
        print(f"\n### Horizon h = {h} s")
        print(f"ST-AGBDT 90% Conformal Coverage: Overall={h_data['empirical_coverage_stagbdt_90pct']['overall_pct']:.2f}% | Profiles={h_data['empirical_coverage_stagbdt_90pct']['by_profile_pct']}")
        print(f"90% Margins: ST-AGBDT=+{h_data['conformal_margins_90pct']['ST-AGBDT']:.2f}, Pers=+{h_data['conformal_margins_90pct']['Persistence']:.2f}, EWMA=+{h_data['conformal_margins_90pct']['EWMA 10s']:.2f}, Peak30=+{h_data['conformal_margins_90pct']['Peak-Hold 30s']:.2f}, Peak60=+{h_data['conformal_margins_90pct']['Peak-Hold 60s']:.2f}")

        pts = h_data["point_evaluations"]
        print("\n| Model Variant | MAE (req/s) | Miss Rate % | Mean Extra Workers |")
        print("| :--- | :---: | :---: | :---: |")
        for m in [
            "ST-AGBDT Padded 90%", "ST-AGBDT (raw)",
            "Persistence Padded 90%", "Persistence (raw)",
            "EWMA 10s Padded 90%", "EWMA 10s (raw)",
            "Peak-Hold 30s Padded 90%", "Peak-Hold 30s (raw)",
            "Peak-Hold 60s Padded 90%", "Peak-Hold 60s (raw)",
            "Linear Trend Padded 90%", "Linear Trend (raw)",
        ]:
            if m in pts:
                p = pts[m]
                bold = "**" if "ST-AGBDT" in m else ""
                print(f"| {bold}{m}{bold} | {p['mae']:.3f} | {p['workers_needed_miss_pct']:.2f}% | {p['mean_extra_workers']:.3f} |")

        print("\n**Paired Differences vs ST-AGBDT (RAW MAE)** (negative = ST-AGBDT has lower MAE):")
        for bm, diff_info in h_data["paired_bootstrap_raw_mae_diff"].items():
            print(f"  - vs {bm:15}: Mean Diff = {diff_info['mean_diff']:+.3f} req/s [95% CI: {diff_info['ci95'][0]:+.3f}, {diff_info['ci95'][1]:+.3f}] | 0 in CI: {diff_info['zero_in_ci']}")

        print("\n**Matched Operating Points (Extra Workers at Matched Miss Rate)**:")
        mof = h_data["matched_operating_frontiers"]
        for om_key, om_data in mof.items():
            t_miss = om_data["target_miss_pct"]
            print(f"  * Target Miss Rate = {t_miss:.2f}%:")
            for m, m_info in om_data["methods"].items():
                if "extra_workers_mean" in m_info:
                    print(f"      {m:15}: {m_info['extra_workers_mean']:.3f} workers [95% CI: {m_info['extra_workers_ci95'][0]:.3f}, {m_info['extra_workers_ci95'][1]:.3f}]")
                else:
                    print(f"      {m:15}: {m_info.get('status', 'not reached')}")


if __name__ == "__main__":
    main()
