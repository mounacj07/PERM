"""
run_part_c_benchmark.py
=======================
Step 3b-L Part C: Speed and early exit benchmarks executed ALONE.
1. Logs 'START QUIET BENCHMARK'
2. Waits 60s in complete quiet
3. Benchmarks single-row prediction latency across 20,000 calls after warm-up:
   - recursive
   - old flat
   - stacked (all trees in 2-D arrays, level-walk)
   - sklearn single-row (HistGradientBoostingRegressor)
   - custom_ml_engine
4. Evaluates early exit on Locust dataset for W in {2, 4, 8}:
   - Threshold = (W * 0.7) / 0.126
   - Reports: % exited, mean trees evaluated, decision agreement %, extra misses %.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from custom_ml_engine import CustomGradientBoostingRegressor
from stagbdt import STAGBDT, EarlyExitCalibrator
from eval_forecaster import extract_run_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("part_c_bench")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "runs_locust")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "locust")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 80)
    print("START QUIET BENCHMARK")
    print("=" * 80)
    log.info("Entering 60-second settling period with zero concurrent workload...")
    for sec_left in range(60, 0, -10):
        log.info("Settling: %ds remaining...", sec_left)
        time.sleep(10)

    log.info("Settling complete. Loading Locust dataset for training and test...")
    # Load canonical Locust dataset
    csv_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "_stats" not in f and "phase_log" not in f])
    runs = [extract_run_data(p) for p in csv_files]
    runs = [r for r in runs if r is not None]

    # Training set: fixed + random_mix 0-19; Test: random_mix 20-29; Cal: random_mix 15-19
    fixed_runs = [r for r in runs if r["profile"] != "random_mix"]
    rmix_runs = {r["seed"]: r for r in runs if r["profile"] == "random_mix"}

    train_runs = fixed_runs + [rmix_runs[s] for s in range(15) if s in rmix_runs]
    cal_runs = [rmix_runs[s] for s in range(15, 20) if s in rmix_runs]
    test_runs = [rmix_runs[s] for s in range(20, 30) if s in rmix_runs]

    h = 20
    X_tr = np.vstack([r["X"] for r in train_runs])
    y_tr = np.concatenate([r["targets_3s"][h] for r in train_runs])

    X_cal = np.vstack([r["X"] for r in cal_runs])
    y_cal = np.concatenate([r["targets_3s"][h] for r in cal_runs])

    X_te = np.vstack([r["X"] for r in test_runs])
    y_te = np.concatenate([r["targets_3s"][h] for r in test_runs])

    log.info("Fitting ST-AGBDT (alpha=0.85, 50 trees, max_depth=5)...")
    stag = STAGBDT(n_estimators=50, max_depth=5, alpha=0.85, random_state=42).fit(X_tr, y_tr)

    log.info("Fitting Sklearn HistGB and Custom ML Engine...")
    hgb = HistGradientBoostingRegressor(max_depth=5, random_state=42).fit(X_tr, y_tr)
    c_gbdt = CustomGradientBoostingRegressor(n_estimators=25, max_depth=4, max_features="sqrt", random_state=42).fit(X_tr, y_tr)

    # 1. Single-row prediction latency benchmark (20,000 evaluations after 1,000 warmups)
    log.info("Running single-sample latency benchmark (20,000 calls after 1,000 warm-ups)...")
    x0 = X_te[0]
    x0_2d = x0.reshape(1, -1)

    n_warmup = 1000
    n_evals = 20000

    # Recursive
    for _ in range(n_warmup):
        stag.predict_one(x0, use_flat=False)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        stag.predict_one(x0, use_flat=False)
    rec_lat_us = round(((time.perf_counter() - t0) / n_evals) * 1e6, 2)

    # Old Flat
    for _ in range(n_warmup):
        stag.predict_one(x0, use_flat=True)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        stag.predict_one(x0, use_flat=True)
    flat_lat_us = round(((time.perf_counter() - t0) / n_evals) * 1e6, 2)

    # Stacked
    for _ in range(n_warmup):
        stag.predict_one_stacked(x0)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        stag.predict_one_stacked(x0)
    stacked_lat_us = round(((time.perf_counter() - t0) / n_evals) * 1e6, 2)

    # Sklearn HistGB single-row
    for _ in range(n_warmup):
        hgb.predict(x0_2d)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        hgb.predict(x0_2d)
    hgb_lat_us = round(((time.perf_counter() - t0) / n_evals) * 1e6, 2)

    # Custom ML engine
    for _ in range(n_warmup):
        c_gbdt.predict(x0_2d)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        c_gbdt.predict(x0_2d)
    c_gbdt_lat_us = round(((time.perf_counter() - t0) / n_evals) * 1e6, 2)

    speed_table = {
        "recursive_us": rec_lat_us,
        "old_flat_us": flat_lat_us,
        "stacked_us": stacked_lat_us,
        "sklearn_HistGB_single_row_us": hgb_lat_us,
        "custom_ml_engine_us": c_gbdt_lat_us,
    }

    print("\n" + "=" * 70)
    print("SINGLE-ROW PREDICTION LATENCY (20,000 Evaluations)")
    print("=" * 70)
    for model_name, lat in speed_table.items():
        print(f"  - {model_name:<30}: {lat:>8.2f} µs")

    # 2. Early Exit Benchmark for W in {2, 4, 8}
    log.info("Running Early Exit Evaluation on Locust data for W in {2, 4, 8}...")
    workers_list = [2, 4, 8]
    K0_list = [10, 20]
    ee_results = {}

    full_preds = stag.predict(X_te)

    for W in workers_list:
        threshold = (W * 0.7) / 0.126
        full_decision = (full_preds >= threshold)
        ee_results[f"W_{W}"] = {"threshold_rps": round(threshold, 2), "configs": {}}

        for K0 in K0_list:
            # Calibrate margin on calibration set
            pred_K0_cal = np.array([
                stag.predict_with_early_exit(x, K0=K0, threshold=threshold, margin=0.0, eps=1.0)[0]
                for x in X_cal
            ])
            ee_cal = EarlyExitCalibrator(coverage_level=0.95)
            margin = ee_cal.calibrate(y_cal, pred_K0_cal)

            # Evaluate on test set
            preds = []
            trees_used = []
            exited = []
            for x in X_te:
                p, k_used, ex = stag.predict_with_early_exit(x, K0=K0, threshold=threshold, margin=margin, eps=1.0)
                preds.append(p)
                trees_used.append(k_used)
                exited.append(ex)

            preds = np.array(preds)
            trees_used = np.array(trees_used)
            exited = np.array(exited)

            pct_exited = round(float(np.mean(exited)) * 100.0, 2)
            mean_trees = round(float(np.mean(trees_used)), 2)
            ee_decision = (preds >= threshold)
            agreement = round(float(np.mean(full_decision == ee_decision)) * 100.0, 2)
            extra_misses = round(float(np.mean((full_decision) & (~ee_decision))) * 100.0, 2)

            ee_results[f"W_{W}"]["configs"][f"K0_{K0}"] = {
                "margin": round(margin, 3),
                "pct_exited": pct_exited,
                "mean_trees": mean_trees,
                "agreement_pct": agreement,
                "extra_misses_pct": extra_misses,
            }

    print("\n" + "=" * 85)
    print("EARLY EXIT BENCHMARK ON LOCUST DATA (W in {2, 4, 8})")
    print("=" * 85)
    header = f"{'Active Workers (W)':<20} | {'Threshold':<11} | {'K0 Trees':<10} | {'% Exited':<10} | {'Mean Trees':<12} | {'Agreement %':<13} | {'Extra Misses %'}"
    print(header)
    print("-" * 95)
    for W in workers_list:
        thr = ee_results[f"W_{W}"]["threshold_rps"]
        for K0 in K0_list:
            cfg = ee_results[f"W_{W}"]["configs"][f"K0_{K0}"]
            print(f"W = {W:<16} | {thr:>7.2f} rps | {K0:<10} | {cfg['pct_exited']:>8.2f}% | {cfg['mean_trees']:>12.2f} | {cfg['agreement_pct']:>11.2f}% | {cfg['extra_misses_pct']:>12.2f}%")

    out_file = os.path.join(OUT_DIR, "part_c_speed_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"speed_table": speed_table, "early_exit": ee_results}, f, indent=2)

    log.info("Part C benchmarks completed. Results written to %s", out_file)


if __name__ == "__main__":
    main()
