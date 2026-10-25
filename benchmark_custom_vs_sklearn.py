"""
PERM - Benchmark Suite: Custom Scratch Model vs. Sklearn Baseline

Compares the from-scratch ML model (custom_ml_engine.py) against sklearn's
prebuilt ensemble model on held-out test splits.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from custom_ml_engine import CustomEnsembleRegressor, CustomGradientBoostingRegressor, CustomLogTransformedTargetRegressor

NON_FEATURE_COLS = ["window_end", "phase", "target_cpu_percent", "target_latency_ms"]


def load_test_split():
    df = pd.read_csv("features_test.csv")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy()
    y_cpu = df["target_cpu_percent"].to_numpy()
    y_lat = df["target_latency_ms"].to_numpy()
    return X, y_cpu, y_lat, feature_cols


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / max(1e-8, ss_tot))
    return mae, rmse, r2


def main():
    print("=" * 80)
    print(" HEAD-TO-HEAD BENCHMARK: Custom Scratch ML Engine vs. Sklearn Ensemble ")
    print("=" * 80)

    X_test, y_cpu_test, y_lat_test, feature_cols = load_test_split()
    n_samples = len(X_test)
    print(f"Evaluated on Held-Out Test Set (n={n_samples} samples, {len(feature_cols)} features)\n")

    # 1. Custom Scratch Model
    custom_cpu_path = os.path.join("models", "custom_cpu_model.json")
    custom_lat_path = os.path.join("models", "custom_latency_model.json")

    custom_cpu = CustomEnsembleRegressor.load_model(custom_cpu_path)
    custom_lat = CustomLogTransformedTargetRegressor.load_model(custom_lat_path)

    t0 = time.time()
    for _ in range(10):
        pred_cpu_custom = custom_cpu.predict(X_test)
        pred_lat_custom = custom_lat.predict(X_test)
    t_custom = (time.time() - t0) / 10.0
    latency_per_sample_custom = (t_custom / n_samples) * 1e6 # microseconds

    mae_cpu_c, rmse_cpu_c, r2_cpu_c = evaluate(y_cpu_test, pred_cpu_custom)
    mae_lat_c, rmse_lat_c, r2_lat_c = evaluate(y_lat_test, pred_lat_custom)

    # 2. Sklearn Baseline Model
    sk_cpu_path = os.path.join("models", "rf_cpu_model.joblib")
    sk_lat_path = os.path.join("models", "rf_latency_model.joblib")

    if os.path.exists(sk_cpu_path) and os.path.exists(sk_lat_path):
        sk_cpu = joblib.load(sk_cpu_path)
        sk_lat = joblib.load(sk_lat_path)

        t0 = time.time()
        for _ in range(10):
            pred_cpu_sk = sk_cpu.predict(X_test)
            pred_lat_sk = sk_lat.predict(X_test)
        t_sk = (time.time() - t0) / 10.0
        latency_per_sample_sk = (t_sk / n_samples) * 1e6

        mae_cpu_s, rmse_cpu_s, r2_cpu_s = evaluate(y_cpu_test, pred_cpu_sk)
        mae_lat_s, rmse_lat_s, r2_lat_s = evaluate(y_lat_test, pred_lat_sk)
    else:
        mae_cpu_s = rmse_cpu_s = r2_cpu_s = mae_lat_s = rmse_lat_s = r2_lat_s = 0.0
        latency_per_sample_sk = 0.0

    print(f"{'Metric':<25} | {'Custom Scratch Model':<24} | {'Sklearn Ensemble Model':<24}")
    print("-" * 80)
    print(f"{'CPU MAE (%)':<25} | {mae_cpu_c:<24.3f} | {mae_cpu_s:<24.3f}")
    print(f"{'CPU RMSE (%)':<25} | {rmse_cpu_c:<24.3f} | {rmse_cpu_s:<24.3f}")
    print(f"{'CPU R² Score':<25} | {r2_cpu_c:<24.3f} | {r2_cpu_s:<24.3f}")
    print("-" * 80)
    print(f"{'Latency MAE (ms)':<25} | {mae_lat_c:<24.3f} | {mae_lat_s:<24.3f}")
    print(f"{'Latency RMSE (ms)':<25} | {rmse_lat_c:<24.3f} | {rmse_lat_s:<24.3f}")
    print(f"{'Latency R² Score':<25} | {r2_lat_c:<24.3f} | {r2_lat_s:<24.3f}")
    print("-" * 80)
    print(f"{'Batch Pred Speed (us/sample)':<25} | {latency_per_sample_custom:<24.2f} | {latency_per_sample_sk:<24.2f}")
    print(f"{'External ML Model Libs':<25} | {'0 (Pure Python/NumPy)':<24} | {'scikit-learn + joblib':<24}")
    print("=" * 80)


if __name__ == "__main__":
    main()
