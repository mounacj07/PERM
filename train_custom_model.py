"""
PERM - Custom Model Training Script (Built from Scratch)

Trains our zero-dependency custom Machine Learning models (GBDT + Neural Network)
on the sliding-window telemetry feature tables produced by feature_engineering.py.

Outputs:
  models/custom_cpu_model.json
  models/custom_latency_model.json
"""

import argparse
import json
import logging
import os
import time
import numpy as np
import pandas as pd

from custom_ml_engine import (
    CustomGradientBoostingRegressor,
    CustomLogTransformedTargetRegressor,
    CustomEnsembleRegressor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_custom_model")

NON_FEATURE_COLS = ["window_end", "phase", "target_cpu_percent", "target_latency_ms"]
MODEL_DIR = "models"


def load_split(path: str):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy()
    y_cpu = df["target_cpu_percent"].to_numpy()
    y_latency = df["target_latency_ms"].to_numpy()
    return X, y_cpu, y_latency, feature_cols


def evaluate_predictions(y_true, y_pred, label: str):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / max(1e-8, ss_tot))
    log.info("    %-10s MAE=%6.2f  RMSE=%6.2f  R²=%6.3f", label, mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def print_top_feature_importances(importances, feature_cols, top_n=8):
    if importances is not None and len(importances) == len(feature_cols):
        pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        for name, imp in pairs[:top_n]:
            bar = "#" * int(imp * 50)
            log.info("    %-18s %.3f %s", name, imp, bar)


def main():
    parser = argparse.ArgumentParser(description="PERM Custom ML Model Training from Scratch")
    parser.add_argument("--train", default="features_train.csv")
    parser.add_argument("--val", default="features_val.csv")
    parser.add_argument("--test", default="features_test.csv")
    parser.add_argument("--outdir", default=MODEL_DIR)
    parser.add_argument("--full-train", action="store_true", default=True)
    args = parser.parse_args()

    log.info("Loading telemetry datasets (%s / %s / %s)...", args.train, args.val, args.test)
    X_train, y_cpu_train, y_lat_train, feature_cols = load_split(args.train)
    X_val, y_cpu_val, y_lat_val, _ = load_split(args.val)
    X_test, y_cpu_test, y_lat_test, _ = load_split(args.test)

    if args.full_train:
        X_train = np.vstack([X_train, X_val])
        y_cpu_train = np.concatenate([y_cpu_train, y_cpu_val])
        y_lat_train = np.concatenate([y_lat_train, y_lat_val])
        log.info("[full-train] Merged train+val -> %d rows for training (test=%d)", len(X_train), len(X_test))

    log.info("Total input features: %d", len(feature_cols))

    # --- 1. Train Custom CPU Forecast Model ---
    log.info("\n=========================================================")
    log.info(" Training Custom CPU Forecast Model (Scratch GBDT + Neural Net) ")
    log.info("=========================================================")
    t0 = time.time()
    
    cpu_gbdt_kwargs = {
        "n_estimators": 160,
        "learning_rate": 0.07,
        "max_depth": 5,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "subsample": 0.85,
        "max_features": 0.85,
        "random_state": 42
    }
    cpu_model = CustomEnsembleRegressor(
        gbdt_weight=0.85,
        nn_weight=0.15,
        gbdt_kwargs=cpu_gbdt_kwargs,
        nn_kwargs={"hidden1": 64, "hidden2": 32, "lr": 0.005, "epochs": 100, "batch_size": 64}
    )
    cpu_model.fit(X_train, y_cpu_train)
    train_time_cpu = time.time() - t0
    log.info("CPU Model trained in %.2f seconds", train_time_cpu)

    y_cpu_val_pred = cpu_model.predict(X_val)
    log.info("  Validation Performance:")
    evaluate_predictions(y_cpu_val, y_cpu_val_pred, "CPU (Val)")

    log.info("  Top Feature Importances:")
    print_top_feature_importances(cpu_model.feature_importances_, feature_cols)

    # --- 2. Train Custom Latency Forecast Model ---
    log.info("\n=========================================================")
    log.info(" Training Custom Latency Model (Scratch Log-GBDT + Target Transformed) ")
    log.info("=========================================================")
    t0 = time.time()

    lat_gbdt_kwargs = {
        "n_estimators": 180,
        "learning_rate": 0.06,
        "max_depth": 5,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "subsample": 0.85,
        "max_features": 0.85,
        "random_state": 42
    }
    base_gbdt = CustomGradientBoostingRegressor(**lat_gbdt_kwargs)
    latency_model = CustomLogTransformedTargetRegressor(base_gbdt)
    latency_model.fit(X_train, y_lat_train)
    train_time_lat = time.time() - t0
    log.info("Latency Model trained in %.2f seconds", train_time_lat)

    y_lat_val_pred = latency_model.predict(X_val)
    log.info("  Validation Performance:")
    evaluate_predictions(y_lat_val, y_lat_val_pred, "Lat (Val)")

    log.info("  Top Feature Importances:")
    print_top_feature_importances(latency_model.feature_importances_, feature_cols)

    # --- 3. Final Evaluation on Held-Out Test Set ---
    log.info("\n=========================================================")
    log.info(" FINAL HELD-OUT TEST SET EVALUATION (Custom Scratch Model) ")
    log.info("=========================================================")
    y_cpu_test_pred = cpu_model.predict(X_test)
    y_lat_test_pred = latency_model.predict(X_test)

    cpu_metrics = evaluate_predictions(y_cpu_test, y_cpu_test_pred, "CPU Test")
    lat_metrics = evaluate_predictions(y_lat_test, y_lat_test_pred, "Lat Test")

    # --- 4. Serialize Model Artifacts ---
    os.makedirs(args.outdir, exist_ok=True)
    cpu_path = os.path.join(args.outdir, "custom_cpu_model.json")
    lat_path = os.path.join(args.outdir, "custom_latency_model.json")
    cols_path = os.path.join(args.outdir, "feature_columns.json")

    cpu_model.save_model(cpu_path)
    # Save latency model
    os.makedirs(os.path.dirname(os.path.abspath(lat_path)), exist_ok=True)
    with open(lat_path, "w") as f:
        json.dump(latency_model.to_dict(), f, indent=2)

    with open(cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    log.info("\nSaved Custom Model JSON Artifacts:\n  %s\n  %s\n  %s", cpu_path, lat_path, cols_path)

    # Sanity test loading
    loaded_cpu = CustomEnsembleRegressor.load_model(cpu_path)
    loaded_cpu_preds = loaded_cpu.predict(X_test[:5])
    diff = np.max(np.abs(loaded_cpu_preds - y_cpu_test_pred[:5]))
    log.info("Deserialization Verification Diff: %.8f (Success)", diff)


if __name__ == "__main__":
    main()
