import warnings
warnings.filterwarnings("ignore")

"""
PERM - Predictive Edge Resource Management
Model Training & Tuning Pipeline

Trains two high-accuracy Ensemble Regressors on the sliding-window feature tables
produced by feature_engineering.py:

  1. A CPU model:     features -> target_cpu_percent  (30s-ahead CPU% forecast)
  2. A latency model:  features -> target_latency_ms    (30s-ahead latency forecast)

Uses a VotingRegressor ensemble (ExtraTrees + HistGradientBoosting) with log-transformed
latency targets for maximum accuracy and robustness against burst spikes.

Run:
    python train_model.py
    python train_model.py --train features_train.csv --val features_val.csv --test features_test.csv
"""

import argparse
import json
import logging
import os
import warnings

# Suppress scikit-learn feature names UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_model")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NON_FEATURE_COLS = ["window_end", "phase", "target_cpu_percent", "target_latency_ms"]
MODEL_DIR = "models"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_split(path: str):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y_cpu = df["target_cpu_percent"]
    y_latency = df["target_latency_ms"]
    return X, y_cpu, y_latency, feature_cols


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(model, X, y_true, label: str):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    log.info("    %-6s MAE=%6.2f  RMSE=%6.2f  R²=%6.3f", label, mae, rmse, r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def print_top_feature_importances(model, feature_cols, top_n=8):
    target_estimator = model
    if isinstance(model, TransformedTargetRegressor):
        target_estimator = model.regressor_

    importances = None
    if hasattr(target_estimator, "feature_importances_"):
        importances = target_estimator.feature_importances_
    elif hasattr(target_estimator, "named_estimators_") and "et" in target_estimator.named_estimators_:
        et = target_estimator.named_estimators_["et"]
        if hasattr(et, "feature_importances_"):
            importances = et.feature_importances_

    if importances is not None:
        pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        for name, imp in pairs[:top_n]:
            bar = "#" * int(imp * 50)
            log.info("    %-18s %.3f %s", name, imp, bar)


# --------------------------------------------------------------------------
# Model Factories
# --------------------------------------------------------------------------
def build_cpu_model(architecture: str):
    if architecture == "ensemble":
        et = ExtraTreesRegressor(n_estimators=600, max_depth=14, min_samples_leaf=2, max_features=0.90, random_state=42, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=500, max_depth=8, min_samples_leaf=3, learning_rate=0.04, l2_regularization=0.01, random_state=42)
        return VotingRegressor(estimators=[("et", et), ("hgb", hgb)])
    elif architecture == "extratrees":
        return ExtraTreesRegressor(n_estimators=600, max_depth=14, min_samples_leaf=2, max_features=0.90, random_state=42, n_jobs=-1)
    else:
        return RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)


def build_latency_model(architecture: str):
    """
    Latency models use TransformedTargetRegressor with log1p transformation to handle
    heavy right-skewed response times during burst traffic.
    """
    if architecture == "ensemble":
        et = ExtraTreesRegressor(n_estimators=600, max_depth=14, min_samples_leaf=2, max_features=0.90, random_state=42, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=500, max_depth=8, min_samples_leaf=3, learning_rate=0.04, l2_regularization=0.01, random_state=42)
        base = VotingRegressor(estimators=[("et", et), ("hgb", hgb)])
    elif architecture == "extratrees":
        base = ExtraTreesRegressor(n_estimators=600, max_depth=14, min_samples_leaf=2, max_features=0.90, random_state=42, n_jobs=-1)
    else:
        base = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)

    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PERM high-accuracy model training")
    parser.add_argument("--train", default="features_train.csv")
    parser.add_argument("--val", default="features_val.csv")
    parser.add_argument("--test", default="features_test.csv")
    parser.add_argument("--outdir", default=MODEL_DIR)
    parser.add_argument("--model-type", choices=["ensemble", "extratrees", "rf_tuned"], default="ensemble",
                        help="Model architecture: 'ensemble' (ExtraTrees+GradientBoosting, default), 'extratrees', or 'rf_tuned'")
    parser.add_argument("--full-train", action="store_true", default=True,
                        help="Train on train+val combined for maximum data usage (recommended)")
    parser.add_argument("--no-full-train", dest="full_train", action="store_false")
    args = parser.parse_args()

    log.info("Loading %s / %s / %s ...", args.train, args.val, args.test)
    X_train, y_cpu_train, y_lat_train, feature_cols = load_split(args.train)
    X_val, y_cpu_val, y_lat_val, _ = load_split(args.val)
    X_test, y_cpu_test, y_lat_test, _ = load_split(args.test)

    if args.full_train:
        X_train = pd.concat([X_train, X_val], ignore_index=True)
        y_cpu_train = pd.concat([y_cpu_train, y_cpu_val], ignore_index=True)
        y_lat_train = pd.concat([y_lat_train, y_lat_val], ignore_index=True)
        log.info("[full-train] Merged train+val -> %d rows for training (test=%d)", len(X_train), len(X_test))

    log.info("Total features: %d", len(feature_cols))

    cpu_model = build_cpu_model(args.model_type)
    latency_model = build_latency_model(args.model_type)

    # -- CPU model ----------------------------------------------------------
    log.info("\nTraining CPU forecast model (%s architecture, 30s ahead) ...", args.model_type)
    cpu_model.fit(X_train, y_cpu_train)
    log.info("  Validation performance:")
    evaluate(cpu_model, X_val, y_cpu_val, "val")
    log.info("  Top feature importances:")
    print_top_feature_importances(cpu_model, feature_cols)

    # -- Latency model --------------------------------------------------------
    log.info("\nTraining Latency forecast model (%s architecture + Log-Target, 30s ahead) ...", args.model_type)
    latency_model.fit(X_train, y_lat_train)
    log.info("  Validation performance:")
    evaluate(latency_model, X_val, y_lat_val, "val")
    log.info("  Top feature importances:")
    print_top_feature_importances(latency_model, feature_cols)

    # -- Final test-set evaluation -------------------------------------------
    log.info("\n=========================================================")
    log.info(" FINAL HELD-OUT TEST SET PERFORMANCE (Reported Once) ")
    log.info("=========================================================")
    evaluate(cpu_model, X_test, y_cpu_test, "CPU")
    evaluate(latency_model, X_test, y_lat_test, "latency")

    # -- Serialize ------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    cpu_path = os.path.join(args.outdir, "rf_cpu_model.joblib")
    lat_path = os.path.join(args.outdir, "rf_latency_model.joblib")
    cols_path = os.path.join(args.outdir, "feature_columns.json")

    joblib.dump(cpu_model, cpu_path)
    joblib.dump(latency_model, lat_path)
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    log.info("\nSaved model artifacts:\n  %s\n  %s\n  %s", cpu_path, lat_path, cols_path)


if __name__ == "__main__":
    main()