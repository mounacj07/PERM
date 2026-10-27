import warnings
warnings.filterwarnings("ignore")

"""
PERM - Predictive Edge Resource Management
Inference Service
"""

import argparse
import json
import logging
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from sklearn.compose import TransformedTargetRegressor

import features as feat_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference_service")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
INFERENCE_INTERVAL_SEC = 5.0    # inference cadence
WINDOW_SEC = 10                 # shorter = burst state captured sooner in feature window
PREDICTION_HORIZON_SEC = 30     # forecast horizon
BUFFER_MAXLEN = 720             # 1 hour of 5s predictions

DEFAULT_COLLECTOR_URL = "http://127.0.0.1:5000"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Thread-safe prediction buffer
_lock = threading.Lock()
prediction_buffer = deque(maxlen=BUFFER_MAXLEN)
_latest_prediction = None


# --------------------------------------------------------------------------
# Feature Extraction (delegates to shared features.py)
# --------------------------------------------------------------------------
def extract_features(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Summarizes a DataFrame of ~30 1-second telemetry rows into a single feature row.
    Delegates to features.window_features() — the single canonical implementation
    shared with feature_engineering.py so training and inference are always identical.
    """
    if len(df) == 0:
        return {col: 0.0 for col in feature_cols}

    feats = feat_lib.window_features(df)
    # Ensure all feature columns required by feature_columns.json are present
    return {col: feats.get(col, 0.0) for col in feature_cols}


# --------------------------------------------------------------------------
# Model Manager
# --------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cpu_model = None
        self.latency_model = None
        self.feature_cols = []
        self._model_lock = threading.Lock()
        self.load_models()

    def load_models(self):
        """Load (or hot-reload) models and feature schema from disk."""
        cpu_path = os.path.join(self.model_dir, "rf_cpu_model.joblib")
        lat_path = os.path.join(self.model_dir, "rf_latency_model.joblib")
        cols_path = os.path.join(self.model_dir, "feature_columns.json")

        if not (os.path.exists(cpu_path) and os.path.exists(lat_path) and os.path.exists(cols_path)):
            raise FileNotFoundError(
                f"Models or feature column schema missing in {self.model_dir}. "
                f"Run `python train_model.py` first."
            )

        cpu_model = joblib.load(cpu_path)
        latency_model = joblib.load(lat_path)
        with open(cols_path, "r") as f:
            feature_cols = json.load(f)

        with self._model_lock:
            self.cpu_model = cpu_model
            self.latency_model = latency_model
            self.feature_cols = feature_cols

        log.info("Models loaded successfully from %s. Features: %d", self.model_dir, len(feature_cols))

    def _get_feature_importances(self, model):
        """Extract feature importances across ensemble models or TransformedTargetRegressors."""
        target_estimator = model
        if isinstance(model, TransformedTargetRegressor):
            target_estimator = model.regressor_

        if hasattr(target_estimator, "feature_importances_"):
            return target_estimator.feature_importances_
        elif hasattr(target_estimator, "named_estimators_") and "et" in target_estimator.named_estimators_:
            et = target_estimator.named_estimators_["et"]
            if hasattr(et, "feature_importances_"):
                return et.feature_importances_
        return None

    def _explain_prediction(self, model, features_dict: dict, top_k=3) -> list:
        """Return top-k feature names by combined importance × magnitude."""
        importances = self._get_feature_importances(model)
        if importances is None:
            return ["req_mean", "cpu_mean", "cpu_momentum"]

        scored = []
        for i, col in enumerate(self.feature_cols):
            val = float(features_dict.get(col, 0.0))
            importance = float(importances[i])
            scored.append((col, abs(val) * importance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in scored[:top_k]]

    def _tree_uncertainty(self, model, X_df: pd.DataFrame) -> float:
        """
        Compute prediction uncertainty as the std dev of individual tree predictions.
        Handles VotingRegressor and TransformedTargetRegressor cleanly.
        """
        target_estimator = model
        is_log_transformed = False
        if isinstance(model, TransformedTargetRegressor):
            target_estimator = model.regressor_
            is_log_transformed = True

        et_model = None
        if hasattr(target_estimator, "estimators_"):
            if hasattr(target_estimator, "estimators_") and len(target_estimator.estimators_) > 10:
                et_model = target_estimator
            elif hasattr(target_estimator, "named_estimators_") and "et" in target_estimator.named_estimators_:
                et_model = target_estimator.named_estimators_["et"]

        if et_model is not None and hasattr(et_model, "estimators_"):
            trees = et_model.estimators_
            if len(trees) > 100:
                trees = trees[::len(trees) // 100][:100]

            tree_preds = np.array([float(t.predict(X_df.values)[0]) for t in trees])
            if is_log_transformed:
                tree_preds = np.expm1(tree_preds)
            return float(np.std(tree_preds))

        return 0.0

    def predict(self, telemetry_records: list) -> dict:
        if not telemetry_records:
            return None

        df = pd.DataFrame(telemetry_records)

        with self._model_lock:
            feature_cols = list(self.feature_cols)
            cpu_model = self.cpu_model
            latency_model = self.latency_model

        features_dict = extract_features(df, feature_cols)

        # Prepare DataFrame matching exact feature_cols order to preserve column names
        X_df = pd.DataFrame(
            [[features_dict[col] for col in feature_cols]],
            columns=feature_cols,
        )

        cpu_pred = float(cpu_model.predict(X_df)[0])
        lat_pred = float(latency_model.predict(X_df)[0])

        # ── Velocity-based momentum correction (CPU only) ───────────────────
        # When traffic is surging rapidly, the sliding-window mean lags behind.
        # We extrapolate the recent CPU velocity forward to catch sharp ramps early.
        # NOTE: Latency is NOT extrapolated here — its non-linear queueing behaviour
        # makes linear projection wildly inaccurate; the ML model handles it directly.
        n_recent = min(8, len(telemetry_records))
        if n_recent >= 4:
            recent_cpu = [float(r.get("cpu_percent", 0)) for r in telemetry_records[-n_recent:]]
            cpu_velocity = (recent_cpu[-1] - recent_cpu[0]) / max(1, n_recent - 1)
            # Only trigger momentum boost during active ramps (CPU >35% AND velocity >2.5%/s)
            if recent_cpu[-1] > 35.0 and cpu_velocity > 2.5:
                extrapolated = recent_cpu[-1] + cpu_velocity * PREDICTION_HORIZON_SEC * 0.60
                cpu_pred = max(cpu_pred, min(100.0, extrapolated))
                log.debug("Momentum boost: cpu_velocity=%.2f/s -> corrected_pred=%.1f",
                          cpu_velocity, cpu_pred)
        # ────────────────────────────────────────────────────────────────────

        # ── Saturation blend correction ───────────────────────────────────
        # Ensemble models shrink extreme predictions toward the training mean.
        # Use a smoothed "current" value (5-sample EWMA) so momentary 1-second
        # spikes don't trigger false blends — needs the TREND to be elevated.
        def _smooth(arr, alpha=0.4):
            v = arr[0]
            for x in arr[1:]:
                v = alpha * x + (1 - alpha) * v
            return float(v)

        n_smooth = min(5, len(telemetry_records))
        smooth_cpu = _smooth([float(r.get("cpu_percent", 0))
                              for r in telemetry_records[-n_smooth:]])
        smooth_lat = _smooth([float(r.get("avg_latency_ms", 0))
                              for r in telemetry_records[-n_smooth:]])

        cpu_gap = smooth_cpu - cpu_pred
        if cpu_gap > 18.0:
            blend = min(0.55, cpu_gap / 80.0)
            cpu_pred = min(100.0, cpu_pred + blend * cpu_gap)
            log.debug("Saturation blend CPU: gap=%.1f blend=%.2f -> pred=%.1f",
                      cpu_gap, blend, cpu_pred)

        # ── Queuing-theory physical saturation boost for Latency ────────────
        # When CPU is forecast to saturate (>65%), response latency scales non-linearly.
        if cpu_pred > 65.0:
            queue_factor = 1.0 + ((cpu_pred - 65.0) / 12.0) ** 1.7
            lat_pred = max(lat_pred, min(350.0, lat_pred * (queue_factor ** 0.65)))

        lat_gap = smooth_lat - lat_pred
        if lat_gap > 30.0:
            blend = min(0.70, lat_gap / 120.0)
            lat_pred = lat_pred + blend * lat_gap
        # ────────────────────────────────────────────────────────────────────



        # Uncertainty: std dev across individual trees
        cpu_uncertainty = self._tree_uncertainty(cpu_model, X_df)
        lat_uncertainty = self._tree_uncertainty(latency_model, X_df)

        current_cpu = float(df["cpu_percent"].iloc[-1])
        current_lat = float(df["avg_latency_ms"].iloc[-1])
        current_mem = float(df["mem_percent"].iloc[-1])

        cpu_drivers = self._explain_prediction(cpu_model, features_dict)
        lat_drivers = self._explain_prediction(latency_model, features_dict)

        now_utc = datetime.now(timezone.utc).isoformat()
        target_time_utc = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + PREDICTION_HORIZON_SEC, tz=timezone.utc
        ).isoformat()

        prediction = {
            "timestamp": now_utc,
            "target_timestamp_30s": target_time_utc,
            "horizon_sec": PREDICTION_HORIZON_SEC,
            "current_cpu": round(current_cpu, 2),
            "current_latency_ms": round(current_lat, 2),
            "current_mem": round(current_mem, 2),
            "predicted_cpu_30s": round(max(0.0, min(100.0, cpu_pred)), 2),
            "predicted_latency_30s": round(max(0.0, lat_pred), 2),
            "predicted_cpu_30s_uncertainty": round(cpu_uncertainty, 2),
            "predicted_latency_30s_uncertainty": round(lat_uncertainty, 2),
            "top_drivers_cpu": cpu_drivers,
            "top_drivers_latency": lat_drivers,
            "window_rows": len(df),
            "features": {k: round(v, 4) for k, v in features_dict.items()},
        }
        return prediction


# --------------------------------------------------------------------------
# Background Inference Loop Thread
# --------------------------------------------------------------------------
def inference_loop(engine: InferenceEngine, collector_url: str, stop_event: threading.Event):
    global _latest_prediction
    log.info("Starting background loop. Querying %s every %.1fs", collector_url, INFERENCE_INTERVAL_SEC)

    next_tick = time.monotonic()
    while not stop_event.is_set():
        next_tick += INFERENCE_INTERVAL_SEC
        try:
            resp = requests.get(
                f"{collector_url}/metrics/history?seconds={WINDOW_SEC}", timeout=3.0
            )
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("records", [])
                if records:
                    pred = engine.predict(records)
                    if pred:
                        with _lock:
                            _latest_prediction = pred
                            prediction_buffer.append(pred)
            else:
                log.warning("Collector returned HTTP %s", resp.status_code)
        except Exception as exc:
            log.warning("Failed to fetch/predict: %s", exc)

        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            stop_event.wait(sleep_time)
        else:
            next_tick = time.monotonic()


# --------------------------------------------------------------------------
# Flask REST API
# --------------------------------------------------------------------------
def create_app(engine: InferenceEngine, collector_url: str):
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "inference_service",
            "time": datetime.now(timezone.utc).isoformat(),
            "collector_url": collector_url,
        })

    @app.route("/predictions/latest", methods=["GET"])
    def predictions_latest():
        with _lock:
            if not _latest_prediction:
                return jsonify({"error": "No predictions generated yet"}), 503
            return jsonify(_latest_prediction)

    @app.route("/predictions/history", methods=["GET"])
    def predictions_history():
        seconds = request.args.get("seconds", default=300, type=int)
        count = max(1, min(int(seconds / INFERENCE_INTERVAL_SEC), BUFFER_MAXLEN))
        with _lock:
            records = list(prediction_buffer)[-count:]
        return jsonify({"count": len(records), "predictions": records})

    @app.route("/predict", methods=["POST"])
    def predict_custom():
        """Accepts custom telemetry array or triggers immediate forecast."""
        payload = request.get_json(silent=True) or {}
        telemetry_records = payload.get("records", [])
        if not telemetry_records:
            try:
                resp = requests.get(
                    f"{collector_url}/metrics/history?seconds={WINDOW_SEC}", timeout=3.0
                )
                if resp.status_code == 200:
                    telemetry_records = resp.json().get("records", [])
            except Exception as exc:
                return jsonify({"error": f"Failed to fetch telemetry: {exc}"}), 500

        if not telemetry_records:
            return jsonify({"error": "No telemetry available for prediction"}), 400

        pred = engine.predict(telemetry_records)
        return jsonify(pred)

    @app.route("/reload", methods=["POST"])
    def reload_models():
        """
        Hot-reload models from disk without restarting the service.
        Call this after running train_model.py to pick up updated .joblib files.
        """
        try:
            engine.load_models()
            return jsonify({
                "status": "reloaded",
                "model_dir": engine.model_dir,
                "features": len(engine.feature_cols),
                "time": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            log.error("Model reload failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    # Optional ST-AGBDT Demand Forecaster endpoint (enabled only via env PERM_FORECASTER=stagbdt)
    if os.environ.get("PERM_FORECASTER", "").strip().lower() == "stagbdt":
        stagbdt_engine = STAGBDTEngine(engine.model_dir)

        @app.route("/forecast/demand", methods=["GET", "POST"])
        def forecast_demand():
            try:
                payload = request.get_json(silent=True) or {}
                horizon = request.args.get("horizon", default=payload.get("horizon", 20), type=int)
                arrivals = payload.get("arrivals")
                if arrivals is None:
                    records = payload.get("records")
                    if not records:
                        try:
                            resp = requests.get(f"{collector_url}/metrics/history?seconds=120", timeout=3.0)
                            if resp.status_code == 200:
                                records = resp.json().get("records", [])
                        except Exception:
                            pass
                    if records:
                        arrivals = [float(r.get("arrivals_count", 0.0)) for r in records]

                if not arrivals:
                    return jsonify({"error": "No arrival history available for prediction"}), 400

                res = stagbdt_engine.predict_demand(arrivals, horizon=horizon)
                return jsonify(res)
            except Exception as exc:
                log.error("Error in /forecast/demand: %s", exc)
                return jsonify({"error": str(exc)}), 500

    return app


class STAGBDTEngine:
    """Helper engine for ST-AGBDT demand forecasting."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.config: Dict[str, Any] = {}
        self.models: Dict[int, Any] = {}
        self.load()

    def load(self):
        cfg_path = os.path.join(self.model_dir, "stagbdt_final_config.json")
        if not os.path.exists(cfg_path):
            log.warning("ST-AGBDT config not found at %s", cfg_path)
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        from stagbdt.model import STAGBDT
        project_root = os.path.dirname(os.path.abspath(self.model_dir))
        for h_key, rel_path in self.config.get("model_files", {}).items():
            h = int(h_key.replace("h", ""))
            full_path = os.path.join(project_root, rel_path)
            if os.path.exists(full_path):
                self.models[h] = STAGBDT.load_json(full_path)

    def predict_demand(self, arrivals_list: List[float], horizon: int = 20) -> Dict[str, Any]:
        if horizon not in self.models:
            raise ValueError(f"Horizon h={horizon} not available. Available: {list(self.models.keys())}")
        model = self.models[horizon]
        from stagbdt.features import extract_features_single

        arr = np.asarray(arrivals_list, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("Arrivals history cannot be empty")

        feats = extract_features_single(arr)
        raw_pred = float(model.predict_one(feats))
        margin = float(self.config.get("conformal_margins", {}).get(f"h{horizon}", 0.0))
        upper_bound = raw_pred + margin
        eff_cap = float(self.config.get("effective_capacity_per_worker", 5.523))
        rec_workers = int(np.clip(math.ceil(upper_bound / eff_cap), 2, 8))

        return {
            "forecaster": "stagbdt",
            "horizon_sec": horizon,
            "forecast": round(raw_pred, 3),
            "conformal_margin": round(margin, 3),
            "upper_bound": round(upper_bound, 3),
            "recommended_workers": rec_workers,
            "capacity_per_worker": float(self.config.get("capacity_per_worker", 7.89)),
            "target_utilization": float(self.config.get("target_utilization", 0.70)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="PERM Inference Service")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--collector-url", type=str, default=DEFAULT_COLLECTOR_URL)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    engine = InferenceEngine(args.model_dir)

    stop_event = threading.Event()
    loop_thread = threading.Thread(
        target=inference_loop,
        args=(engine, args.collector_url, stop_event),
        daemon=True,
    )
    loop_thread.start()

    app = create_app(engine, args.collector_url)
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        stop_event.set()
        loop_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
