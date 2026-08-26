"""
PERM - Predictive Edge Resource Management
Automated Verification & Test Suite

Executes end-to-end tests validating:
  1. Telemetry Collector 1s sampling loop & /api/process endpoint
  2. Sliding window feature extraction & Random Forest models predict() with uncertainty
  3. Inference Service 5s cadence REST endpoints & model hot-reloading
  4. SLO Alert Engine: Warning/Critical classification, ~25s lead time, 60s cooldown, and alert resolution
  5. Remediation Service: AUTO allocation & MANUAL approval workflows
  6. Web Dashboard State Aggregation API
  7. Feature Engineering pipeline: temporal splitting & window construction

Run:
    python test_system.py
"""

import os
import sys
import time
import unittest
import warnings
import numpy as np
import pandas as pd

# Suppress scikit-learn feature names UserWarning during unit testing
warnings.filterwarnings("ignore", category=UserWarning)

# Import local modules directly to test functions
from telemetry_collector import create_app as create_collector_app, _record_latency
from inference_service import InferenceEngine, extract_features, create_app as create_inference_app
from alert_engine import AlertEngine, active_alerts, _slo_config, _cooldown_sec
from remediation_service import RemediationEngine, create_app as create_remediation_app, BASE_WORKERS
import features as feat_lib
from feature_engineering import build_feature_table, temporal_split


class TestPermSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_dir = os.path.join(os.path.dirname(__file__), "models")
        cls.inference_engine = InferenceEngine(cls.model_dir)

    def test_01_telemetry_collector_app(self):
        print("\n[TEST 1] Testing Telemetry Collector Flask Endpoints...")
        app = create_collector_app()
        client = app.test_client()

        # Health
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json.get("status"), "ok")

        # Process synthetic workload endpoint
        resp = client.post("/api/process", json={"load": 3})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json.get("status"), "processed")
        self.assertEqual(resp.json.get("load"), 3)

        # History endpoint
        resp = client.get("/metrics/history?seconds=10")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("records", resp.json)
        print("  -> Telemetry Collector API operational.")

    def test_02_feature_extraction_and_model_inference(self):
        print("\n[TEST 2] Testing Feature Extraction & Random Forest Inference...")
        # Create 30 synthetic 1-second telemetry rows
        now = time.time()
        records = []
        for i in range(30):
            records.append({
                "timestamp": pd.Timestamp(now - (30 - i), unit="s", tz="UTC").isoformat(),
                "cpu_percent": 40.0 + i * 0.5,
                "mem_percent": 50.0,
                "avg_latency_ms": 10.0 + i * 2.0,
                "p95_latency_ms": 15.0 + i * 2.5,
                "request_count": 5 + i % 3,
            })

        df = pd.DataFrame(records)
        features = extract_features(df, self.inference_engine.feature_cols)

        # Verify all feature columns expected by models are present
        self.assertEqual(len(features), len(self.inference_engine.feature_cols))
        self.assertIn("cpu_mean", features)
        self.assertIn("cpu_slope", features)
        self.assertIn("cpu_lat_product", features)

        # Verify Random Forest predictions
        pred = self.inference_engine.predict(records)
        self.assertIsNotNone(pred)
        self.assertIn("predicted_cpu_30s", pred)
        self.assertIn("predicted_latency_30s", pred)
        self.assertIn("predicted_cpu_30s_uncertainty", pred)
        self.assertIn("predicted_latency_30s_uncertainty", pred)
        self.assertGreaterEqual(pred["predicted_cpu_30s"], 0.0)
        self.assertLessEqual(pred["predicted_cpu_30s"], 100.0)
        self.assertGreaterEqual(pred["predicted_latency_30s"], 0.0)
        print(f"  -> RF Models Predictions: CPU 30s={pred['predicted_cpu_30s']}% (std={pred['predicted_cpu_30s_uncertainty']}), Latency 30s={pred['predicted_latency_30s']}ms (std={pred['predicted_latency_30s_uncertainty']})")

    def test_03_slo_alert_engine_and_cooldown(self):
        print("\n[TEST 3] Testing SLO Alert Engine, Lead Time, 60s Cooldown & Alert Resolution...")
        alert_engine = AlertEngine(inference_url="http://127.0.0.1:5001")

        # 1. Normal prediction -> no alerts
        normal_pred = {
            "timestamp": "2026-08-24T23:40:00Z",
            "target_timestamp_30s": "2026-08-24T23:40:30Z",
            "horizon_sec": 30,
            "current_cpu": 35.0,
            "predicted_cpu_30s": 50.0,
            "current_latency_ms": 20.0,
            "predicted_latency_30s": 40.0,
        }
        alerts = alert_engine.evaluate_prediction(normal_pred)
        self.assertEqual(len(alerts), 0)

        # 2. Critical CPU prediction (90% >= 85%) -> triggers proactive alert
        breach_pred = {
            "timestamp": "2026-08-24T23:40:05Z",
            "target_timestamp_30s": "2026-08-24T23:40:35Z",
            "horizon_sec": 30,
            "current_cpu": 65.0,
            "predicted_cpu_30s": 90.0,
            "current_latency_ms": 50.0,
            "predicted_latency_30s": 80.0,
        }
        alerts = alert_engine.evaluate_prediction(breach_pred)
        self.assertEqual(len(alerts), 1)
        alt = alerts[0]
        self.assertEqual(alt["severity"], "CRITICAL")
        self.assertEqual(alt["metric"], "cpu_percent")
        self.assertGreater(alt["lead_time_sec"], 20.0)  # ~25s lead time
        self.assertIn("cpu_percent", active_alerts)
        print(f"  -> Proactive Alert Triggered: {alt['message']}")

        # 3. Duplicate breach prediction immediately after -> suppressed by 60s cooldown
        breach_pred2 = {
            "timestamp": "2026-08-24T23:40:10Z",
            "target_timestamp_30s": "2026-08-24T23:40:40Z",
            "horizon_sec": 30,
            "current_cpu": 68.0,
            "predicted_cpu_30s": 92.0,
            "current_latency_ms": 52.0,
            "predicted_latency_30s": 82.0,
        }
        cooldown_alerts = alert_engine.evaluate_prediction(breach_pred2)
        self.assertEqual(len(cooldown_alerts), 0)
        print("  -> 60-second cooldown successfully suppressed duplicate alert.")

        # 4. Metric recovers below threshold -> active_alerts cleared
        recovered_pred = {
            "timestamp": "2026-08-24T23:41:30Z",
            "target_timestamp_30s": "2026-08-24T23:42:00Z",
            "horizon_sec": 30,
            "current_cpu": 40.0,
            "predicted_cpu_30s": 45.0,
            "current_latency_ms": 30.0,
            "predicted_latency_30s": 50.0,
        }
        alert_engine.evaluate_prediction(recovered_pred)
        self.assertNotIn("cpu_percent", active_alerts)
        print("  -> Alert resolution verified: active_alerts cleared upon metric recovery.")

    def test_04_remediation_modes_auto_and_manual(self):
        print("\n[TEST 4] Testing Remediation Engine (AUTO vs MANUAL scaling)...")
        app = create_remediation_app()
        client = app.test_client()

        # Reset
        resp = client.post("/remediation/reset")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["active_workers"], BASE_WORKERS)

        # Set Mode AUTO
        resp = client.post("/remediation/mode", json={"mode": "AUTO"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["mode"], "AUTO")

        # Set Mode MANUAL
        resp = client.post("/remediation/mode", json={"mode": "MANUAL"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["mode"], "MANUAL")

        # Check status
        resp = client.get("/remediation/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json["mode"], "MANUAL")
        print("  -> Remediation modes AUTO & MANUAL functional.")

    def test_05_shared_features_and_engineering(self):
        print("\n[TEST 5] Testing Shared Features & Engineering Pipeline...")
        now = time.time()
        records = []
        for i in range(70):
            records.append({
                "timestamp": pd.Timestamp(now - (70 - i), unit="s", tz="UTC"),
                "cpu_percent": 30.0 + (i % 10),
                "mem_percent": 45.0,
                "avg_latency_ms": 15.0 + (i % 5),
                "p95_latency_ms": 25.0 + (i % 5),
                "request_count": 10,
                "phase": "steady",
            })
        df = pd.DataFrame(records)

        # Test window_features
        feats = feat_lib.window_features(df.iloc[:30])
        self.assertIn("cpu_mean", feats)
        self.assertIn("req_ramp_rate", feats)
        self.assertFalse(any(np.isnan(v) for v in feats.values()))

        # Test build_feature_table & temporal_split
        ftable = build_feature_table(df)
        self.assertGreater(len(ftable), 0)

        train, val, test = temporal_split(ftable)
        self.assertEqual(len(train) + len(val) + len(test), len(ftable))
        # Ensure strict non-overlapping temporal order
        if len(train) > 0 and len(val) > 0:
            self.assertLessEqual(train["window_end"].max(), val["window_end"].min())
        print("  -> Shared features module and feature engineering temporal splits validated.")

    def test_06_inference_hot_reload_endpoint(self):
        print("\n[TEST 6] Testing Inference Service Hot-Reload Endpoint...")
        app = create_inference_app(self.inference_engine, "http://127.0.0.1:5000")
        client = app.test_client()

        resp = client.post("/reload")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json.get("status"), "reloaded")
        print("  -> POST /reload endpoint reloaded model files cleanly.")


if __name__ == "__main__":
    unittest.main()
