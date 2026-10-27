"""
tests/test_dashboard_routes.py
==============================
Comprehensive integration tests verifying:
- Every HTML page returns 200 (Overview, CPU, Latency, Forecast, Alerts, Experiments)
- Every JSON endpoint returns 200 and expected fields
- Old endpoints remain backwards compatible with all original JSON fields intact
- Replay works end-to-end with a recorded run from data/runs_locust
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_dashboard.app import create_app


class TestDashboardAllRoutes(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_all_html_pages_return_200(self):
        routes = ["/", "/overview", "/cpu", "/latency", "/forecast", "/alerts", "/scaling", "/experiments"]
        for route in routes:
            with self.subTest(route=route):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200, f"Route {route} failed with status {resp.status_code}")
                self.assertIn("<!DOCTYPE html>", resp.get_data(as_text=True))

    def test_dashboard_state_endpoint_fields(self):
        resp = self.client.get("/api/dashboard/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Original / backwards compatible top-level snapshot keys
        expected_original = [
            "telemetry_history", "telemetry_latest", "prediction_history",
            "prediction_latest", "alert_status", "active_alerts", "alerts_history",
            "remediation", "capacity"
        ]
        for field in expected_original:
            self.assertIn(field, data, f"Missing original field '{field}' in /api/dashboard/state")

        # Newly enriched fields for P1, P3, and D4
        expected_new = [
            "worker_cpu_pct_active", "whole_machine_cpu", "slo_latency_ms",
            "pool_utilization", "controller_events", "controller_state", "model_mode_label"
        ]
        for field in expected_new:
            self.assertIn(field, data, f"Missing added field '{field}' in /api/dashboard/state")

    def test_forecast_runs_endpoint(self):
        resp = self.client.get("/api/forecast/runs")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("runs", data)
        self.assertIsInstance(data["runs"], list)
        self.assertGreater(len(data["runs"]), 0)

        # Check no sawtooth runs included
        run_ids = [r["run_id"] for r in data["runs"]]
        for r_id in run_ids:
            self.assertFalse(r_id.startswith("sawtooth"), f"Sawtooth run '{r_id}' should be excluded")

    def test_forecast_replay_with_recorded_run(self):
        resp = self.client.get("/api/forecast/runs")
        run_id = resp.get_json()["runs"][0]["run_id"]

        replay_resp = self.client.get(f"/api/forecast/replay?run_id={run_id}")
        self.assertEqual(replay_resp.status_code, 200)
        r_data = replay_resp.get_json()

        self.assertIn("run_id", r_data)
        self.assertIn("records", r_data)
        self.assertIn("summary", r_data)
        self.assertIn("is_training", r_data)
        self.assertIn("banner", r_data)
        self.assertGreater(len(r_data["records"]), 0)

        # Verify record schema contains demand, forecast, conformal bound, recommended & required workers
        sample = r_data["records"][0]
        self.assertIn("second", sample)
        self.assertIn("current_arrivals", sample)
        self.assertIn("raw_forecast", sample)
        self.assertIn("upper_bound", sample)
        self.assertIn("recommended_workers", sample)
        self.assertIn("actual_workers_needed", sample)

    def test_forecast_live_offline_state(self):
        resp = self.client.get("/api/forecast/live")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("message", data)
        if data["status"] == "offline":
            self.assertIn("forecast service offline", data["message"].lower())

    def test_launcher_status_endpoint(self):
        resp = self.client.get("/api/launcher/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("valid_profiles", data)


if __name__ == "__main__":
    unittest.main()
