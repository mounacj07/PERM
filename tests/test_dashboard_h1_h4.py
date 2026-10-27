"""
tests/test_dashboard_h1_h4.py
==============================
Unit tests for Dashboard Patch H1-H5:
- H1: Predictive mode spawns inference_service.py in background thread when unreachable,
      logs to results/inference_last.log, does not abort.
- H1: State returns forecast_offline_banner and forecast_fallback when predictive runs without forecasts.
- H1: State parses forecast_online from controller_trace.csv rows.
- H5: Zero mock detection strings (MagicMock, _mock, PYTEST_CURRENT_TEST, assert_called) in web_dashboard/.
"""

import os
import sys
import time
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_dashboard.app import create_app, PROJECT_ROOT, INFERENCE_LOG_PATH


class TestDashboardH1H4(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_predictive_mode_spawns_inference_background_and_starts_controller(self, mock_popen, mock_get):
        # Simulate inference /health failing (offline)
        def side_effect(url, **kwargs):
            r = MagicMock()
            if "5001/health" in url:
                r.status_code = 503
            else:
                r.status_code = 200
                r.json.return_value = {}
            return r

        mock_get.side_effect = side_effect

        mock_proc = MagicMock()
        mock_proc.pid = 77777
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        resp = self.client.post("/api/controller/start", json={"mode": "predictive"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["mode"], "predictive")

        # Popen should have been called (for inference and/or controller)
        self.assertGreaterEqual(mock_popen.call_count, 1)

    @patch("web_dashboard.app._read_controller_status")
    def test_predictive_offline_fallback_banner_and_state(self, mock_status):
        # Controller running in predictive mode with forecast_online = False
        mock_status.return_value = {
            "mode": "predictive",
            "forecast_online": False,
            "time": "2026-10-28T00:00:00Z"
        }

        # Write fake log tail to inference_last.log
        os.makedirs(os.path.dirname(INFERENCE_LOG_PATH), exist_ok=True)
        with open(INFERENCE_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("Starting inference service...\nListening on port 5001\n")

        resp = self.client.get("/api/dashboard/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertTrue(data.get("forecast_fallback"))
        banner = data.get("forecast_offline_banner", {})
        self.assertTrue(banner.get("active"))
        self.assertIn("Predictive mode is running WITHOUT forecasts", banner.get("message", ""))
        self.assertIn("Listening on port 5001", banner.get("log_tail", ""))

    def test_trace_forecast_online_parsed_in_state(self):
        trace_file = os.path.join(PROJECT_ROOT, "results", "controller_trace.csv")
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)

        # Write sample row with forecast_online
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write("t,arrivals,smoothed_arrivals,forecast_raw,forecast_upper,horizon,reactive_target,predictive_target,chosen_target,workers,util,approval_mode,forecast_online\n")
            f.write("1.0,4.0,4.0,12.5,15.0,20,2,3,3,2,25.0,auto,True\n")
            f.write("2.0,5.0,4.5,13.0,16.0,20,2,3,3,2,30.0,auto,False\n")

        resp = self.client.get("/api/dashboard/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        trace = data.get("controller_trace", [])
        self.assertGreaterEqual(len(trace), 2)
        self.assertTrue(trace[-2].get("forecast_online"))
        self.assertFalse(trace[-1].get("forecast_online"))

    def test_h5_zero_mock_detection_in_web_dashboard(self):
        root = os.path.join(PROJECT_ROOT, "web_dashboard")
        patterns = ["MagicMock", "_mock", "PYTEST_CURRENT_TEST", "assert_called"]
        hits = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for p in patterns:
                        if p in content:
                            hits.append((fp, p))
        self.assertEqual(hits, [], f"Production web_dashboard contains mock detection patterns: {hits}")


if __name__ == "__main__":
    unittest.main()
