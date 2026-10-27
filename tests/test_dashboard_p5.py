"""
tests/test_dashboard_p5.py
==========================
Unit and integration tests for P5 Scenario Launcher:
- Profile validation (step, ramp, spike, double_burst, random_walk, random_mix)
- Refusal to start if another run is active (HTTP 409)
- Stop controls and status reporting
- Route rendering for /experiments
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_dashboard.app import create_app


class TestDashboardP5Launcher(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_experiments_page_renders_200(self):
        resp = self.client.get("/experiments")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        for profile in ["step", "ramp", "spike", "double_burst", "random_walk", "random_mix"]:
            self.assertIn(f"launchProfile('{profile}')", html)
        self.assertIn("Stop Run", html)
        self.assertIn("STATUS: IDLE", html)

    def test_launcher_status_initial_idle(self):
        resp = self.client.get("/api/launcher/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "idle")
        self.assertIn("step", data["valid_profiles"])
        self.assertIn("random_mix", data["valid_profiles"])

    def test_launcher_rejects_invalid_profile(self):
        resp = self.client.post("/api/launcher/start", json={"profile": "invalid_profile"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_launcher_start_and_conflict_refusal(self, mock_popen, mock_get):
        # Mock collector health check 200
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        # Mock active process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # None indicates still running
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc

        # First start should succeed
        resp = self.client.post("/api/launcher/start", json={"profile": "step", "seed": 42, "duration": 10})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["profile"], "step")
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["pid"], 99999)

        # Status should show running
        s_resp = self.client.get("/api/launcher/status")
        s_data = s_resp.get_json()
        self.assertEqual(s_data["status"], "running")
        self.assertEqual(s_data["pid"], 99999)

        # Second start while running must return 409 Conflict
        conflict_resp = self.client.post("/api/launcher/start", json={"profile": "ramp", "seed": 44})
        self.assertEqual(conflict_resp.status_code, 409)
        c_data = conflict_resp.get_json()
        self.assertIn("already active", c_data["error"])

        # Stop the run
        stop_resp = self.client.post("/api/launcher/stop")
        self.assertEqual(stop_resp.status_code, 200)
        stop_data = stop_resp.get_json()
        self.assertEqual(stop_data["status"], "stopped")
        mock_proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
