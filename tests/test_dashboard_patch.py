"""
tests/test_dashboard_patch.py
==============================
Unit tests for Dashboard Patch & Ops Console Refactoring (Parts A, B, C, D):
- returncode 0 or 1 -> status "finished"; note when 1: "Finished. Requests were dropped (expected during bursts)"
- returncode 2 (or dying within 5s) -> status "failed" + log_tail
- locust missing in sys.executable -> error 500 naming interpreter & "python -m pip install locust"
- collector /health unreachable -> 503 with exact start command hint
- derived duration from burst_test.generate_schedule(profile, seed) total_dur + 5s
- /api/dashboard/health for all 5 services (collector, inference, alert, remediation, controller)
- stale flag & telemetry_age_sec when newest sample is >5s or missing
- /api/controller/start and /api/controller/stop endpoints
"""

import os
import sys
import time
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web_dashboard.app as dashboard_app
from web_dashboard.app import create_app, LAUNCHER_LOG_PATH


class TestDashboardPatch(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    # ─── A1 & A2: RETURNCODE 0 AND 1 -> FINISHED + NOTE ───
    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_returncode_1_marks_finished_with_note(self, mock_popen, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_proc = MagicMock()
        mock_proc.pid = 11111
        mock_proc.poll.return_value = None  # running at start
        mock_popen.return_value = mock_proc

        start_resp = self.client.post("/api/launcher/start", json={"profile": "spike", "seed": 42, "duration": 10})
        self.assertEqual(start_resp.status_code, 200)

        # Simulate process exiting with returncode 1 after running > 5s
        # Set start_time to 10 seconds ago
        with self.app.launcher_lock:
            self.app.launcher_state["start_time"] = time.time() - 10.0
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        status_resp = self.client.get("/api/launcher/status")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.get_json()
        self.assertEqual(data["status"], "finished")
        self.assertIn("Requests were dropped (expected during bursts)", data.get("note", ""))

    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_returncode_0_marks_finished_normal(self, mock_popen, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_proc = MagicMock()
        mock_proc.pid = 22222
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        start_resp = self.client.post("/api/launcher/start", json={"profile": "step", "seed": 42, "duration": 10})
        self.assertEqual(start_resp.status_code, 200)

        # Simulate clean exit 0 after running > 5s
        with self.app.launcher_lock:
            self.app.launcher_state["start_time"] = time.time() - 10.0
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        status_resp = self.client.get("/api/launcher/status")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.get_json()
        self.assertEqual(data["status"], "finished")

    # ─── A1 & A2: NONZERO RETURNCODE 2 OR EARLY DEATH -> FAILED + LOG_TAIL ───
    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_returncode_2_marks_failed_with_log_tail(self, mock_popen, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_proc = MagicMock()
        mock_proc.pid = 33333
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        start_resp = self.client.post("/api/launcher/start", json={"profile": "step", "seed": 42, "duration": 10})
        self.assertEqual(start_resp.status_code, 200)

        # Write sample lines to launcher log after start (since start truncates it)
        os.makedirs(os.path.dirname(LAUNCHER_LOG_PATH), exist_ok=True)
        with open(LAUNCHER_LOG_PATH, "w", encoding="utf-8") as f:
            for i in range(20):
                f.write(f"locust sample log line {i}\n")

        with self.app.launcher_lock:
            self.app.launcher_state["start_time"] = time.time() - 10.0
        # Simulate abnormal exit 2
        mock_proc.poll.return_value = 2
        mock_proc.returncode = 2

        status_resp = self.client.get("/api/launcher/status")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.get_json()
        self.assertEqual(data["status"], "failed")
        self.assertIn("log_tail", data)
        self.assertIn("message", data)
        self.assertIn("locust sample log line 19", data["log_tail"])

    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_early_exit_under_5s_marks_failed(self, mock_popen, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_proc = MagicMock()
        mock_proc.pid = 44444
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        start_resp = self.client.post("/api/launcher/start", json={"profile": "ramp", "seed": 42, "duration": 10})
        self.assertEqual(start_resp.status_code, 200)

        # Died after only 2s elapsed
        with self.app.launcher_lock:
            self.app.launcher_state["start_time"] = time.time() - 2.0
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        status_resp = self.client.get("/api/launcher/status")
        data = status_resp.get_json()
        self.assertEqual(data["status"], "failed")
        self.assertIn("died within 5s", data.get("message", ""))

    # ─── A3: PREFLIGHT CHECKS ───
    @patch("web_dashboard.app._http_session.get")
    def test_preflight_collector_unreachable_returns_503(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        resp = self.client.post("/api/launcher/start", json={"profile": "spike", "seed": 42})
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertIn("Collector not reachable", data["error"])
        self.assertIn("python telemetry_collector.py --port 5000 --workers 2", data["error"])

    @patch("web_dashboard.app._http_session.get")
    @patch("importlib.util.find_spec")
    def test_preflight_locust_missing_returns_500(self, mock_find, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_find.return_value = None  # locust not installed

        resp = self.client.post("/api/launcher/start", json={"profile": "spike", "seed": 42})
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertIn("python -m pip install locust", data["error"])
        self.assertIn(sys.executable, data["error"])

    # ─── A4: DERIVED DURATION ───
    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_derived_duration_from_burst_test(self, mock_popen, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        import burst_test
        _, _, total_dur = burst_test.generate_schedule("ramp", 44)
        expected_dur = int(total_dur + 5.0)

        # Do not supply duration parameter
        resp = self.client.post("/api/launcher/start", json={"profile": "ramp", "seed": 44})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["duration_sec"], expected_dur)

    # ─── B1: /api/dashboard/health ───
    @patch("web_dashboard.app._http_session.get")
    def test_dashboard_health_endpoint(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        resp = self.client.get("/api/dashboard/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        for svc in ["collector", "inference", "alert", "remediation", "controller"]:
            self.assertIn(svc, data)
            self.assertIn("status", data[svc])
            self.assertIn("start_cmd", data[svc])
            self.assertIn("latency_ms", data[svc])

        # Verify exact command hints
        self.assertIn("telemetry_collector.py", data["collector"]["start_cmd"])
        self.assertIn("inference_service.py", data["inference"]["start_cmd"])
        self.assertIn("alert_engine.py", data["alert"]["start_cmd"])
        self.assertIn("remediation_service.py", data["remediation"]["start_cmd"])
        self.assertIn("demo_controller.py", data["controller"]["start_cmd"])

    # ─── B3: STALE FLAG AND TELEMETRY AGE ───
    def test_stale_flag_and_telemetry_age(self):
        resp = self.client.get("/api/dashboard/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("stale", data)
        self.assertIn("telemetry_age_sec", data)
        self.assertIn("logical_cpus", data)

        # In offline testing without live collector, data should be marked stale
        self.assertTrue(data["stale"])

    # ─── C1 & C2: CONTROLLER START & STOP ───
    @patch("subprocess.Popen")
    def test_controller_start_and_stop(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.pid = 66666
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        start_resp = self.client.post("/api/controller/start", json={"mode": "capacity_aware"})
        self.assertEqual(start_resp.status_code, 200)
        data = start_resp.get_json()
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["mode"], "capacity_aware")
        self.assertEqual(data["pid"], 66666)

        # Conflict when starting while already running
        conflict_resp = self.client.post("/api/controller/start", json={"mode": "reactive"})
        self.assertEqual(conflict_resp.status_code, 409)

        # Stop controller
        stop_resp = self.client.post("/api/controller/stop")
        self.assertEqual(stop_resp.status_code, 200)
        s_data = stop_resp.get_json()
        self.assertEqual(s_data["status"], "stopped")
        mock_proc.terminate.assert_called_once()

    # ─── F1: TIMEZONE STALENESS UNDER NON-UTC TIMEZONE ───
    @patch("web_dashboard.app._http_session.get")
    def test_timezone_staleness_non_utc(self, mock_get):
        from datetime import datetime, timezone, timedelta

        # Simulate local machine in UTC+5:30
        tz_plus530 = timezone(timedelta(hours=5, minutes=30))
        now_local = datetime.now(tz_plus530)

        # 1. Fresh sample (1.5 seconds old in local TZ)
        fresh_ts = (now_local - timedelta(seconds=1.5)).isoformat()
        sample_metrics = [
            {"timestamp": fresh_ts, "arrival_rate_req_per_s": 0.0, "active_workers": 2, "worker_cpu_percent": 10.0}
        ]

        def side_effect(url, **kwargs):
            r = MagicMock()
            r.status_code = 200
            if "/metrics/history" in url:
                r.json.return_value = {"count": len(sample_metrics), "records": sample_metrics, "history": sample_metrics}
            elif "/metrics" in url or "/capacity" in url:
                r.json.return_value = {}
            else:
                r.json.return_value = {}
            return r

        mock_get.side_effect = side_effect

        resp = self.client.get("/api/dashboard/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Telemetry age should be around 1.5s, definitely NOT ~19800s
        self.assertIsNotNone(data["telemetry_age_sec"])
        self.assertLess(data["telemetry_age_sec"], 5.0)
        self.assertFalse(data["stale"])

        # 2. Stale sample (12.0 seconds old in local TZ)
        stale_ts = (now_local - timedelta(seconds=12.0)).isoformat()
        sample_metrics[0]["timestamp"] = stale_ts
        resp2 = self.client.get("/api/dashboard/state")
        data2 = resp2.get_json()
        self.assertGreater(data2["telemetry_age_sec"], 5.0)
        self.assertTrue(data2["stale"])

    # ─── F1: HISTORY TIMEOUT FALLBACK ───
    @patch("web_dashboard.app._safe_get")
    def test_history_timeout_fallback(self, mock_safe_get):
        # Seed _cached_state with valid prior history
        prior_history = [
            {"timestamp": "2026-10-27T12:00:00Z", "arrival_rate_req_per_s": 5.0, "active_workers": 2}
        ]
        dashboard_app._cached_state["telemetry_history"] = prior_history

        # Simulate timeout on /metrics/history (returns None)
        def mock_fetch(url, timeout=None):
            if "/metrics/history" in url:
                return None  # timeout / fetch error
            return {}

        mock_safe_get.side_effect = mock_fetch

        resp = self.client.get("/api/dashboard/state")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Verify cached history is preserved rather than blanked
        self.assertEqual(data["telemetry_history"], prior_history)
        self.assertEqual(data["history"], prior_history)
        self.assertTrue(data["stale"])

    # ─── F2: REQUIRED VS OPTIONAL PILLS HEALTH ───
    @patch("web_dashboard.app._http_session.get")
    def test_required_vs_optional_pill_state(self, mock_get):
        # Simulate collector and controller online, alert and remediation offline
        def side_effect(url, **kwargs):
            r = MagicMock()
            if "5000" in url:  # collector
                r.status_code = 200
            else:
                r.status_code = 503
            return r

        mock_get.side_effect = side_effect

        resp = self.client.get("/api/dashboard/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertEqual(data["collector"]["status"], "online")
        self.assertEqual(data["alert"]["status"], "offline")
        self.assertEqual(data["remediation"]["status"], "offline")

    # ─── F4: ELAPSED SEC RECORDED AT PROCESS EXIT AND CAPPED AT DURATION ───
    @patch("web_dashboard.app._http_session.get")
    @patch("subprocess.Popen")
    def test_elapsed_sec_capped_at_duration_on_exit(self, mock_popen, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        start_resp = self.client.post("/api/launcher/start", json={"profile": "spike", "seed": 42, "duration": 80})
        self.assertEqual(start_resp.status_code, 200)

        # Simulate process exiting cleanly after 80s run, but queried at 143.4s
        with self.app.launcher_lock:
            self.app.launcher_state["start_time"] = time.time() - 143.4
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        status_resp = self.client.get("/api/launcher/status")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.get_json()

        self.assertEqual(data["status"], "finished")
        # Elapsed sec must be capped at duration_sec (80.0s), not 143.4s!
        self.assertEqual(data["elapsed_sec"], 80.0)


if __name__ == "__main__":
    unittest.main()
