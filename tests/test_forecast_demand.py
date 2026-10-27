"""
tests/test_forecast_demand.py
==============================
Unit tests for inference_service optional endpoint /forecast/demand
controlled by PERM_FORECASTER=stagbdt.
"""

import os
import unittest
from unittest.mock import patch
import numpy as np

from inference_service import InferenceEngine, create_app


class TestForecastDemandEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        cls.engine = InferenceEngine(cls.model_dir)

    def test_default_behavior_endpoint_not_present(self):
        """When PERM_FORECASTER is not set, /forecast/demand is not registered."""
        with patch.dict(os.environ, {"PERM_FORECASTER": ""}):
            app = create_app(self.engine, "http://127.0.0.1:5000")
            client = app.test_client()

            # Existing health endpoint operational
            r_health = client.get("/health")
            self.assertEqual(r_health.status_code, 200)

            # /forecast/demand endpoint not registered
            r_fc = client.post("/forecast/demand", json={"arrivals": [10.0] * 30})
            self.assertEqual(r_fc.status_code, 404)

    def test_stagbdt_forecaster_endpoint_active(self):
        """When PERM_FORECASTER=stagbdt, /forecast/demand returns forecast, upper bound, recommended workers."""
        with patch.dict(os.environ, {"PERM_FORECASTER": "stagbdt"}):
            app = create_app(self.engine, "http://127.0.0.1:5000")
            client = app.test_client()

            # Test POST with synthetic arrivals array
            arrivals = [5.0] * 20 + [25.0] * 10
            resp = client.post("/forecast/demand?horizon=20", json={"arrivals": arrivals})
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()

            self.assertEqual(data["forecaster"], "stagbdt")
            self.assertEqual(data["horizon_sec"], 20)
            self.assertIn("forecast", data)
            self.assertIn("upper_bound", data)
            self.assertIn("recommended_workers", data)
            self.assertIn("conformal_margin", data)

            self.assertGreaterEqual(data["upper_bound"], data["forecast"])
            self.assertGreaterEqual(data["recommended_workers"], 2)
            self.assertLessEqual(data["recommended_workers"], 8)

    def test_stagbdt_forecaster_horizons(self):
        """Verify /forecast/demand supports horizons h=10, 20, 30."""
        with patch.dict(os.environ, {"PERM_FORECASTER": "stagbdt"}):
            app = create_app(self.engine, "http://127.0.0.1:5000")
            client = app.test_client()

            arrivals = [12.0] * 30
            for h in [10, 20, 30]:
                resp = client.post(f"/forecast/demand?horizon={h}", json={"arrivals": arrivals})
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertEqual(data["horizon_sec"], h)
                self.assertGreaterEqual(data["upper_bound"], data["forecast"])


if __name__ == "__main__":
    unittest.main()
