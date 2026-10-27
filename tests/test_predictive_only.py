"""
Unit tests for predictive_only policy in demo_controller.py.

Tests:
1. predictive_only ignores a high reactive_target (scales to predictive_target only).
2. predictive_only holds current workers when the forecast is None / unavailable,
   logging FORECAST_UNAVAILABLE and marking forecast_online=False.
3. hybrid behavior remains unchanged (scales to max(reactive, predictive)).
"""

import json
import logging
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from demo_controller import DemoController


@pytest.fixture
def temp_csv():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_status():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_predictive_only_ignores_high_reactive(temp_csv, temp_status):
    """
    predictive_only ignores high reactive target.
    Given: arrivals 40/s (which gives reactive target 8).
    Mock forecast returning recommended target 3.
    In predictive_only: chosen_target is 3, NOT 8!
    """
    ctrl = DemoController(
        mode="predictive",
        policy="predictive_only",
        initial_workers=2,
        csv_path=temp_csv,
        status_path=temp_status,
    )
    # Populate arrivals history with 40.0 so smoothed arrivals is 40.0 and past 10s warmup
    ctrl.arrivals_history.extend([40.0] * 12)

    record = {
        "active_workers": 2,
        "worker_cpu_percent": 12.0,  # high pool utilization
        "queue_depth": 10,
        "arrivals_count": 40.0,
        "error_count": 0,
        "timestamp": "2026-10-28T02:00:00Z",
    }

    # upper_bound = 15.0 -> ceil(15.0 / (7.89 * 0.7)) = 3 workers
    mock_fc = {
        "forecast": 12.0,
        "upper_bound": 15.0,
        "horizon_sec": 20,
    }

    with patch.object(ctrl, "query_forecast", return_value=mock_fc), \
         patch.object(ctrl, "apply_capacity", return_value=3):
        res = ctrl.evaluate_step(record, now_monotonic=100.0)

        # Reactive target is 8 due to arrivals 40 req/s
        assert res["reactive_target"] == 8
        # Predictive target is 3
        assert res["predictive_target"] == 3
        # In predictive_only, reactive target is ignored!
        assert res["chosen_target"] == 3
        assert res["trigger_rule"] == "forecast"
        assert res["policy"] == "predictive_only"


def test_predictive_only_holds_when_forecast_none(temp_csv, temp_status, caplog):
    """
    predictive_only holds current workers when the forecast is None / unavailable,
    logs FORECAST_UNAVAILABLE (max once per 10s), and sets forecast_online=False.
    """
    ctrl = DemoController(
        mode="predictive",
        policy="predictive_only",
        initial_workers=4,
        csv_path=temp_csv,
        status_path=temp_status,
    )
    ctrl.arrivals_history.extend([40.0] * 12)

    record = {
        "active_workers": 4,
        "worker_cpu_percent": 20.0,
        "queue_depth": 15,
        "arrivals_count": 40.0,
        "error_count": 0,
        "timestamp": "2026-10-28T02:00:01Z",
    }

    with caplog.at_level(logging.WARNING), \
         patch.object(ctrl, "query_forecast", return_value=None):
        res = ctrl.evaluate_step(record, now_monotonic=100.0)

        # High reactive target calculated for monitoring
        assert res["reactive_target"] == 8
        # Forecast is None -> predictive_target is 0
        assert res["predictive_target"] == 0
        # Holds current workers (4)!
        assert res["chosen_target"] == 4
        assert res["trigger_rule"] == "forecast_unavailable"
        assert ctrl.forecast_online is False

        # Logs FORECAST_UNAVAILABLE
        assert any("FORECAST_UNAVAILABLE" in msg for msg in caplog.messages)


def test_hybrid_behavior_unchanged(temp_csv, temp_status):
    """
    In hybrid mode (default), scaling_target is max(reactive_target, predictive_target).
    When forecast is None or low, high reactive target dominates.
    """
    ctrl = DemoController(
        mode="predictive",
        policy="hybrid",
        initial_workers=2,
        csv_path=temp_csv,
        status_path=temp_status,
    )
    ctrl.arrivals_history.extend([40.0] * 12)

    record = {
        "active_workers": 2,
        "worker_cpu_percent": 10.0,
        "queue_depth": 5,
        "arrivals_count": 40.0,
        "error_count": 0,
        "timestamp": "2026-10-28T02:00:00Z",
    }

    # When forecast is None, reactive dominates
    with patch.object(ctrl, "query_forecast", return_value=None), \
         patch.object(ctrl, "apply_capacity", return_value=8):
        res = ctrl.evaluate_step(record, now_monotonic=100.0)
        assert res["reactive_target"] == 8
        assert res["predictive_target"] == 0
        assert res["chosen_target"] == 8
        assert res["policy"] == "hybrid"
