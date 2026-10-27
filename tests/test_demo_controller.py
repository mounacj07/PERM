"""
Unit tests for demo_controller.py
Mock inputs; no Locust or live services needed.

Tests:
1. 2 workers with arrivals 40/s give a target of 8
2. Utilization above 85% for 2 s adds 2 workers
3. Scale-down hysteresis works (requires 15 s of target below current workers)
4. Forecast-offline fallback works (logs warning and falls back to reactive)
5. An alert is written when pool utilization crosses 70%
"""

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
def controller(temp_csv):
    return DemoController(
        mode="predictive",
        initial_workers=2,
        csv_path=temp_csv,
    )


def test_target_at_40_arrivals(controller):
    """
    1. 2 workers with arrivals 40/s give a target of 8.
    service_time = 0.126 s.
    ceil(40 * 0.126 / 0.7) = ceil(7.2) = 8.
    """
    # Simulate high utilization and 40 req/s arrivals
    record = {
        "active_workers": 2,
        "worker_cpu_percent": 10.0,  # 10% * 16 / 2 = 80% pool utilization
        "queue_depth": 3,
        "arrivals_count": 40.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:00Z",
    }
    # Mock forecast endpoint to return 0 so reactive scaling dominates
    with patch.object(controller, "query_forecast", return_value=None):
        res = controller.evaluate_step(record, now_monotonic=100.0)
        assert res["reactive_target"] == 8
        assert res["chosen_target"] == 8


def test_utilization_above_85_for_2s_adds_2_workers(controller):
    """
    2. Utilization above 85% for 2 s adds 2 workers.
    """
    # Step 1: utilization > 85%, arrivals 15/s -> base target = ceil(15 * 0.126 / 0.7) = 3
    # worker_cpu_percent = 12.0 -> 12 * 16 / 2 = 96% pool util
    record1 = {
        "active_workers": 2,
        "worker_cpu_percent": 12.0,
        "queue_depth": 6,
        "arrivals_count": 15.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:01Z",
    }
    with patch.object(controller, "query_forecast", return_value=None), \
         patch.object(controller, "apply_capacity", return_value=3):
        res1 = controller.evaluate_step(record1, now_monotonic=101.0)
        # First second above 85%: base target is 3 (max(2, 3)), not yet 2 consecutive seconds
        assert res1["consecutive_above_85"] == 1 if hasattr(res1, "consecutive_above_85") else True
        assert controller.consecutive_above_85 == 1
        assert res1["chosen_target"] == 3

    # Step 2: second second above 85%
    record2 = {
        "active_workers": 3,
        "worker_cpu_percent": 18.0,  # 18 * 16 / 3 = 96% pool util
        "queue_depth": 7,
        "arrivals_count": 15.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:02Z",
    }
    with patch.object(controller, "query_forecast", return_value=None), \
         patch.object(controller, "apply_capacity", return_value=5):
        res2 = controller.evaluate_step(record2, now_monotonic=102.0)
        assert controller.consecutive_above_85 == 2
        # Base target is 3; safety rule adds 2 -> target = 5
        assert res2["chosen_target"] == 5
        assert res2["trigger_rule"] == "safety"


def test_scale_down_hysteresis(controller):
    """
    3. Scale-down hysteresis works:
    Target stays below current count for 15 s before scaling down.
    """
    controller.current_workers = 6

    # Low load: arrivals = 2/s, util = 10%
    record_low = {
        "active_workers": 6,
        "worker_cpu_percent": 1.0,
        "queue_depth": 0,
        "arrivals_count": 2.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:00Z",
    }

    with patch.object(controller, "query_forecast", return_value=None), \
         patch.object(controller, "apply_capacity") as mock_apply:
        
        # t = 0s: target is 2 (< 6). Timer starts, no scale down yet.
        res0 = controller.evaluate_step(record_low, now_monotonic=100.0)
        assert res0["chosen_target"] == 2
        assert res0["scale_event"] is None
        mock_apply.assert_not_called()

        # t = 10s: target is 2. Timer elapsed = 10s < 15s. No scale down yet.
        res10 = controller.evaluate_step(record_low, now_monotonic=110.0)
        assert res10["chosen_target"] == 2
        assert res10["scale_event"] is None
        mock_apply.assert_not_called()

        # t = 14.9s: still not scaled down
        res14 = controller.evaluate_step(record_low, now_monotonic=114.9)
        assert res14["chosen_target"] == 2
        assert res14["scale_event"] is None
        mock_apply.assert_not_called()

        # t = 15.1s: elapsed >= 15 s -> scale down triggers!
        mock_apply.return_value = 2
        res15 = controller.evaluate_step(record_low, now_monotonic=115.1)
        assert res15["chosen_target"] == 2
        assert res15["scale_event"] is not None
        assert res15["scale_event"]["action"] == "scale_down"
        assert res15["scale_event"]["to_workers"] == 2
        mock_apply.assert_called_once_with(2)


def test_forecast_offline_fallback(controller, caplog):
    """
    4. Forecast-offline fallback works:
    When /forecast/demand is offline (and past warmup), fall back to reactive and log 'forecast offline'.
    """
    # Prime history with 10 samples to complete warmup
    controller.arrivals_history.extend([25.0] * 10)

    record = {
        "active_workers": 2,
        "worker_cpu_percent": 10.0,
        "queue_depth": 5,
        "arrivals_count": 25.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:00Z",
    }

    # Simulate forecast endpoint failure (returns None)
    with patch.object(controller, "query_forecast", return_value=None), \
         patch.object(controller, "apply_capacity", return_value=5):
        res = controller.evaluate_step(record, now_monotonic=100.0)
        # Should fall back to reactive target ceil(25 * 0.126 / 0.7) = ceil(4.5) = 5
        assert res["predictive_target"] == 0
        assert res["reactive_target"] == 5
        assert res["chosen_target"] == 5
        # Verify 'forecast offline' is logged
        assert any("forecast offline" in record.message for record in caplog.records)


def test_alert_written_at_70_utilization(controller, temp_csv):
    """
    5. An alert is written when utilization crosses 70%.
    Also test CRITICAL at 85% and RECOVERY below 70%.
    """
    # 1. Util at 50% (Normal)
    record_50 = {
        "active_workers": 2,
        "worker_cpu_percent": 6.25,  # 6.25 * 16 / 2 = 50%
        "queue_depth": 0,
        "arrivals_count": 5.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:00Z",
    }
    with patch.object(controller, "query_forecast", return_value=None):
        controller.evaluate_step(record_50, now_monotonic=100.0)
        assert controller.alert_state == "NORMAL"

    # 2. Util crosses 70% -> WARN
    record_75 = {
        "active_workers": 2,
        "worker_cpu_percent": 9.375,  # 9.375 * 16 / 2 = 75%
        "queue_depth": 0,
        "arrivals_count": 10.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:01Z",
    }
    with patch.object(controller, "query_forecast", return_value=None):
        # 3 samples to smooth
        controller.evaluate_step(record_75, now_monotonic=101.0)
        controller.evaluate_step(record_75, now_monotonic=102.0)
        res_warn = controller.evaluate_step(record_75, now_monotonic=103.0)
        assert controller.alert_state == "WARN"

    # Check that WARN event was written to CSV
    with open(temp_csv, "r", encoding="utf-8") as f:
        content = f.read()
    assert "WARN,pool_utilization" in content

    # 3. Util crosses 85% -> CRITICAL
    record_90 = {
        "active_workers": 2,
        "worker_cpu_percent": 11.25,  # 11.25 * 16 / 2 = 90%
        "queue_depth": 2,
        "arrivals_count": 12.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:04Z",
    }
    with patch.object(controller, "query_forecast", return_value=None):
        controller.evaluate_step(record_90, now_monotonic=104.0)
        controller.evaluate_step(record_90, now_monotonic=105.0)
        controller.evaluate_step(record_90, now_monotonic=106.0)
        assert controller.alert_state == "CRITICAL"

    with open(temp_csv, "r", encoding="utf-8") as f:
        content = f.read()
    assert "CRITICAL,pool_utilization" in content

    # 4. Util falls back below 70% -> RECOVERY
    with patch.object(controller, "query_forecast", return_value=None):
        controller.evaluate_step(record_50, now_monotonic=107.0)
        controller.evaluate_step(record_50, now_monotonic=108.0)
        controller.evaluate_step(record_50, now_monotonic=109.0)
        assert controller.alert_state == "NORMAL"

    with open(temp_csv, "r", encoding="utf-8") as f:
        content = f.read()
    assert "RECOVERY,pool_utilization" in content


def test_predictive_warmup_3s_history_not_used(controller, caplog):
    """
    With 3 s of history, the predictive path is not used:
    - query_forecast is NOT called
    - 'predictive warming up' is logged
    - only reactive logic is used
    """
    caplog.set_level(logging.INFO)
    record = {
        "active_workers": 2,
        "worker_cpu_percent": 1.0,
        "queue_depth": 0,
        "arrivals_count": 5.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:00Z",
    }
    with patch.object(controller, "query_forecast") as mock_forecast:
        # Step 1 (history length 1)
        res1 = controller.evaluate_step(record, now_monotonic=100.0)
        assert res1["predictive_target"] == 0
        mock_forecast.assert_not_called()

        # Step 2 (history length 2)
        res2 = controller.evaluate_step(record, now_monotonic=101.0)
        assert res2["predictive_target"] == 0
        mock_forecast.assert_not_called()

        # Step 3 (history length 3)
        res3 = controller.evaluate_step(record, now_monotonic=102.0)
        assert res3["predictive_target"] == 0
        mock_forecast.assert_not_called()

        # Verify predictive path was not used and log message emitted
        assert len(controller.arrivals_history) == 3
        assert any("predictive warming up" in r.message for r in caplog.records)
        assert res3["trigger_rule"] in ("reactive", "hold")
        assert res3["chosen_target"] == 2


def test_capacity_aware_mode(temp_csv):
    """
    capacity_aware mode:
    - computes ceil(smoothed_arrivals * service_time / 0.7) directly from arrivals
    - NO utilization threshold required to scale up
    - scale-up is immediate
    - scale-down respects 15s hysteresis
    """
    cap_ctrl = DemoController(
        mode="capacity_aware",
        initial_workers=2,
        csv_path=temp_csv,
    )
    assert cap_ctrl.mode == "capacity_aware"

    # Step 1: High arrivals (40 req/s) with 0% CPU and 0 queue depth
    # Reactive mode would not scale up because util < 70% and queue < 5.
    # But capacity_aware mode scales up immediately!
    record_high = {
        "active_workers": 2,
        "worker_cpu_percent": 0.0,  # 0% CPU!
        "queue_depth": 0,          # 0 queue!
        "arrivals_count": 40.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:00Z",
    }
    with patch.object(cap_ctrl, "query_forecast") as mock_forecast, \
         patch.object(cap_ctrl, "apply_capacity", return_value=8) as mock_apply:
        res = cap_ctrl.evaluate_step(record_high, now_monotonic=100.0)
        mock_forecast.assert_not_called()
        # ceil(40 * 0.126 / 0.7) = 8
        assert res["chosen_target"] == 8
        assert res["trigger_rule"] == "capacity_aware"
        assert res["scale_event"] is not None
        assert res["scale_event"]["to_workers"] == 8
        mock_apply.assert_called_once_with(8)

    # Step 2: Low arrivals (4 req/s), currently at 8 workers -> scale down hysteresis
    cap_ctrl.current_workers = 8
    cap_ctrl.arrivals_history.clear()
    record_low = {
        "active_workers": 8,
        "worker_cpu_percent": 0.0,
        "queue_depth": 0,
        "arrivals_count": 4.0,
        "error_count": 0,
        "timestamp": "2026-10-27T12:00:01Z",
    }
    with patch.object(cap_ctrl, "apply_capacity") as mock_apply:
        # t = 0s: target = 2 (< 8). Hysteresis begins, no scale down yet.
        r0 = cap_ctrl.evaluate_step(record_low, now_monotonic=200.0)
        assert r0["chosen_target"] == 2
        assert r0["scale_event"] is None
        mock_apply.assert_not_called()

        # t = 10s: still within 15s hysteresis
        r10 = cap_ctrl.evaluate_step(record_low, now_monotonic=210.0)
        assert r10["chosen_target"] == 2
        assert r10["scale_event"] is None
        mock_apply.assert_not_called()

        # t = 15.5s: elapsed >= 15s -> scale down triggers!
        mock_apply.return_value = 2
        r15 = cap_ctrl.evaluate_step(record_low, now_monotonic=215.5)
        assert r15["chosen_target"] == 2
        assert r15["scale_event"] is not None
        assert r15["scale_event"]["action"] == "scale_down"
        assert r15["scale_event"]["to_workers"] == 2
        mock_apply.assert_called_once_with(2)

