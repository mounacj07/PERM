"""
tests/test_manual_and_scale_down.py

Unit tests for:
1. Manual-mode controller:
   - Propose: Scale-up creates pending proposal file and logs SCALE_PROPOSED
   - Approve: Operator approval applies the proposed target
   - Dismiss: Operator dismissal clears pending and suppresses target for 20s
   - Expire: Load subsiding clears pending and logs 'proposal expired, load subsided'
   - Auto unchanged: Default auto mode scales immediately without pending files
2. Scale-down path:
   - After a burst with calm arrivals, workers fall back toward min_workers after 15s hysteresis
"""

import json
import os
import sys
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo_controller import DemoController


@pytest.fixture
def controller_env(tmp_path):
    csv_path = str(tmp_path / "events.jsonl")
    status_path = str(tmp_path / "controller_status.json")
    state_path = str(tmp_path / "controller_state.json")
    control_path = str(tmp_path / "controller_control.json")
    pending_path = str(tmp_path / "controller_pending.json")
    approval_path = str(tmp_path / "controller_approval.json")
    trace_path = str(tmp_path / "controller_trace.csv")

    ctrl = DemoController(
        mode="predictive",
        approval="auto",
        initial_workers=2,
        csv_path=csv_path,
        status_path=status_path,
        state_path=state_path,
        control_path=control_path,
        pending_path=pending_path,
        approval_path=approval_path,
        trace_path=trace_path,
    )
    return ctrl, tmp_path


def test_manual_mode_propose(controller_env):
    ctrl, tmp_path = controller_env
    ctrl.approval_mode = "manual"

    # Simulate high arrivals that would scale up from 2 to 6
    high_telem = {
        "timestamp": "2026-10-27T12:00:00Z",
        "arrivals_count": 45.0,
        "worker_cpu_percent": 18.0,
        "active_workers": 2,
        "queue_depth": 0,
        "error_count": 0,
    }

    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}), \
         patch.object(ctrl, "apply_capacity", return_value=6) as mock_apply:
        res = ctrl.evaluate_step(high_telem, now_monotonic=100.0)

        # In manual mode, scale_up is NOT immediately applied
        mock_apply.assert_not_called()
        assert ctrl.current_workers == 2

        # Pending file must be written
        assert os.path.exists(ctrl.pending_path)
        with open(ctrl.pending_path, "r", encoding="utf-8") as f:
            pending = json.load(f)
        assert pending["action"] == "scale_up"
        assert pending["from_workers"] == 2
        assert pending["to_workers"] > 2
        assert "id" in pending


def test_manual_mode_approve(controller_env):
    ctrl, tmp_path = controller_env
    ctrl.approval_mode = "manual"

    high_telem = {
        "timestamp": "2026-10-27T12:00:00Z",
        "arrivals_count": 45.0,
        "worker_cpu_percent": 18.0,
        "active_workers": 2,
        "queue_depth": 0,
        "error_count": 0,
    }

    # Step 1: Generate pending proposal
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}):
        ctrl.evaluate_step(high_telem, now_monotonic=100.0)

    with open(ctrl.pending_path, "r", encoding="utf-8") as f:
        pending = json.load(f)
    prop_id = pending["id"]
    target = pending["to_workers"]

    # Step 2: Write approval file
    with open(ctrl.approval_path, "w", encoding="utf-8") as f:
        json.dump({"id": prop_id, "decision": "approve"}, f)

    # Step 3: Run next step
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}), \
         patch.object(ctrl, "apply_capacity", return_value=target) as mock_apply:
        ctrl.evaluate_step(high_telem, now_monotonic=101.0)
        mock_apply.assert_called_once_with(target)
        assert ctrl.current_workers == target
        # Pending and approval files should be cleaned up
        assert not os.path.exists(ctrl.pending_path)


def test_manual_mode_dismiss(controller_env):
    ctrl, tmp_path = controller_env
    ctrl.approval_mode = "manual"

    high_telem = {
        "timestamp": "2026-10-27T12:00:00Z",
        "arrivals_count": 45.0,
        "worker_cpu_percent": 18.0,
        "active_workers": 2,
        "queue_depth": 0,
        "error_count": 0,
    }

    # Step 1: Generate pending proposal
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}):
        ctrl.evaluate_step(high_telem, now_monotonic=100.0)

    with open(ctrl.pending_path, "r", encoding="utf-8") as f:
        pending = json.load(f)
    prop_id = pending["id"]
    target = pending["to_workers"]

    # Step 2: Write dismissal file
    with open(ctrl.approval_path, "w", encoding="utf-8") as f:
        json.dump({"id": prop_id, "decision": "dismiss"}, f)

    # Step 3: Run step with dismissal
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}), \
         patch.object(ctrl, "apply_capacity") as mock_apply:
        ctrl.evaluate_step(high_telem, now_monotonic=101.0)
        mock_apply.assert_not_called()
        assert ctrl.current_workers == 2
        assert not os.path.exists(ctrl.pending_path)
        assert target in ctrl.dismissed_targets

    # Step 4: Immediate next tick should not re-propose same target (suppressed for 20s)
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}):
        ctrl.evaluate_step(high_telem, now_monotonic=105.0)
        assert not os.path.exists(ctrl.pending_path)


def test_manual_mode_expire(controller_env):
    ctrl, tmp_path = controller_env
    ctrl.approval_mode = "manual"

    high_telem = {
        "timestamp": "2026-10-27T12:00:00Z",
        "arrivals_count": 45.0,
        "worker_cpu_percent": 18.0,
        "active_workers": 2,
        "queue_depth": 0,
        "error_count": 0,
    }

    # Step 1: High load generates proposal
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}):
        ctrl.evaluate_step(high_telem, now_monotonic=100.0)
        assert os.path.exists(ctrl.pending_path)

    # Step 2: Load subsides back to calm (arrivals 4.0, target 2)
    calm_telem = {
        "timestamp": "2026-10-27T12:00:01Z",
        "arrivals_count": 4.0,
        "worker_cpu_percent": 5.0,
        "active_workers": 2,
        "queue_depth": 0,
        "error_count": 0,
    }
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 4.0, "upper_bound": 5.0, "horizon_sec": 20}):
        # Prime arrivals history so smoothed arrivals drops
        for s in range(5):
            ctrl.evaluate_step(calm_telem, now_monotonic=110.0 + s)

        assert not os.path.exists(ctrl.pending_path)
        assert ctrl.current_pending_proposal is None


def test_auto_mode_unchanged(controller_env):
    ctrl, tmp_path = controller_env
    assert ctrl.approval_mode == "auto"

    high_telem = {
        "timestamp": "2026-10-27T12:00:00Z",
        "arrivals_count": 45.0,
        "worker_cpu_percent": 18.0,
        "active_workers": 2,
        "queue_depth": 0,
        "error_count": 0,
    }

    # In auto mode, scale-up applies immediately without writing pending file
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 42.0, "upper_bound": 45.0, "horizon_sec": 20}), \
         patch.object(ctrl, "apply_capacity", return_value=6) as mock_apply:
        ctrl.evaluate_step(high_telem, now_monotonic=100.0)
        mock_apply.assert_called_once()
        assert ctrl.current_workers == 6
        assert not os.path.exists(ctrl.pending_path)


def test_scale_down_path(controller_env):
    ctrl, tmp_path = controller_env
    # Start with workers = 6 after a burst
    ctrl.current_workers = 6

    calm_telem = {
        "timestamp": "2026-10-27T12:00:00Z",
        "arrivals_count": 5.0,
        "worker_cpu_percent": 10.0,
        "active_workers": 6,
        "queue_depth": 0,
        "error_count": 0,
    }

    t0 = 1000.0

    # Step for 14 seconds: workers must remain 6 due to 15s hysteresis
    with patch.object(ctrl, "query_forecast", return_value={"forecast": 5.0, "upper_bound": 6.0, "horizon_sec": 20}), \
         patch.object(ctrl, "apply_capacity", return_value=2) as mock_apply:

        for sec in range(14):
            ctrl.evaluate_step(calm_telem, now_monotonic=t0 + sec)
            mock_apply.assert_not_called()
            assert ctrl.current_workers == 6

        # Step at 16 seconds (>= 15s hysteresis elapsed): scale-down must trigger
        ctrl.evaluate_step(calm_telem, now_monotonic=t0 + 16.0)
        mock_apply.assert_called_once_with(2)
        assert ctrl.current_workers == 2
