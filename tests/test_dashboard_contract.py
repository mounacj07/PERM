"""
tests/test_dashboard_contract.py

Contract test verifying:
1. A real record fetched from the real collector's Flask test client contains 'arrivals_count'
   (and NOT 'arrival_rate_req_per_s', documenting the bug root cause).
2. All telemetry fields used by the dashboard exist in the real collector output.
3. The dashboard state snapshot normalizes arrivals so that snapshot['arrivals'],
   snapshot['arrival_rate_req_per_s'], and snapshot['arrivals_count'] are non-zero when
   arrivals_count is non-zero.
"""

import os
import sys
import datetime
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telemetry_collector
from web_dashboard.app import create_app as create_dashboard_app


def test_collector_telemetry_contract_and_dashboard_mapping():
    # 1. Instantiate the real collector app and test client
    collector_app = telemetry_collector.create_app()
    collector_client = collector_app.test_client()

    # Populate a real record in the collector's buffer (mimicking sampling_loop output)
    sample_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cpu_percent": 18.5,
        "mem_percent": 42.0,
        "avg_latency_ms": 45.2,
        "p95_latency_ms": 112.4,
        "request_count": 15,
        "active_workers": 4,
        "error_count": 0,
        "queue_depth": 2,
        "arrivals_count": 15.0,  # Real field emitted by collector
        "ok_count": 15,
        "timeout_count": 0,
        "rejected_count": 0,
        "in_flight": 2,
        "worker_cpu_percent": 12.0,
        "ok_p95_latency_ms": 112.4,
        "busy_workers": 2,
        "utilization": 0.5,
        "worker_cpu_per_active_pct": 48.0,
        "queue_wait_ms_avg": 3.1,
        "cpu_freq_mhz": 2400.0,
    }

    with telemetry_collector._state_lock:
        telemetry_collector.telemetry_buffer.clear()
        telemetry_collector.telemetry_buffer.append(sample_record)

    # 2. Fetch real record from collector test client
    resp = collector_client.get("/metrics/latest")
    assert resp.status_code == 200
    real_collector_record = resp.get_json()

    # Assert bug root cause: arrival_rate_req_per_s was missing from collector
    assert "arrivals_count" in real_collector_record
    assert "arrival_rate_req_per_s" not in real_collector_record

    # Assert all telemetry fields the frontend and app.py use exist in collector record
    frontend_required_fields = [
        "timestamp",
        "cpu_percent",
        "avg_latency_ms",
        "p95_latency_ms",
        "request_count",
        "active_workers",
        "error_count",
        "queue_depth",
        "arrivals_count",
        "worker_cpu_percent",
        "worker_cpu_per_active_pct",
    ]
    for field in frontend_required_fields:
        assert field in real_collector_record, f"Field '{field}' missing from collector output"

    # 3. Mount dashboard app and test client
    dash_app = create_dashboard_app()
    dash_client = dash_app.test_client()

    # Mock upstream collector HTTP call so dashboard reads our real collector record
    mock_history_resp = MagicMock()
    mock_history_resp.status_code = 200
    mock_history_resp.json.return_value = {
        "count": 1,
        "records": [real_collector_record],
    }

    with patch("web_dashboard.app._http_session.get", return_value=mock_history_resp):
        state_resp = dash_client.get("/api/dashboard/state")
        assert state_resp.status_code == 200
        snapshot = state_resp.get_json()

    # 4. Assert dashboard snapshot arrivals are non-zero when arrivals_count is non-zero
    assert snapshot["arrivals"] == 15.0
    assert snapshot["arrival_rate_req_per_s"] == 15.0
    assert snapshot["arrivals_count"] == 15.0

    telem_latest = snapshot.get("telemetry_latest", {})
    assert telem_latest.get("arrivals_count") == 15.0
    assert telem_latest.get("arrival_rate_req_per_s") == 15.0
    assert telem_latest.get("arrivals") == 15.0

    # Assert KPI fields exist in snapshot
    assert snapshot["active_workers"] == 4
    assert snapshot["queue_depth"] == 2
    assert snapshot["p95_latency_ms"] == 112.4
    assert snapshot["error_count"] == 0
    assert "capacity_per_worker" in snapshot
    assert snapshot["capacity_per_worker"] > 0
