"""
Unit tests for Dashboard Priority 1 (P1): Navigation, Overview, CPU, and Latency pages.
"""

import pytest
from web_dashboard.app import create_app, compute_worker_cpu_active


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_worker_cpu_calculation_exact_spec():
    """
    P1 requirement:
    Add a test: 8 active workers with worker_cpu_percent 50 gives 100;
                4 active workers with worker_cpu_percent 12.5 gives 50.
    """
    res1 = compute_worker_cpu_active(worker_cpu_percent=50.0, active_workers=8, cpu_count=16)
    assert res1 == 100.0, f"Expected 100.0, got {res1}"

    res2 = compute_worker_cpu_active(worker_cpu_percent=12.5, active_workers=4, cpu_count=16)
    assert res2 == 50.0, f"Expected 50.0, got {res2}"


def test_worker_cpu_calculation_mocked_8_cpus():
    """
    With 8 logical CPUs mocked, 8 active workers at worker_cpu_percent 100 gives 100.
    Keeps 16-CPU calculation test intact above.
    """
    from unittest.mock import patch
    with patch("web_dashboard.app.get_system_logical_cpus", return_value=8):
        res = compute_worker_cpu_active(worker_cpu_percent=100.0, active_workers=8)
        assert res == 100.0, f"Expected 100.0, got {res}"


def test_routes_return_200(client):
    """
    P1 requirement:
    Shared base template and nav bar: Overview, CPU, Latency, Forecast, Alerts & Scaling, Experiments.
    Each page has its own route and space. Every route returns 200.
    """
    routes = ["/", "/overview", "/cpu", "/latency", "/forecast", "/alerts", "/scaling", "/experiments"]
    for route in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with status {resp.status_code}"


def test_cpu_page_exact_note(client):
    """
    P1 requirement:
    Page note (exactly): 'Worker CPU = how busy the active workers are (0-100%). Scaling up adds workers.'
    """
    resp = client.get("/cpu")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    exact_note = "Worker CPU = how busy the active workers are (0-100%). Scaling up adds workers."
    assert exact_note in html, f"Exact note '{exact_note}' not found in /cpu page content"


def test_api_dashboard_state_fields(client):
    """
    P1 requirement:
    Keep every existing dashboard endpoint and JSON field working (only ADD).
    """
    resp = client.get("/api/dashboard/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "telemetry_latest" in data
    assert "telemetry_history" in data
    assert "prediction_latest" in data
    assert "remediation" in data
    # Newly added fields
    assert "worker_cpu_pct_active" in data
    assert "whole_machine_cpu" in data
    assert "slo_latency_ms" in data
    assert data["slo_latency_ms"] == 300.0
