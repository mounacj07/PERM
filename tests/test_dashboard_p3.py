"""
Unit tests for Dashboard Priority 3 (P3): Alerts & Scaling page and capacity scale events.
"""

import os
import pytest
from web_dashboard.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_alerts_page_structure(client):
    """
    P3 requirement:
    Alerts table (reuse /api existing alerts) and a worker-count timeline.
    Scale events list: time, from, to, cold-start seconds.
    """
    resp = client.get("/alerts")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Worker Count Timeline" in html
    assert "Active & Recent Alerts Stream" in html
    assert "Scale Events Audit" in html
    assert "Cold-Start" in html


def test_scaling_route_alias(client):
    resp = client.get("/scaling")
    assert resp.status_code == 200


def test_mode_label_controller_not_running(client, monkeypatch, tmp_path):
    """
    Mode label: read controller_status.json.
    If the file is missing or older than 5 s, show 'Controller not running'.
    """
    from datetime import datetime, timezone, timedelta
    import json

    # 1. Missing file
    non_existent = str(tmp_path / "non_existent_status.json")
    monkeypatch.setattr("web_dashboard.app.STATUS_JSON_PATH", non_existent)
    resp = client.get("/api/dashboard/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode_label"] == "Controller not running"
    assert data["model_mode_label"] == "Controller not running"

    # 2. File older than 5 seconds
    stale_file = tmp_path / "stale_status.json"
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    stale_file.write_text(json.dumps({
        "mode": "predictive",
        "forecast_online": True,
        "last_decision": {"action": "hold", "target": 2, "rule": "reactive", "current_workers": 2},
        "time": old_time
    }), encoding="utf-8")
    monkeypatch.setattr("web_dashboard.app.STATUS_JSON_PATH", str(stale_file))

    resp_stale = client.get("/api/dashboard/state")
    assert resp_stale.status_code == 200
    data_stale = resp_stale.get_json()
    assert data_stale["mode_label"] == "Controller not running"


def test_mode_label_fresh_controller_status(client, monkeypatch, tmp_path):
    """
    Fresh controller_status.json shows: 'Demo controller: <mode> (forecast online/offline)'.
    Does not depend on PERM_FORECASTER env var.
    """
    from datetime import datetime, timezone
    import json

    fresh_file = tmp_path / "fresh_status.json"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Case A: Predictive mode with forecast online
    fresh_file.write_text(json.dumps({
        "mode": "predictive",
        "forecast_online": True,
        "last_decision": {"action": "hold", "target": 2, "rule": "reactive", "current_workers": 2},
        "time": now_iso
    }), encoding="utf-8")
    monkeypatch.setattr("web_dashboard.app.STATUS_JSON_PATH", str(fresh_file))

    resp = client.get("/api/dashboard/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mode_label"] == "Demo controller: predictive (forecast online)"

    # Case B: Reactive mode with forecast offline (even if PERM_FORECASTER is set, ignore it)
    monkeypatch.setenv("PERM_FORECASTER", "stagbdt")
    fresh_file.write_text(json.dumps({
        "mode": "reactive",
        "forecast_online": False,
        "last_decision": {"action": "hold", "target": 2, "rule": "reactive", "current_workers": 2},
        "time": now_iso
    }), encoding="utf-8")

    resp2 = client.get("/api/dashboard/state")
    assert resp2.status_code == 200
    data2 = resp2.get_json()
    assert data2["mode_label"] == "Demo controller: reactive (forecast offline)"
