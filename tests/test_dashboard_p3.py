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


def test_mode_label_default_old_model(client, monkeypatch):
    """
    P3 requirement:
    Mode label: 'PERM v0 (old model)' unless the final model is wired in live.
    """
    monkeypatch.delenv("PERM_FORECASTER", raising=False)
    resp = client.get("/api/dashboard/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "model_mode_label" in data
    assert data["model_mode_label"] == "PERM v0 (old model)"


def test_mode_label_with_final_model(client, monkeypatch):
    """
    P3 requirement:
    Mode label changes when final model is wired in live via PERM_FORECASTER=stagbdt.
    """
    monkeypatch.setenv("PERM_FORECASTER", "stagbdt")
    resp = client.get("/api/dashboard/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_mode_label"] == "ST-AGBDT (final model)"
