"""
Unit tests for Dashboard Priority 2 (P2): Forecast page Live / Replay switch and backend APIs.
"""

import pytest
from web_dashboard.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_list_forecast_runs_excludes_sawtooth(client):
    """
    P2 requirement:
    Stream any run from data/runs_locust (exclude sawtooth).
    """
    resp = client.get("/api/forecast/runs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "runs" in data
    assert data["count"] > 0
    run_ids = [r["run_id"] for r in data["runs"]]
    for r_id in run_ids:
        assert not r_id.startswith("sawtooth"), f"Sawtooth run {r_id} was not excluded"
    assert "random_mix_seed20" in run_ids


def test_get_forecast_replay_held_out_banner(client):
    """
    P2 requirement:
    Banner: 'ST-AGBDT trained offline; replay uses runs not used in training'
    (only if that is true for the selected run; otherwise mark the run as training data).
    Show recommended workers vs the workers required by the actual demand.
    """
    resp = client.get("/api/forecast/replay?run_id=random_mix_seed20&horizon=20")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_available"] is True
    assert data["is_training"] is False
    assert data["banner"] == "ST-AGBDT trained offline; replay uses runs not used in training"
    assert "records" in data
    assert len(data["records"]) > 0

    first_rec = data["records"][0]
    assert "current_arrivals" in first_rec
    assert "raw_forecast" in first_rec
    assert "upper_bound" in first_rec
    assert "recommended_workers" in first_rec
    assert "actual_workers_needed" in first_rec


def test_get_forecast_replay_training_run_banner(client):
    """
    P2 requirement:
    If selected run is training data, mark the run as training data.
    """
    resp = client.get("/api/forecast/replay?run_id=step_seed43&horizon=20")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_available"] is True
    assert data["is_training"] is True
    assert "Training run: This run was used during offline model training." in data["banner"]


def test_forecast_live_offline_clean_state(client):
    """
    P2 requirement:
    Live: poll the existing endpoints; if no forecast service is running,
    show a clear 'forecast service offline' state (never fake data).
    """
    resp = client.get("/api/forecast/live?horizon=20")
    assert resp.status_code == 200
    data = resp.get_json()
    # Since live service is not running on 5001 during test, returns clean offline status
    assert data["status"] == "offline"
    assert "forecast service offline" in data["message"]
