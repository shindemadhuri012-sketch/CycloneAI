"""
Integration tests for Phase 9 full-system integration.
Tests unified multi-model pipeline execution, storm catalog, and system metrics.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_system_health_all_subsystems_active():
    """Verifies /api/health reports active service state."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "operational"]
    assert "active_models" in data or "model_status" in data


def test_system_metrics_endpoint():
    """Verifies /api/system/metrics returns verified evaluation reports."""
    response = client.get("/api/system/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "models" in data
    assert "cyclone_detector" in data["models"]
    assert "cyclone_classifier" in data["models"]
    assert "cyclone_intensity" in data["models"]
    assert "cyclone_track" in data["models"]
    assert "explainable_ai" in data["models"]
    # Check that real numbers exist
    det = data["models"]["cyclone_detector"]
    assert "test_metrics" in det or "roc_auc" in det or "sample_count" in det


def test_storms_presets_and_catalog():
    """Verifies /api/storms/presets and /api/storms/catalog return real data."""
    presets_res = client.get("/api/storms/presets")
    assert presets_res.status_code == 200
    presets = presets_res.json()
    assert isinstance(presets, list)
    assert len(presets) > 0
    # Must contain verified storms like FANI or AMPHAN
    names = [p["name"].upper() for p in presets]
    assert any(n in names for n in ["FANI", "AMPHAN", "BIPARJOY", "MOCHA", "REMAL"])

    # Test catalog filtering
    cat_res = client.get("/api/storms/catalog?limit=10")
    assert cat_res.status_code == 200
    cat_data = cat_res.json()
    assert cat_data["status"] == "success"
    assert len(cat_data["storms"]) <= 10
    first_sid = cat_data["storms"][0]["sid"]

    # Test points query
    pts_res = client.get(f"/api/storms/{first_sid}/points")
    assert pts_res.status_code == 200
    pts_data = pts_res.json()
    assert pts_data["sid"] == first_sid
    assert len(pts_data["points"]) > 0
    assert "latitude" in pts_data["points"][0]
    assert "longitude" in pts_data["points"][0]


def test_unified_pipeline_run():
    """Verifies POST /api/pipeline/run executes all 5 analytical components in one pass."""
    payload = {
        "storm_id": "TEST_INTEGRATION",
        "lat": 14.5,
        "lon": 86.2,
        "central_pres": 970.0,
        "current_wind_speed_knots": 85.0,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "month": 5,
        "subbasin": "BB",
        "past_track": [
            {"latitude": 13.8, "longitude": 87.0, "intensity_knots": 75.0}
        ]
    }
    response = client.post("/api/pipeline/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["execution_time_ms"] > 0.0  # Execution time recorded

    # 1. Identification
    assert "identification" in data
    assert "is_cyclone" in data["identification"]
    assert data["identification"]["is_cyclone"] is True
    assert data["identification"]["probability_cs"] > 0.5

    # 2. Classification
    assert "classification" in data
    assert "predicted_category" in data["classification"]
    assert "confidence" in data["classification"]
    assert "all_probabilities" in data["classification"]

    # 3. Intensity
    assert "intensity" in data
    assert "forecast_horizons" in data["intensity"]
    horizons = data["intensity"]["forecast_horizons"]
    assert "+6h" in horizons
    assert "+12h" in horizons
    assert "+24h" in horizons
    assert "low_knots" in horizons["+24h"]["wind_bounds_90"]
    assert "high_knots" in horizons["+24h"]["wind_bounds_90"]

    # 4. Track
    assert "track" in data
    assert "predicted_track" in data["track"]
    assert len(data["track"]["predicted_track"]) == 4  # +6h, +12h, +24h, +48h
    assert "track_error_cone" in data["track"]
    assert len(data["track"]["track_error_cone"]) == 4

    # 5. Explainable AI
    assert "explainability" in data
    assert "top_drivers" in data["explainability"]
    assert len(data["explainability"]["top_drivers"]) > 0
    assert "synoptic_briefing" in data["explainability"]
    assert "axisymmetry_score" in data["explainability"]
    assert "dvorak_bd_distribution" in data["explainability"]
