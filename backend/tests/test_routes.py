"""
Unit tests for placeholder API endpoints.
Verifies that placeholders return explicit 'not_connected' states and zero fake metrics.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_satellite_analyze_active():
    response = client.post("/api/satellite/analyze", json={"sensor": "INSAT-3D", "channel": "TIR-1"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True
    assert data["model_version"] == "1.0.0"
    assert "detection_result" in data
    assert data["detection_result"] is not None

def test_cyclone_classify_active():
    response = client.post("/api/cyclone/classify", json={"basin": "North Indian Ocean", "lat": 16.0, "lon": 87.0, "central_pres": 980.0})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True
    assert data["predicted_category"] is not None
    assert data["confidence"] is not None
    assert data["category_code"] is not None

def test_intensity_predict_active():
    response = client.post("/api/intensity/predict", json={
        "storm_id": "TEST_001",
        "lat": 16.5,
        "lon": 88.0,
        "current_wind_speed_knots": 65.0,
        "central_pressure_hpa": 980.0,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "month": 10,
        "subbasin": "BB",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True
    assert data["current_intensity"] is not None
    assert "wind_knots" in data["current_intensity"]
    assert data["forecast_horizons"] is not None
    assert "+6h" in data["forecast_horizons"]
    assert "+12h" in data["forecast_horizons"]
    assert "+24h" in data["forecast_horizons"]
    assert data["rapid_intensification"] is not None
    assert "is_alert_active" in data["rapid_intensification"]

def test_track_predict_active():
    response = client.post("/api/track/predict", json={
        "storm_id": "TEST_TRACK_001",
        "lat": 16.0,
        "lon": 87.5,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "central_pres": 985.0,
        "current_wind_speed_knots": 60.0,
        "month": 10,
        "subbasin": "BB",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True
    assert data["initial_position"] is not None
    assert len(data["predicted_track"]) == 4
    for wp in data["predicted_track"]:
        assert "lead_time" in wp
        assert "latitude" in wp
        assert "longitude" in wp
        assert "forward_speed_kmh" in wp
        assert "uncertainty_radius_km" in wp
    assert len(data["track_error_cone"]) == 4
    for cone in data["track_error_cone"]:
        assert "polygon_coordinates" in cone
        assert len(cone["polygon_coordinates"]) > 0
