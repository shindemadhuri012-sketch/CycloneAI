"""
Unit and Integration Tests for Phase 7: Cyclone Track Prediction Model.
Validates multi-horizon waypoint forecasting (+6h, +12h, +24h, +48h), physical speed/turn limits,
orthogonal along/cross-track error decomposition, uncertainty cone polygons, and FastAPI endpoint integration.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from ml.features.extractor import CycloneFeatureExtractor
from ml.models.track_predictor import (
    CycloneTrackPredictor,
    TRACK_HORIZONS,
    MAX_PHYSICAL_SPEED_KMH,
    haversine_km,
    decompose_track_error,
)
from backend.app.main import app

client = TestClient(app)


def test_track_model_loading_and_checkpoint():
    """Verifies that the serialized track model exists, loads cleanly, and has all components."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_track.joblib")
    assert os.path.exists(model_path), f"Checkpoint missing: {model_path}"

    predictor = CycloneTrackPredictor.load(model_path)
    assert predictor.is_fitted is True
    assert set(predictor.horizons) == {6, 12, 24, 48}

    # Verify models exist for every lead horizon
    for h in [6, 12, 24, 48]:
        assert h in predictor.dlat_models
        assert h in predictor.dlon_models
        assert h in predictor.cone_radii_km
        assert predictor.cone_radii_km[h] > 0.0


def test_multi_horizon_track_waypoint_generation():
    """Verifies that predict_track generates sequential waypoints for +6h, +12h, +24h, +48h."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_track.joblib")
    predictor = CycloneTrackPredictor.load(model_path)
    extractor = CycloneFeatureExtractor()

    obs = {
        "lat": 16.0,
        "lon": 87.5,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "central_pres": 980.0,
        "month": 10,
        "subbasin": "BB",
    }
    X_df = extractor.extract_from_dict(obs)
    X_df["past_dlat_6h"] = 0.5
    X_df["past_dlon_6h"] = -0.4
    X_df["past_dlat_12h"] = 1.0
    X_df["past_dlon_12h"] = -0.8
    X_df["current_wind"] = 65.0

    forecast = predictor.predict_track(current_lat=16.0, current_lon=87.5, obs_features=X_df)

    assert "initial_position" in forecast
    assert forecast["initial_position"]["latitude"] == 16.0
    assert forecast["initial_position"]["longitude"] == 87.5

    waypoints = forecast["predicted_track"]
    assert len(waypoints) == 4

    expected_leads = ["+6h", "+12h", "+24h", "+48h"]
    for idx, wp in enumerate(waypoints):
        assert wp["lead_time"] == expected_leads[idx]
        assert 0.0 <= wp["latitude"] <= 35.0
        assert 45.0 <= wp["longitude"] <= 105.0
        assert wp["forward_speed_kmh"] <= MAX_PHYSICAL_SPEED_KMH
        assert wp["uncertainty_radius_km"] > 0.0


def test_physical_speed_and_curvature_constraints():
    """
    Verifies that enforce_kinematic_consistency bounds unphysical forward speeds
    and clamps coordinates to the North Indian Ocean basin.
    """
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_track.joblib")
    predictor = CycloneTrackPredictor.load(model_path)

    # Test Case 1: Unphysically high speed (e.g. 150 km/h leap in 6h = 900 km)
    # Origin: (15.0, 85.0), Pred: (23.0, 85.0) -> ~888 km in 6h = 148 km/h
    cons_lat, cons_lon, speed, bearing = predictor.enforce_kinematic_consistency(
        origin_lat=15.0, origin_lon=85.0,
        pred_lat=23.0, pred_lon=85.0,
        lead_hours=6
    )
    assert speed <= MAX_PHYSICAL_SPEED_KMH + 0.1
    # Clamped distance should be <= 45 * 6 = 270 km
    clamped_dist = haversine_km(15.0, 85.0, cons_lat, cons_lon)
    assert clamped_dist <= 271.0

    # Test Case 2: Out of basin coordinates (e.g. lat = -5.0, lon = 120.0)
    cons_lat2, cons_lon2, _, _ = predictor.enforce_kinematic_consistency(
        origin_lat=10.0, origin_lon=90.0,
        pred_lat=-5.0, pred_lon=120.0,
        lead_hours=6
    )
    assert cons_lat2 >= 0.0
    assert cons_lon2 <= 105.0


def test_uncertainty_cone_polygon_structure():
    """Verifies that generate_cone_polygon returns a closed polygon ring suitable for Leaflet maps."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_track.joblib")
    predictor = CycloneTrackPredictor.load(model_path)

    ring = predictor.generate_cone_polygon(center_lat=18.0, center_lon=86.0, radius_km=100.0, num_points=18)
    assert len(ring) == 19  # 18 points + 1 closing point
    assert ring[0] == ring[-1]  # Closed polygon

    for pt in ring:
        assert len(pt) == 2
        lat, lon = pt[0], pt[1]
        dist = haversine_km(18.0, 86.0, lat, lon)
        assert abs(dist - 100.0) < 5.0  # Approx circular


def test_geodesic_error_decomposition():
    """Verifies that decompose_track_error satisfies orthogonal error property ATE^2 = ATE_A^2 + XTE^2."""
    origin_lat, origin_lon = 15.0, 85.0
    true_lat, true_lon = 17.0, 84.0
    pred_lat, pred_lon = 17.5, 83.5

    ate, ate_a, xte, heading_err = decompose_track_error(
        lat_pred=pred_lat, lon_pred=pred_lon,
        lat_true=true_lat, lon_true=true_lon,
        lat_origin=origin_lat, lon_origin=origin_lon,
    )

    reconstructed_ate = np.sqrt(ate_a ** 2 + xte ** 2)
    assert abs(ate - reconstructed_ate) < 0.01
    assert ate > 0.0
    assert 0.0 <= heading_err <= 180.0


def test_persistence_and_cliper_baselines():
    """Verifies that persistence and CLIPER baseline extrapolations operate reasonably."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_track.joblib")
    predictor = CycloneTrackPredictor.load(model_path)

    o_lat, o_lon = 15.0, 88.0
    dlat, dlon = 0.4, -0.3

    per_lat, per_lon = predictor.predict_persistence(o_lat, o_lon, dlat, dlon, lead_hours=24)
    assert abs(per_lat - (15.0 + 0.4 * 4)) < 0.01
    assert abs(per_lon - (88.0 - 0.3 * 4)) < 0.01

    clip_lat, clip_lon = predictor.predict_cliper(o_lat, o_lon, dlat, dlon, lead_hours=24)
    assert 0.0 <= clip_lat <= 35.0
    assert 45.0 <= clip_lon <= 105.0


def test_api_track_predict_endpoint():
    """Verifies that POST /api/track/predict returns live multi-horizon waypoints and cones."""
    payload = {
        "storm_id": "SIH_TRACK_TEST",
        "lat": 16.5,
        "lon": 87.5,
        "forward_speed": 17.0,
        "bearing": 315.0,
        "central_pres": 982.0,
        "current_wind_speed_knots": 65.0,
        "month": 10,
        "subbasin": "BB",
    }
    response = client.post("/api/track/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True
    assert data["model_version"] == "1.0.0"

    assert data["initial_position"]["latitude"] == 16.5
    assert data["initial_position"]["longitude"] == 87.5

    waypoints = data["predicted_track"]
    assert len(waypoints) == 4
    for wp in waypoints:
        assert wp["lead_time"] in ["+6h", "+12h", "+24h", "+48h"]
        assert "latitude" in wp
        assert "longitude" in wp
        assert "forward_speed_kmh" in wp
        assert "uncertainty_radius_km" in wp

    cones = data["track_error_cone"]
    assert len(cones) == 4
    for cone in cones:
        assert len(cone["polygon_coordinates"]) > 0
        assert cone["radius_km"] > 0.0
