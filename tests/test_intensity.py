"""
Unit and Integration Tests for Phase 6: Cyclone Intensity Prediction Model.
Validates multi-horizon wind and pressure regression, quantile uncertainty bounds,
hydrodynamic physical consistency enforcement, Rapid Intensification (RI) alerting,
and backend API endpoint integration.
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
from ml.models.intensity_predictor import CycloneIntensityPredictor, FORECAST_HORIZONS, RI_THRESHOLD_KT
from backend.app.main import app

client = TestClient(app)


def test_intensity_model_loading_and_checkpoint():
    """Verifies that the serialized intensity model exists, loads cleanly, and has all components."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    assert os.path.exists(model_path), f"Checkpoint missing: {model_path}"

    predictor = CycloneIntensityPredictor.load(model_path)
    assert predictor.is_fitted is True
    assert set(predictor.horizons) == {0, 6, 12, 24}

    # Verify all horizon models are present
    for h in [0, 6, 12, 24]:
        assert h in predictor.wind_models
        assert h in predictor.pres_models
        assert h in predictor.wind_q10
        assert h in predictor.wind_q90
        assert h in predictor.pres_q10
        assert h in predictor.pres_q90

    # Verify RI classifier is present and fitted
    assert predictor.ri_model is not None


def test_intensity_multi_horizon_forecasting():
    """Verifies that predict_forecast returns complete forecast dictionary with required horizons."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    predictor = CycloneIntensityPredictor.load(model_path)
    extractor = CycloneFeatureExtractor()

    obs = {
        "lat": 17.0,
        "lon": 86.5,
        "central_pres": 985.0,
        "forward_speed": 15.0,
        "bearing": 330.0,
        "month": 10,
        "subbasin": "BB",
    }
    X_df = extractor.extract_from_dict(obs)
    X_df["current_wind"] = 60.0

    forecast = predictor.predict_forecast(X_df)

    assert "current_intensity" in forecast
    assert "forecast_horizons" in forecast
    assert "rapid_intensification" in forecast
    assert "intensity_trend" in forecast

    # Check horizons +6h, +12h, +24h
    for h_label in ["+6h", "+12h", "+24h"]:
        assert h_label in forecast["forecast_horizons"]
        horizon_data = forecast["forecast_horizons"][h_label]
        assert "wind_knots" in horizon_data
        assert "wind_kmh" in horizon_data
        assert "pressure_hpa" in horizon_data
        assert "wind_bounds_90" in horizon_data
        assert "pressure_bounds_90" in horizon_data
        assert horizon_data["wind_knots"] > 0.0
        assert horizon_data["pressure_hpa"] > 800.0


def test_quantile_uncertainty_monotonicity():
    """Verifies that 90% uncertainty intervals satisfy low <= forecast <= high."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    predictor = CycloneIntensityPredictor.load(model_path)
    extractor = CycloneFeatureExtractor()

    obs = {"lat": 14.0, "lon": 89.0, "central_pres": 975.0, "forward_speed": 20.0}
    X_df = extractor.extract_from_dict(obs)
    X_df["current_wind"] = 70.0

    forecast = predictor.predict_forecast(X_df)

    for h_label, data in forecast["forecast_horizons"].items():
        w_low = data["wind_bounds_90"]["low_knots"]
        w_val = data["wind_knots"]
        w_high = data["wind_bounds_90"]["high_knots"]

        assert w_low <= w_high, f"Wind bounds reversed at {h_label}: {w_low} > {w_high}"
        assert w_low <= w_val <= w_high, f"Wind forecast {w_val} outside [{w_low}, {w_high}] at {h_label}"

        p_low = data["pressure_bounds_90"]["low_hpa"]
        p_val = data["pressure_hpa"]
        p_high = data["pressure_bounds_90"]["high_hpa"]

        assert p_low <= p_high, f"Pressure bounds reversed at {h_label}: {p_low} > {p_high}"
        assert p_low <= p_val <= p_high, f"Pressure forecast {p_val} outside [{p_low}, {p_high}] at {h_label}"


def test_hydrodynamic_consistency_enforcement():
    """
    Verifies that enforce_hydrodynamic_consistency couples wind and pressure
    and pulls physically uncoupled values into an acceptable physical envelope.
    """
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    predictor = CycloneIntensityPredictor.load(model_path)

    # Test Case 1: Unphysically high wind (150 kt) for high central pressure (1005 hPa, delta = 5 hPa)
    # Theoretical wind for 5 hPa deficit = 14.2 * sqrt(5) ~ 31.7 kt.
    # Hydrodynamic enforcement should clamp wind down significantly.
    w_cons, p_cons = predictor.enforce_hydrodynamic_consistency(150.0, 1005.0)
    assert w_cons < 60.0, f"Expected wind to be constrained, got {w_cons} kt"

    # Test Case 2: Deep pressure (910 hPa, delta = 100 hPa) with very weak wind (20 kt)
    # Theoretical wind for 100 hPa deficit = 14.2 * 10 = 142 kt.
    # Hydrodynamic enforcement should pull wind up.
    w_cons2, p_cons2 = predictor.enforce_hydrodynamic_consistency(20.0, 910.0)
    assert w_cons2 > 100.0, f"Expected wind to be elevated for 910 hPa, got {w_cons2} kt"


def test_rapid_intensification_detection():
    """Verifies that Rapid Intensification assessment returns valid probability and risk categorization."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    predictor = CycloneIntensityPredictor.load(model_path)
    extractor = CycloneFeatureExtractor()

    obs = {"lat": 12.0, "lon": 88.0, "central_pres": 960.0, "forward_speed": 12.0}
    X_df = extractor.extract_from_dict(obs)
    X_df["current_wind"] = 80.0

    forecast = predictor.predict_forecast(X_df)
    ri = forecast["rapid_intensification"]

    assert "is_alert_active" in ri
    assert isinstance(ri["is_alert_active"], bool)
    assert "probability" in ri
    assert 0.0 <= ri["probability"] <= 1.0
    assert "threshold_knots_24h" in ri
    assert ri["threshold_knots_24h"] == RI_THRESHOLD_KT
    assert "risk_level" in ri


def test_api_intensity_predict_endpoint():
    """Verifies that POST /api/intensity/predict generates live multi-horizon forecast via FastAPI."""
    payload = {
        "storm_id": "SIH_CYCLONE_TEST",
        "lat": 16.0,
        "lon": 87.5,
        "current_wind_speed_knots": 55.0,
        "central_pressure_hpa": 988.0,
        "forward_speed": 18.0,
        "bearing": 325.0,
        "month": 10,
        "subbasin": "BB",
    }
    response = client.post("/api/intensity/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True
    assert data["model_version"] == "1.0.0"

    assert data["current_intensity"]["wind_knots"] == 55.0
    assert data["current_intensity"]["pressure_hpa"] == 988.0

    horizons = data["forecast_horizons"]
    assert "+6h" in horizons
    assert "+12h" in horizons
    assert "+24h" in horizons

    for h in ["+6h", "+12h", "+24h"]:
        h_data = horizons[h]
        assert h_data["wind_knots"] > 0
        assert h_data["pressure_hpa"] > 850
        assert h_data["wind_bounds_90"]["low_knots"] <= h_data["wind_bounds_90"]["high_knots"]

    assert data["rapid_intensification"] is not None
    assert "is_alert_active" in data["rapid_intensification"]
    assert "probability" in data["rapid_intensification"]
    assert data["intensity_trend"] is not None
