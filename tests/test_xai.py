"""
Unit and Integration Tests for Phase 8: Explainable AI (XAI).
Validates Tree-Path Efficiency Axiom conservation, Dvorak convective saliency,
physics-constrained counterfactuals, meteorological narrations, and FastAPI REST endpoints.
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
from ml.models.classifier import CycloneClassifier, CATEGORY_NAMES
from ml.models.intensity_predictor import CycloneIntensityPredictor
from ml.explainability.tabular_explainer import TabularExplainer
from ml.explainability.satellite_saliency import SatelliteSaliencyEngine
from ml.explainability.counterfactuals import CounterfactualEngine
from ml.explainability.narrator import MeteorologicalNarrator
from backend.app.main import app

client = TestClient(app)


def test_tabular_explainer_efficiency_axiom():
    """Verifies that the sum of feature attributions satisfies Efficiency Axiom: sum(phi_i) == diff."""
    cls_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    classifier = CycloneClassifier.load(cls_path)
    extractor = CycloneFeatureExtractor()

    obs = {"lat": 16.5, "lon": 88.0, "central_pres": 970.0, "forward_speed": 18.0}
    X_df = extractor.extract_from_dict(obs)

    explainer = TabularExplainer(feature_names=classifier.feature_names)
    exp = explainer.explain_classification_sample(classifier, X_df)

    assert "efficiency_error" in exp
    assert exp["efficiency_error"] < 1e-4, f"Efficiency Axiom violated: error {exp['efficiency_error']}"
    assert len(exp["waterfall_steps"]) == len(classifier.feature_names)
    assert len(exp["top_drivers"]) <= 5


def test_satellite_saliency_engine():
    """Verifies that SatelliteSaliencyEngine produces normalized saliency and Dvorak BD indicators."""
    engine = SatelliteSaliencyEngine(grid_size=128)
    tb_synth = engine.generate_synthetic_cyclone_field(intensity_factor=0.8)

    res = engine.compute_convective_saliency(tb_synth)

    assert 0.0 <= res["axisymmetry_score"] <= 1.0
    assert res["eye_contrast_celsius"] >= 0.0
    assert res["deep_convective_area_km2"] >= 0.0

    saliency_mat = np.array(res["saliency_matrix"])
    assert np.all(saliency_mat >= 0.0)
    assert np.all(saliency_mat <= 1.0)
    assert len(res["dvorak_bd_distribution"]) == 8


def test_counterfactual_engine_monotonicity():
    """Verifies that pressure deficit sensitivity exhibits positive physical monotonicity."""
    cls_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    int_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    classifier = CycloneClassifier.load(cls_path)
    intensity_predictor = CycloneIntensityPredictor.load(int_path)

    cf_engine = CounterfactualEngine(classifier=classifier, intensity_predictor=intensity_predictor)
    extractor = CycloneFeatureExtractor()
    obs = {"lat": 15.0, "lon": 87.0, "central_pres": 990.0, "forward_speed": 15.0}
    X_df = extractor.extract_from_dict(obs)
    X_df["current_wind"] = 50.0

    sens = cf_engine.compute_pressure_deficit_sensitivity(X_df, num_points=10)
    assert sens["physical_monotonicity_verified"] is True
    curve = sens["sensitivity_curve"]
    assert len(curve) == 10
    # First point (low deficit) should have lower wind than last point (high deficit)
    assert curve[0]["predicted_wind_24h_knots"] <= curve[-1]["predicted_wind_24h_knots"]


def test_counterfactual_category_shift():
    """Verifies that find_counterfactual_category_shift returns physically bounded perturbation."""
    cls_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    classifier = CycloneClassifier.load(cls_path)
    cf_engine = CounterfactualEngine(classifier=classifier)

    extractor = CycloneFeatureExtractor()
    obs = {"lat": 15.0, "lon": 87.0, "central_pres": 995.0, "forward_speed": 15.0}
    X_df = extractor.extract_from_dict(obs)

    # Shift to Very Severe Cyclonic Storm (index 5)
    res = cf_engine.find_counterfactual_category_shift(X_df, target_category_idx=5)
    assert res["status"] in ["counterfactual_found", "already_satisfied"]
    if res["status"] == "counterfactual_found":
        assert "required_pressure_deficit_hpa" in res
        assert res["required_pressure_deficit_hpa"] > 0.0


def test_meteorological_narrator():
    """Verifies that MeteorologicalNarrator formats comprehensive IMD-style briefings."""
    narrator = MeteorologicalNarrator()
    top_drivers = [
        {"feature": "pressure_deficit", "direction": "positive", "contribution": 0.35},
        {"feature": "forward_speed", "direction": "negative", "contribution": -0.12},
    ]

    briefing = narrator.generate_diagnostic_narrative(
        category_name="Severe Cyclonic Storm",
        category_code="SCS",
        confidence_pct=68.5,
        top_drivers=top_drivers,
        current_wind_kt=55.0,
        central_pres_hpa=985.0,
        predicted_wind_24h_kt=75.0,
        ri_alert=False,
        subbasin="BB",
    )

    assert "Bay of Bengal" in briefing
    assert "Severe Cyclonic Storm" in briefing
    assert "985.0 hPa" in briefing
    assert "pressure deficit" in briefing

    track_briefing = narrator.generate_track_narrative(
        current_lat=15.0, current_lon=88.0,
        pred_lat_24h=17.5, pred_lon_24h=86.5,
        forward_speed_kmh=18.0, bearing_deg=320.0,
        subbasin="BB",
    )
    assert "translating" in track_briefing
    assert "18.0 km/h" in track_briefing


def test_api_xai_explain_endpoint():
    """Verifies that POST /api/xai/explain returns full multi-model explanation payload."""
    payload = {
        "storm_id": "SIH_XAI_TEST",
        "lat": 16.5,
        "lon": 87.5,
        "central_pres": 980.0,
        "current_wind_speed_knots": 65.0,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "month": 10,
        "subbasin": "BB",
    }
    response = client.post("/api/xai/explain", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["model_status"] == "active"
    assert data["model_connected"] is True

    assert "classification_explanation" in data
    assert "intensity_explanation" in data
    assert "track_steering_explanation" in data
    assert "synoptic_narrative" in data
    assert "counterfactual_summary" in data

    cls_exp = data["classification_explanation"]
    assert "predicted_category" in cls_exp
    assert "waterfall_steps" in cls_exp
    assert len(cls_exp["waterfall_steps"]) > 0


def test_api_xai_saliency_endpoint():
    """Verifies that POST /api/xai/saliency returns 2D convective saliency grid and Dvorak BD zones."""
    payload = {
        "sensor": "INSAT-3D",
        "channel": "TIR-1",
        "intensity_factor": 0.85,
    }
    response = client.post("/api/xai/saliency", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "vortex_center" in data
    assert "axisymmetry_score" in data
    assert "dvorak_bd_distribution" in data
    assert len(data["saliency_matrix"]) > 0


def test_api_xai_sensitivity_endpoint():
    """Verifies that POST /api/xai/sensitivity returns monotonic physical parameter curves."""
    payload = {
        "parameter": "pressure_deficit_hpa",
        "lat": 16.0,
        "lon": 87.5,
        "central_pres": 985.0,
        "subbasin": "BB",
    }
    response = client.post("/api/xai/sensitivity", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "sensitivity_curve" in data
    assert len(data["sensitivity_curve"]) > 0
    assert data["physical_monotonicity_verified"] is True
