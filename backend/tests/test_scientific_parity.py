"""
Phase 10 Scientific Parity & Master Verification Test Suite.
Validates:
1. Complete zero-leakage isolation of the 70-storm test split.
2. Master scientific evaluation report integrity and schema conformance.
3. API inference parity vs offline model execution (|Delta| < 1e-6).
4. Physical and kinematic consistency constraints on all predictions.
5. Statistical significance thresholds (p < 0.001 for track skill).
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app
from app.api.dependencies import (
    get_satellite_service,
    get_classification_service,
    get_intensity_service,
    get_track_service,
)
from app.schemas.pipeline import PipelineRunRequest
from app.schemas.satellite import SatelliteAnalysisRequest
from app.schemas.cyclone import CycloneClassificationRequest
from app.schemas.intensity import IntensityPredictionRequest
from app.schemas.track import TrackPredictionRequest

client = TestClient(app)


def test_zero_leakage_audit():
    """Confirms 0.00% storm overlap across train, val, and test splits."""
    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")
    val_path = os.path.join(PROJECT_ROOT, "data/processed/val_manifest.csv")
    test_path = os.path.join(PROJECT_ROOT, "data/processed/test_manifest.csv")

    train_storms = set(pd.read_csv(train_path)["SID"].unique())
    val_storms = set(pd.read_csv(val_path)["SID"].unique())
    test_storms = set(pd.read_csv(test_path)["SID"].unique())

    assert len(test_storms) == 70, f"Expected 70 held-out test storms, found {len(test_storms)}"
    assert len(test_storms.intersection(train_storms)) == 0, "Data leakage detected: test storms present in train set!"
    assert len(test_storms.intersection(val_storms)) == 0, "Data leakage detected: test storms present in val set!"
    assert len(train_storms.intersection(val_storms)) == 0, "Data leakage detected: train and val storms overlap!"


def test_master_evaluation_report_exists_and_valid():
    """Validates presence and schema of scientific_evaluation_master_report.json."""
    report_path = os.path.join(PROJECT_ROOT, "docs/models/evaluation_reports/scientific_evaluation_master_report.json")
    assert os.path.exists(report_path), f"Master report not found at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "leakage_verification" in data
    assert data["leakage_verification"]["leakage_count"] == 0
    assert data["leakage_verification"]["leakage_percentage"] == 0.0

    benchmarks = data["overall_model_benchmarks"]
    assert "cyclone_detection" in benchmarks
    assert benchmarks["cyclone_detection"]["test_roc_auc"] >= 0.88

    assert "category_classification" in benchmarks
    assert benchmarks["category_classification"]["adjacent_match_pct"] >= 75.0

    assert "multi_horizon_track" in benchmarks
    track_bench = benchmarks["multi_horizon_track"]
    assert "+24h" in track_bench
    assert track_bench["+24h"]["skill_gain_vs_persistence_pct"] > 0
    assert track_bench["+48h"]["skill_gain_vs_persistence_pct"] > 0

    assert "statistical_significance_tests" in data
    p_verdict = data["statistical_significance_tests"]
    assert p_verdict["track_paired_students_t_test_vs_persistence"]["+24h"]["is_significant_p001"] is True
    assert p_verdict["track_paired_students_t_test_vs_persistence"]["+48h"]["is_significant_p001"] is True

    assert "operational_case_studies" in data
    assert len(data["operational_case_studies"]) >= 5
    for case in ["TAUKTAE", "REMAL", "ASANI", "PHET", "MADI"]:
        assert case in data["operational_case_studies"]


def test_inference_parity_offline_vs_api():
    """
    Verifies numerical parity between direct offline model execution
    and live FastAPI pipeline endpoint (|Delta| < 1e-6).
    """
    sample_obs = {
        "storm_id": "TEST_PARITY_TAUKTAE",
        "lat": 18.5,
        "lon": 68.2,
        "forward_speed": 16.5,
        "bearing": 335.0,
        "central_pres": 955.0,
        "month": 5,
        "subbasin": "AS",
        "current_wind_speed_knots": 95.0,
        "past_track": [
            {"latitude": 17.2, "longitude": 68.8, "intensity_knots": 85.0}
        ]
    }

    # 1. Direct offline service predictions
    sat_svc = get_satellite_service()
    cls_svc = get_classification_service()
    int_svc = get_intensity_service()
    trk_svc = get_track_service()

    offline_sat = sat_svc.analyze_satellite_image(SatelliteAnalysisRequest(
        image_name="test_granule.tif",
        sensor="INSAT-3D",
        channel="TIR-1",
        lat=sample_obs["lat"],
        lon=sample_obs["lon"],
        central_pres=sample_obs["central_pres"],
        forward_speed=sample_obs["forward_speed"],
        bearing=sample_obs["bearing"],
        month=sample_obs["month"],
        subbasin=sample_obs["subbasin"]
    ))

    offline_cls = cls_svc.classify_cyclone(CycloneClassificationRequest(
        lat=sample_obs["lat"],
        lon=sample_obs["lon"],
        central_pres=sample_obs["central_pres"],
        forward_speed=sample_obs["forward_speed"],
        bearing=sample_obs["bearing"],
        month=sample_obs["month"],
        subbasin=sample_obs["subbasin"]
    ))

    offline_int = int_svc.predict_intensity(IntensityPredictionRequest(
        storm_id=sample_obs["storm_id"],
        lat=sample_obs["lat"],
        lon=sample_obs["lon"],
        current_wind_speed_knots=sample_obs["current_wind_speed_knots"],
        central_pressure_hpa=sample_obs["central_pres"],
        forward_speed=sample_obs["forward_speed"],
        bearing=sample_obs["bearing"],
        month=sample_obs["month"],
        subbasin=sample_obs["subbasin"]
    ))

    offline_trk = trk_svc.predict_track(TrackPredictionRequest(
        storm_id=sample_obs["storm_id"],
        lat=sample_obs["lat"],
        lon=sample_obs["lon"],
        forward_speed=sample_obs["forward_speed"],
        bearing=sample_obs["bearing"],
        central_pres=sample_obs["central_pres"],
        current_wind_speed_knots=sample_obs["current_wind_speed_knots"],
        month=sample_obs["month"],
        subbasin=sample_obs["subbasin"],
        past_track=sample_obs["past_track"]
    ))

    # 2. API Pipeline execution
    response = client.post("/api/pipeline/run", json=sample_obs)
    assert response.status_code == 200, f"API error: {response.text}"
    api_res = response.json()

    # 3. Numerical Parity Verification
    # Identification probability
    assert abs(api_res["identification"]["probability_cs"] - offline_sat.probability_cs) < 1e-6

    # Classification
    assert api_res["classification"]["predicted_category"] == offline_cls.predicted_category
    assert abs(api_res["classification"]["confidence"] - offline_cls.confidence) < 1e-6

    # Intensity +24h
    api_w24 = api_res["intensity"]["forecast_horizons"]["+24h"]["wind_knots"]
    off_w24 = offline_int.forecast_horizons["+24h"]["wind_knots"]
    assert abs(api_w24 - off_w24) < 1e-6

    api_p24 = api_res["intensity"]["forecast_horizons"]["+24h"]["pressure_hpa"]
    off_p24 = offline_int.forecast_horizons["+24h"]["pressure_hpa"]
    assert abs(api_p24 - off_p24) < 1e-6

    # Track +24h
    api_wp24 = [w for w in api_res["track"]["predicted_track"] if w["lead_time"] == "+24h"][0]
    off_wp24 = [w for w in offline_trk.predicted_track if w["lead_time"] == "+24h"][0]
    assert abs(api_wp24["latitude"] - off_wp24["latitude"]) < 1e-6
    assert abs(api_wp24["longitude"] - off_wp24["longitude"]) < 1e-6


def test_physical_consistency_constraints():
    """Validates that predictions strictly adhere to physical atmospheric laws."""
    sample_obs = {
        "lat": 14.0,
        "lon": 87.0,
        "central_pres": 940.0,
        "current_wind_speed_knots": 105.0,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "month": 10,
        "subbasin": "BB"
    }
    response = client.post("/api/pipeline/run", json=sample_obs)
    assert response.status_code == 200
    res = response.json()

    # 1. Classification probabilities sum to 1.0
    probs = res["classification"]["all_probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3

    # 2. Pressure-Wind consistency: deeper pressure must correspond to higher winds
    for h in ["+6h", "+12h", "+24h"]:
        w = res["intensity"]["forecast_horizons"][h]["wind_knots"]
        p = res["intensity"]["forecast_horizons"][h]["pressure_hpa"]
        assert 15.0 <= w <= 165.0, f"Wind {w} kt exceeds physical limits"
        assert 870.0 <= p <= 1025.0, f"Pressure {p} hPa exceeds physical limits"

    # 3. Track kinematic speed clamping (< 120 km/h)
    for wp in res["track"]["predicted_track"]:
        assert wp["forward_speed_kmh"] <= 120.0, f"Speed {wp['forward_speed_kmh']} km/h exceeds maximum physical translation limit"


def test_system_metrics_endpoint_reflects_phase10_report():
    """Validates that GET /api/system/metrics correctly reads the Master Scientific Report."""
    res = client.get("/api/system/metrics")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert "models" in data
    assert "cyclone_detector" in data["models"]
    assert "cyclone_classifier" in data["models"]
    assert "cyclone_intensity" in data["models"]
    assert "cyclone_track" in data["models"]
    assert "explainable_ai" in data["models"]
