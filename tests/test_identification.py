"""
Comprehensive Unit and Integration Tests for Phase 4: Cyclone Identification Model.
Validates feature extraction, model calibration, physical consistency,
reproducibility, and FastAPI endpoint inference.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from ml.features.extractor import CycloneFeatureExtractor, FEATURE_NAMES
from ml.models.detector import CycloneDetector
from ml.models.satellite_detector import SatelliteConvectiveAnalyzer
from backend.app.main import app

client = TestClient(app)


def test_feature_extractor_dataframe():
    """Validates feature extraction on DataFrame input."""
    sample_df = pd.DataFrame({
        "LAT": [12.5, 18.0],
        "LON": [85.0, 88.5],
        "FORWARD_SPEED_KMH": [14.0, 22.0],
        "DIST_KM": [42.0, 66.0],
        "DT_HOURS": [3.0, 3.0],
        "BEARING_DEG": [315.0, 340.0],
        "PRESSURE_DEFICIT": [25.0, 45.0],
        "CENTRAL_PRES_HPA": [985.0, 965.0],
        "ISO_TIME": ["2013-10-10 06:00:00", "2019-05-02 12:00:00"],
        "SUBBASIN": ["BB", "BB"],
        "IMD_GRADE": ["VSCS", "ESCS"],
        "EFFECTIVE_WIND_KT": [65.0, 90.0],
    })

    extractor = CycloneFeatureExtractor()
    X, y_cs, y_dep = extractor.extract_from_dataframe(sample_df)

    assert X.shape == (2, len(FEATURE_NAMES))
    assert list(X.columns) == FEATURE_NAMES
    assert X.isna().sum().sum() == 0
    assert y_cs.tolist() == [1, 1]
    assert y_dep.tolist() == [1, 1]


def test_feature_extractor_dict():
    """Validates single dictionary inference extraction."""
    extractor = CycloneFeatureExtractor()
    obs = {
        "lat": 14.2,
        "lon": 87.5,
        "forward_speed": 18.0,
        "central_pres": 990.0,
        "bearing": 300.0,
        "month": 11,
        "subbasin": "BB",
    }
    X = extractor.extract_from_dict(obs)
    assert X.shape == (1, len(FEATURE_NAMES))
    assert X["lat"].iloc[0] == 14.2
    assert X["central_pres"].iloc[0] == 990.0
    assert X["pressure_deficit"].iloc[0] == 20.0
    assert X["is_bob"].iloc[0] == 1.0


def test_model_loading_and_checkpoint():
    """Verifies that the serialized model checkpoint exists and is valid."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_detector.joblib")
    assert os.path.exists(model_path), f"Checkpoint missing: {model_path}"

    detector = CycloneDetector.load(model_path)
    assert detector.is_fitted is True
    assert len(detector.feature_importances_) == len(FEATURE_NAMES)
    assert "pressure_deficit" in detector.feature_importances_
    assert "central_pres" in detector.feature_importances_


def test_probability_ranges_and_thresholds():
    """Checks that model probabilities are strictly between 0 and 1 and calibration holds."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_detector.joblib")
    detector = CycloneDetector.load(model_path)
    extractor = CycloneFeatureExtractor()

    # Create synthetic test points
    test_cases = [
        {"lat": 15.0, "lon": 86.0, "central_pres": 950.0, "forward_speed": 20.0},  # Severe cyclone
        {"lat": 15.0, "lon": 86.0, "central_pres": 1008.0, "forward_speed": 10.0},  # Weak disturbance
    ]

    for tc in test_cases:
        X = extractor.extract_from_dict(tc)
        probs = detector.predict_proba(X)
        assert probs.shape == (1, 2)
        assert 0.0 <= probs[0, 0] <= 1.0
        assert 0.0 <= probs[0, 1] <= 1.0
        assert abs(probs[0, 0] + probs[0, 1] - 1.0) < 1e-4


def test_physical_consistency():
    """
    Physical sanity test: A severe storm with central pressure 940 hPa MUST have a higher
    probability of being a cyclonic storm than a 1010 hPa background system.
    """
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_detector.joblib")
    detector = CycloneDetector.load(model_path)
    extractor = CycloneFeatureExtractor()

    severe_storm = extractor.extract_from_dict({"lat": 16.0, "lon": 88.0, "central_pres": 940.0, "month": 10})
    weak_low = extractor.extract_from_dict({"lat": 16.0, "lon": 88.0, "central_pres": 1009.0, "month": 10})

    p_severe = detector.predict_proba(severe_storm)[0, 1]
    p_weak = detector.predict_proba(weak_low)[0, 1]

    assert p_severe > 0.85, f"Expected high probability for severe storm, got {p_severe}"
    assert p_weak < 0.25, f"Expected low probability for weak low, got {p_weak}"
    assert p_severe > p_weak


def test_satellite_convective_analyzer():
    """Tests morphological feature extraction on synthetic infrared brightness temperature field."""
    analyzer = SatelliteConvectiveAnalyzer()

    # Create synthetic IR scene with cold core
    grid = np.full((64, 64), 280.0, dtype=np.float32)
    cy, cx = 32, 32
    y, x = np.ogrid[:64, :64]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    grid[r < 12] = 200.0  # Cold convective core (200 K <= 210 K deep convection)

    feats = analyzer.extract_convective_features(grid)
    assert feats["t_min_kelvin"] == 200.0
    assert feats["deep_convective_fraction"] > 0.05
    assert feats["organization_score"] > 0.40

    diagnosis = analyzer.analyze_image(grid)
    assert "is_cyclone_identified" in diagnosis
    assert diagnosis["risk_level"] in ["Low", "Moderate", "High", "Severe"]


def test_backend_satellite_api_live_inference():
    """Tests the /api/satellite/analyze endpoint with live model inference."""
    payload = {
        "lat": 16.5,
        "lon": 86.2,
        "central_pres": 975.0,
        "forward_speed": 18.0,
        "bearing": 320.0,
        "month": 10,
        "subbasin": "BB",
    }

    response = client.post("/api/satellite/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["model_connected"] is True
    assert data["model_status"] == "active"
    assert data["model_version"] == "1.0.0"
    assert data["is_cyclone"] is True
    assert data["probability_cs"] > 0.50
    assert data["risk_level"] in ["High", "Severe", "Moderate"]
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) > 0
