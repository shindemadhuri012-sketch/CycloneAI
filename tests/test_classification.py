"""
Unit and Integration Tests for Phase 5: Cyclone Pattern & Category Classification Model.
Validates multi-class prediction, probability conservation across all 8 IMD categories,
monotonic physical pressure-wind relationships, and API endpoint integration.
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
from ml.models.classifier import CycloneClassifier, IMD_CATEGORIES, IMD_GRADE_TO_INDEX
from backend.app.main import app

client = TestClient(app)


def test_classifier_loading_and_checkpoint():
    """Verifies that the serialized classification model exists and is fitted."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    assert os.path.exists(model_path), f"Checkpoint missing: {model_path}"

    classifier = CycloneClassifier.load(model_path)
    assert classifier.is_fitted is True
    assert classifier.num_classes == 8
    assert len(classifier.feature_importances_) == 14


def test_probability_conservation():
    """Verifies that predicted class probabilities strictly sum to 1.0 across all 8 categories."""
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    classifier = CycloneClassifier.load(model_path)
    extractor = CycloneFeatureExtractor()

    obs = {"lat": 15.5, "lon": 87.0, "central_pres": 970.0, "forward_speed": 18.0}
    X = extractor.extract_from_dict(obs)
    probs = classifier.predict_proba(X)

    assert probs.shape == (1, 8)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert abs(np.sum(probs) - 1.0) < 1e-4


def test_physical_monotonicity():
    """
    Physical sanity check: A severe system with core pressure 910 hPa MUST be classified
    into a higher intensity category than a 1008 hPa weak low-pressure system.
    """
    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    classifier = CycloneClassifier.load(model_path)
    extractor = CycloneFeatureExtractor()

    super_cyclone = extractor.extract_from_dict({"lat": 18.0, "lon": 88.0, "central_pres": 910.0})
    weak_low = extractor.extract_from_dict({"lat": 18.0, "lon": 88.0, "central_pres": 1008.0})

    cat_super = classifier.predict(super_cyclone)[0]
    cat_weak = classifier.predict(weak_low)[0]

    assert cat_super >= 5, f"Expected VSCS or higher (index >= 5) for 910 hPa, got {cat_super}"
    assert cat_weak <= 1, f"Expected LOW or D (index <= 1) for 1008 hPa, got {cat_weak}"
    assert cat_super > cat_weak


def test_backend_cyclone_classify_endpoint():
    """Tests POST /api/cyclone/classify endpoint with active model inference."""
    payload = {
        "lat": 17.5,
        "lon": 88.5,
        "central_pres": 960.0,
        "forward_speed": 20.0,
        "bearing": 330.0,
        "month": 10,
        "subbasin": "BB",
    }

    response = client.post("/api/cyclone/classify", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["model_connected"] is True
    assert data["model_status"] == "active"
    assert data["category_index"] is not None
    assert 0 <= data["category_index"] <= 7
    assert data["category_code"] in ["LOW", "D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]
    assert data["confidence"] > 0.0
    assert "wind_range_kt" in data
    assert "damage_potential" in data
    assert "all_probabilities" in data
    assert len(data["all_probabilities"]) == 8
