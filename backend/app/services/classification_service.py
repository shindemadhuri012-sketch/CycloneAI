"""
Service layer for cyclone pattern and category classification model inference.
Integrates trained Phase 5 CycloneClassifier model supporting the 8 official IMD categories.
"""
import os
import sys
import logging
from typing import Optional

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.cyclone import CycloneClassificationRequest, CycloneClassificationResponse
from ml.features.extractor import CycloneFeatureExtractor
from ml.models.classifier import CycloneClassifier, IMD_CATEGORIES

logger = logging.getLogger(__name__)


class ClassificationService:
    def __init__(self):
        self.classifier: Optional[CycloneClassifier] = None
        self.feature_extractor: CycloneFeatureExtractor = CycloneFeatureExtractor()
        self.is_loaded: bool = False
        self.model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained Phase 5 CycloneClassifier checkpoint."""
        try:
            if os.path.exists(self.model_path):
                self.classifier = CycloneClassifier.load(self.model_path)
                self.is_loaded = True
                logger.info(f"Successfully loaded CycloneClassifier from {self.model_path}")
            else:
                logger.warning(f"Model checkpoint not found at {self.model_path}. Running in uninitialized mode.")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load CycloneClassifier checkpoint: {e}")
            self.is_loaded = False

    def classify_cyclone(self, request: CycloneClassificationRequest) -> CycloneClassificationResponse:
        """
        Classifies tropical cyclone pattern into one of the 8 official IMD categories.
        Uses trained Phase 5 CycloneClassifier and physics-grounded feature extraction.
        """
        if not self.is_loaded or self.classifier is None:
            self._load_model()

        if not self.is_loaded or self.classifier is None:
            return CycloneClassificationResponse(
                status="model_unavailable",
                service="Cyclone Classification Service",
                model_status="not_connected",
                model_connected=False,
                message="Cyclone classification model checkpoint is not loaded on server.",
                predicted_category=None,
                confidence=None,
            )

        # 1. Prepare observation dictionary
        obs_dict = {
            "lat": request.lat if request.lat is not None else 15.0,
            "lon": request.lon if request.lon is not None else 87.0,
            "central_pres": request.central_pres if request.central_pres is not None else 995.0,
            "forward_speed": request.forward_speed if request.forward_speed is not None else 16.0,
            "bearing": request.bearing if request.bearing is not None else 315.0,
            "month": request.month if request.month is not None else 10,
            "subbasin": request.subbasin if request.subbasin is not None else "BB",
        }

        # 2. Extract feature matrix
        X_df = self.feature_extractor.extract_from_dict(obs_dict)

        # 3. Model inference and diagnostic explanation
        explanation = self.classifier.explain_sample(X_df)

        return CycloneClassificationResponse(
            status="success",
            service="Cyclone Classification Service",
            model_status="active",
            model_connected=True,
            model_version="1.0.0",
            predicted_category=explanation["predicted_category"],
            category_code=explanation["category_code"],
            category_index=explanation["category_index"],
            confidence=explanation["confidence"],
            wind_range_kt=explanation["wind_range_kt"],
            wind_range_kmh=explanation["wind_range_kmh"],
            damage_potential=explanation["damage_potential"],
            all_probabilities=explanation["all_probabilities"],
            top_categories=explanation["top_categories"],
            top_contributing_features=explanation["top_features"],
            message="Cyclone category classification executed successfully using trained Phase 5 model checkpoint."
        )


classification_service = ClassificationService()
