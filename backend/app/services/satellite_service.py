"""
Service layer for satellite data processing and cyclone identification model inference.
Integrates trained Phase 4 CycloneDetector model and morphological satellite convective analyzer.
"""
import os
import sys
import logging
from typing import Optional
import numpy as np

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.satellite import SatelliteAnalysisRequest, SatelliteAnalysisResponse
from ml.features.extractor import CycloneFeatureExtractor
from ml.models.detector import CycloneDetector
from ml.models.satellite_detector import SatelliteConvectiveAnalyzer

logger = logging.getLogger(__name__)


class SatelliteService:
    def __init__(self):
        self.detector: Optional[CycloneDetector] = None
        self.convective_analyzer: SatelliteConvectiveAnalyzer = SatelliteConvectiveAnalyzer()
        self.feature_extractor: CycloneFeatureExtractor = CycloneFeatureExtractor()
        self.is_loaded: bool = False
        self.model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_detector.joblib")
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained Phase 4 CycloneDetector checkpoint."""
        try:
            if os.path.exists(self.model_path):
                self.detector = CycloneDetector.load(self.model_path)
                self.is_loaded = True
                logger.info(f"Successfully loaded CycloneDetector from {self.model_path}")
            else:
                logger.warning(f"Model checkpoint not found at {self.model_path}. Running in uninitialized mode.")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load CycloneDetector checkpoint: {e}")
            self.is_loaded = False

    def analyze_satellite_image(self, request: SatelliteAnalysisRequest) -> SatelliteAnalysisResponse:
        """
        Processes observation parameters and satellite metadata to identify cyclone presence.
        Uses trained Phase 4 CycloneDetector and physics-grounded convective feature extraction.
        """
        # If model is not loaded, attempt lazy reload
        if not self.is_loaded or self.detector is None:
            self._load_model()

        if not self.is_loaded or self.detector is None:
            return SatelliteAnalysisResponse(
                status="model_unavailable",
                service="Satellite Analysis & Cyclone Identification Service",
                model_status="not_connected",
                model_connected=False,
                message="Identification model checkpoint not loaded on server.",
                is_cyclone=None,
                confidence=None,
            )

        # 1. Prepare observation dictionary from request parameters
        obs_dict = {
            "lat": request.lat if request.lat is not None else 14.5,
            "lon": request.lon if request.lon is not None else 86.0,
            "central_pres": request.central_pres if request.central_pres is not None else 1002.0,
            "forward_speed": request.forward_speed if request.forward_speed is not None else 15.0,
            "bearing": request.bearing if request.bearing is not None else 310.0,
            "month": request.month if request.month is not None else 10,
            "subbasin": request.subbasin if request.subbasin is not None else "BB",
        }

        # 2. Extract feature matrix
        X_df = self.feature_extractor.extract_from_dict(obs_dict)

        # 3. Model Inference (calibrated probability)
        probs = self.detector.predict_proba(X_df)[0]
        prob_cs = float(probs[1])
        is_cyclone_std = bool(prob_cs >= self.detector.standard_threshold)
        is_cyclone_ops = bool(prob_cs >= self.detector.operational_threshold)

        # 4. Feature attribution & explainability
        explanation = self.detector.explain_sample(X_df)
        risk_level = explanation["risk_level"]
        top_features = explanation["top_features"]

        # 5. Diagnostic string
        if is_cyclone_std:
            diagnosis = (
                f"Organized Cyclonic Storm identified (P = {prob_cs*100:.1f}%). "
                f"Sustained winds >= 34 kt indicated with {risk_level.lower()} coastal threat."
            )
            detection_result = "CYCLONIC_STORM"
            confidence = prob_cs
        elif is_cyclone_ops:
            diagnosis = (
                f"Formative Tropical Depression / Alert threshold exceeded (P = {prob_cs*100:.1f}%). "
                f"System exceeds operational disaster alert threshold ({self.detector.operational_threshold:.2f})."
            )
            detection_result = "TROPICAL_DEPRESSION_ALERT"
            confidence = prob_cs
        else:
            diagnosis = (
                f"No organized cyclonic storm detected (P = {prob_cs*100:.1f}%). "
                f"Observation corresponds to non-cyclonic disturbance or low-pressure area."
            )
            detection_result = "NON_CYCLONIC_DISTURBANCE"
            confidence = float(1.0 - prob_cs)

        # 6. Satellite Convective Analysis (simulated representative IR grid if an image identifier is given)
        convective_feats = None
        if request.image_name:
            # Generate synthetic representative brightness field for visual check
            # Centered around the given core pressure
            mock_grid = np.full((64, 64), 280.0, dtype=np.float32)
            # Cold convective core
            cy, cx = 32, 32
            y, x = np.ogrid[:64, :64]
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            # Depressed temperature based on central pressure
            core_t = max(195.0, 195.0 + (obs_dict["central_pres"] - 950.0) * 0.8)
            mock_grid[r < 16] = core_t
            convective_feats = self.convective_analyzer.extract_convective_features(mock_grid)

        return SatelliteAnalysisResponse(
            status="success",
            service="Satellite Analysis & Cyclone Identification Service",
            model_status="active",
            model_connected=True,
            model_version="1.0.0",
            is_cyclone=is_cyclone_std,
            confidence=round(confidence, 4),
            probability_cs=round(prob_cs, 4),
            risk_level=risk_level,
            diagnosis=diagnosis,
            detection_result=detection_result,
            convective_features=convective_feats,
            top_contributing_features=top_features,
            gradcam_generated=False,
            message="Identification inference executed successfully using trained Phase 4 model checkpoint."
        )


satellite_service = SatelliteService()
