"""
Service layer for cyclone intensity estimation and multi-horizon forecasting.
Integrates trained Phase 6 CycloneIntensityPredictor model supporting multi-horizon
wind speed (Vmax) and central pressure (Pmin) regression, 90% quantile uncertainty bounds,
Rapid Intensification (RI) alert gating, and IMD hydrodynamic physical coupling.
"""
import os
import sys
import logging
from typing import Optional

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.intensity import IntensityPredictionRequest, IntensityPredictionResponse
from ml.features.extractor import CycloneFeatureExtractor
from ml.models.intensity_predictor import CycloneIntensityPredictor

logger = logging.getLogger(__name__)


class IntensityService:
    def __init__(self):
        self.predictor: Optional[CycloneIntensityPredictor] = None
        self.feature_extractor: CycloneFeatureExtractor = CycloneFeatureExtractor()
        self.is_loaded: bool = False
        self.model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained Phase 6 CycloneIntensityPredictor checkpoint."""
        try:
            if os.path.exists(self.model_path):
                self.predictor = CycloneIntensityPredictor.load(self.model_path)
                self.is_loaded = True
                logger.info(f"Successfully loaded CycloneIntensityPredictor from {self.model_path}")
            else:
                logger.warning(f"Intensity model checkpoint not found at {self.model_path}. Running in uninitialized mode.")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load CycloneIntensityPredictor checkpoint: {e}")
            self.is_loaded = False

    def predict_intensity(self, request: IntensityPredictionRequest) -> IntensityPredictionResponse:
        """
        Generates multi-horizon intensity forecast (+6h, +12h, +24h), 90% confidence intervals,
        and Rapid Intensification (RI) risk assessment.
        """
        if not self.is_loaded or self.predictor is None:
            self._load_model()

        if not self.is_loaded or self.predictor is None:
            return IntensityPredictionResponse(
                status="model_unavailable",
                service="Cyclone Intensity Prediction Service",
                model_status="not_connected",
                model_connected=False,
                message="Cyclone intensity prediction model checkpoint is not loaded on server.",
                current_intensity=None,
                forecast_horizons=None,
                rapid_intensification=None,
                intensity_trend=None,
            )

        # 1. Prepare base observation dictionary
        obs_dict = {
            "lat": request.lat if request.lat is not None else 15.0,
            "lon": request.lon if request.lon is not None else 87.0,
            "central_pres": request.central_pressure_hpa if request.central_pressure_hpa is not None else 995.0,
            "forward_speed": request.forward_speed if request.forward_speed is not None else 16.0,
            "bearing": request.bearing if request.bearing is not None else 315.0,
            "month": request.month if request.month is not None else 10,
            "subbasin": request.subbasin if request.subbasin is not None else "BB",
        }

        # 2. Extract engineered features
        X_df = self.feature_extractor.extract_from_dict(obs_dict)
        if request.current_wind_speed_knots is not None:
            X_df["current_wind"] = float(request.current_wind_speed_knots)

        # 3. Predict multi-horizon forecast, quantiles, and RI risk
        forecast = self.predictor.predict_forecast(X_df)

        return IntensityPredictionResponse(
            status="success",
            service="Cyclone Intensity Prediction Service",
            model_status="active",
            model_connected=True,
            model_version="1.0.0",
            current_intensity=forecast["current_intensity"],
            forecast_horizons=forecast["forecast_horizons"],
            rapid_intensification=forecast["rapid_intensification"],
            intensity_trend=forecast["intensity_trend"],
            message="Multi-horizon intensity forecast generated successfully using trained Phase 6 model checkpoint."
        )


intensity_service = IntensityService()
