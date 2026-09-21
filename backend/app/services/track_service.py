"""
Service layer for cyclone track trajectory forecasting.
Integrates trained Phase 7 CycloneTrackPredictor model supporting multi-horizon
displacement regression (+6h, +12h, +24h, +48h), physical speed/turn limits,
and empirical uncertainty cone polygon assembly.
"""
import os
import sys
import logging
from typing import Optional

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.track import TrackPredictionRequest, TrackPredictionResponse
from ml.features.extractor import CycloneFeatureExtractor, BASE_PRESSURE_HPA
from ml.models.track_predictor import CycloneTrackPredictor

logger = logging.getLogger(__name__)


class TrackService:
    def __init__(self):
        self.predictor: Optional[CycloneTrackPredictor] = None
        self.feature_extractor: CycloneFeatureExtractor = CycloneFeatureExtractor()
        self.is_loaded: bool = False
        self.model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_track.joblib")
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained Phase 7 CycloneTrackPredictor checkpoint."""
        try:
            if os.path.exists(self.model_path):
                self.predictor = CycloneTrackPredictor.load(self.model_path)
                self.is_loaded = True
                logger.info(f"Successfully loaded CycloneTrackPredictor from {self.model_path}")
            else:
                logger.warning(f"Track model checkpoint not found at {self.model_path}. Running in uninitialized mode.")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load CycloneTrackPredictor checkpoint: {e}")
            self.is_loaded = False

    def predict_track(self, request: TrackPredictionRequest) -> TrackPredictionResponse:
        """
        Predicts future coordinate progression (+6h, +12h, +24h, +48h),
        forward translation speeds, heading bearings, and uncertainty cones.
        """
        if not self.is_loaded or self.predictor is None:
            self._load_model()

        if not self.is_loaded or self.predictor is None:
            return TrackPredictionResponse(
                status="model_unavailable",
                service="Cyclone Track Prediction Service",
                model_status="not_connected",
                model_connected=False,
                message="Cyclone track prediction model checkpoint is not loaded on server.",
                initial_position=None,
                predicted_track=[],
                track_error_cone=[],
                total_horizons=0,
            )

        # 1. Resolve current position and kinematic history
        current_lat = 15.0
        current_lon = 88.0
        past_dlat_6h = 0.5
        past_dlon_6h = -0.4

        if request.lat is not None and request.lon is not None:
            current_lat = float(request.lat)
            current_lon = float(request.lon)
        elif request.past_track and len(request.past_track) > 0:
            current_lat = float(request.past_track[-1].latitude)
            current_lon = float(request.past_track[-1].longitude)
            if len(request.past_track) >= 2:
                prev = request.past_track[-2]
                past_dlat_6h = current_lat - float(prev.latitude)
                past_dlon_6h = current_lon - float(prev.longitude)

        # 2. Prepare base observation dictionary
        central_p = request.central_pres if request.central_pres is not None else 995.0
        deficit = max(0.0, BASE_PRESSURE_HPA - central_p)
        obs_dict = {
            "lat": current_lat,
            "lon": current_lon,
            "forward_speed": request.forward_speed if request.forward_speed is not None else 16.0,
            "dist_km": 45.0,
            "dt_hours": 3.0,
            "bearing": request.bearing if request.bearing is not None else 315.0,
            "central_pres": central_p,
            "pressure_deficit": deficit,
            "month": request.month if request.month is not None else 10,
            "day_of_year": 290.0,
            "subbasin": request.subbasin if request.subbasin is not None else "BB",
        }

        # 3. Extract features and add track-specific fields
        X_df = self.feature_extractor.extract_from_dict(obs_dict)
        X_df["past_dlat_6h"] = past_dlat_6h
        X_df["past_dlon_6h"] = past_dlon_6h
        X_df["past_dlat_12h"] = past_dlat_6h * 2.0
        X_df["past_dlon_12h"] = past_dlon_6h * 2.0

        if request.current_wind_speed_knots is not None:
            X_df["current_wind"] = float(request.current_wind_speed_knots)
        else:
            X_df["current_wind"] = 14.2 * (deficit ** 0.5)

        # 4. Generate multi-horizon track prediction
        forecast = self.predictor.predict_track(
            current_lat=current_lat,
            current_lon=current_lon,
            obs_features=X_df,
        )

        return TrackPredictionResponse(
            status="success",
            service="Cyclone Track Prediction Service",
            model_status="active",
            model_connected=True,
            model_version="1.0.0",
            initial_position=forecast["initial_position"],
            predicted_track=forecast["predicted_track"],
            track_error_cone=forecast["track_error_cone"],
            total_horizons=forecast["total_horizons"],
            message="Multi-horizon track trajectory forecast generated successfully using trained Phase 7 model checkpoint."
        )


track_service = TrackService()
