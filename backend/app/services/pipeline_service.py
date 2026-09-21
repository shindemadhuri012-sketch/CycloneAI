"""
PipelineOrchestratorService: Unified multi-model coordination engine.
Executes Identification, Classification, Intensity Forecasting,
Track Trajectory Prediction, and Explainable AI in a single coordinated pass.
"""
import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.pipeline import PipelineRunRequest, PipelineRunResponse
from app.schemas.satellite import SatelliteAnalysisRequest
from app.schemas.cyclone import CycloneClassificationRequest
from app.schemas.intensity import IntensityPredictionRequest
from app.schemas.track import TrackPredictionRequest
from app.schemas.xai import XAIExplainRequest, XAISaliencyRequest

from app.services.satellite_service import SatelliteService
from app.services.classification_service import ClassificationService
from app.services.intensity_service import IntensityService
from app.services.track_service import TrackService
from app.services.xai_service import XAIService

logger = logging.getLogger(__name__)


class PipelineOrchestratorService:
    def __init__(
        self,
        satellite_service: SatelliteService,
        classification_service: ClassificationService,
        intensity_service: IntensityService,
        track_service: TrackService,
        xai_service: XAIService,
    ):
        self.satellite_service = satellite_service
        self.classification_service = classification_service
        self.intensity_service = intensity_service
        self.track_service = track_service
        self.xai_service = xai_service

    def run_pipeline(self, request: PipelineRunRequest) -> PipelineRunResponse:
        """
        Executes the entire multi-model pipeline in a single synchronized pass.
        Returns unified holistic analysis dossier with sub-100ms latency.
        """
        t0 = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Identification (Phase 4)
        sat_req = SatelliteAnalysisRequest(
            image_name=f"{request.storm_id or 'cyclone'}_granule.tif",
            sensor="INSAT-3D",
            channel="TIR-1",
            lat=request.lat,
            lon=request.lon,
            central_pres=request.central_pres,
            forward_speed=request.forward_speed,
            bearing=request.bearing,
            month=request.month,
            subbasin=request.subbasin
        )
        sat_res = self.satellite_service.analyze_satellite_image(sat_req)

        # 2. Classification (Phase 5)
        cls_req = CycloneClassificationRequest(
            storm_id=request.storm_id,
            basin=request.subbasin,
            lat=request.lat,
            lon=request.lon,
            central_pres=request.central_pres,
            forward_speed=request.forward_speed,
            bearing=request.bearing,
            month=request.month,
            subbasin=request.subbasin
        )
        cls_res = self.classification_service.classify_cyclone(cls_req)

        # Estimate current wind speed if not provided
        current_wind = request.current_wind_speed_knots
        if current_wind is None:
            # Derived from pressure deficit or classification median
            dp = max(0.0, 1010.0 - request.central_pres)
            current_wind = round(14.2 * (dp ** 0.5), 1)

        # 3. Intensity Forecasting (Phase 6)
        int_req = IntensityPredictionRequest(
            storm_id=request.storm_id,
            lat=request.lat,
            lon=request.lon,
            current_wind_speed_knots=current_wind,
            central_pressure_hpa=request.central_pres,
            forward_speed=request.forward_speed,
            bearing=request.bearing,
            month=request.month,
            subbasin=request.subbasin
        )
        int_res = self.intensity_service.predict_intensity(int_req)

        # 4. Track Prediction (Phase 7)
        trk_req = TrackPredictionRequest(
            storm_id=request.storm_id,
            lat=request.lat,
            lon=request.lon,
            past_track=request.past_track,
            forward_speed=request.forward_speed,
            bearing=request.bearing,
            central_pres=request.central_pres,
            current_wind_speed_knots=current_wind,
            month=request.month,
            subbasin=request.subbasin
        )
        trk_res = self.track_service.predict_track(trk_req)

        # 5. Explainable AI (Phase 8)
        xai_req = XAIExplainRequest(
            storm_id=request.storm_id,
            lat=request.lat,
            lon=request.lon,
            central_pres=request.central_pres,
            current_wind_speed_knots=current_wind,
            forward_speed=request.forward_speed,
            bearing=request.bearing,
            month=request.month or 5,
            subbasin=request.subbasin or "BB"
        )
        xai_res = self.xai_service.explain(xai_req)

        intensity_factor = min(1.5, max(0.2, current_wind / 70.0))
        sal_req = XAISaliencyRequest(
            sensor="INSAT-3D",
            channel="TIR-1",
            intensity_factor=intensity_factor
        )
        sal_res = self.xai_service.compute_saliency(sal_req)

        t1 = time.perf_counter()
        execution_time_ms = round((t1 - t0) * 1000.0, 2)

        return PipelineRunResponse(
            status="success",
            storm_id=request.storm_id,
            timestamp=timestamp,
            execution_time_ms=execution_time_ms,
            identification={
                "is_cyclone": sat_res.is_cyclone,
                "confidence": sat_res.confidence,
                "probability_cs": sat_res.probability_cs,
                "risk_level": sat_res.risk_level,
                "diagnosis": sat_res.diagnosis,
                "convective_features": sat_res.convective_features
            },
            classification={
                "predicted_category": cls_res.predicted_category,
                "category_code": cls_res.category_code,
                "category_index": cls_res.category_index,
                "confidence": cls_res.confidence,
                "wind_range_kt": cls_res.wind_range_kt,
                "damage_potential": cls_res.damage_potential,
                "all_probabilities": cls_res.all_probabilities,
                "top_categories": cls_res.top_categories
            },
            intensity={
                "current_intensity": int_res.current_intensity,
                "forecast_horizons": int_res.forecast_horizons,
                "rapid_intensification": int_res.rapid_intensification,
                "intensity_trend": int_res.intensity_trend
            },
            track={
                "initial_position": trk_res.initial_position,
                "predicted_track": trk_res.predicted_track,
                "track_error_cone": trk_res.track_error_cone,
                "total_horizons": trk_res.total_horizons
            },
            explainability={
                "top_drivers": xai_res.classification_explanation.get("top_drivers", []),
                "synoptic_briefing": xai_res.synoptic_narrative,
                "axisymmetry_score": sal_res.axisymmetry_score,
                "eye_contrast_celsius": sal_res.eye_contrast_celsius,
                "eyewall_min_temp_celsius": sal_res.eyewall_min_temp_celsius,
                "eye_max_temp_celsius": sal_res.eye_max_temp_celsius,
                "counterfactual_summary": xai_res.counterfactual_summary,
                "dvorak_bd_distribution": [b if isinstance(b, dict) else b.model_dump() for b in sal_res.dvorak_bd_distribution]
            },
            message=f"Pipeline executed in {execution_time_ms} ms across all Phase 4-8 ML engines."
        )
