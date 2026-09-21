"""
Cyclone track prediction routes.
"""
from fastapi import APIRouter, Depends
from app.schemas.track import TrackPredictionRequest, TrackPredictionResponse
from app.services.track_service import TrackService
from app.api.dependencies import get_track_service

router = APIRouter(prefix="/track", tags=["Track Prediction"])

@router.post("/predict", response_model=TrackPredictionResponse, summary="Predict Cyclone Track")
async def predict_track(
    request: TrackPredictionRequest,
    service: TrackService = Depends(get_track_service)
) -> TrackPredictionResponse:
    """
    Submits storm coordinates sequence for future trajectory path forecasting.
    Currently indicates that the track model is awaiting training (Phase 7).
    """
    return service.predict_track(request)
