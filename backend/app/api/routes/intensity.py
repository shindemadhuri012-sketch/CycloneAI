"""
Cyclone intensity prediction routes.
"""
from fastapi import APIRouter, Depends
from app.schemas.intensity import IntensityPredictionRequest, IntensityPredictionResponse
from app.services.intensity_service import IntensityService
from app.api.dependencies import get_intensity_service

router = APIRouter(prefix="/intensity", tags=["Intensity Prediction"])

@router.post("/predict", response_model=IntensityPredictionResponse, summary="Predict Cyclone Intensity")
async def predict_intensity(
    request: IntensityPredictionRequest,
    service: IntensityService = Depends(get_intensity_service)
) -> IntensityPredictionResponse:
    """
    Submits storm observations for intensity regression and trend forecasting.
    Currently indicates that the intensity model is awaiting training (Phase 6).
    """
    return service.predict_intensity(request)
