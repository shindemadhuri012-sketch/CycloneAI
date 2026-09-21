"""
Health check route.
"""
from fastapi import APIRouter
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Backend Health Check")
async def get_health() -> HealthResponse:
    """
    Returns operational health and ML model connection status across all phases.
    """
    from app.api.dependencies import (
        satellite_service, classification_service,
        intensity_service, track_service, xai_service
    )
    models_state = {
        "cyclone_detector": satellite_service.is_loaded,
        "cyclone_classifier": classification_service.is_loaded,
        "cyclone_intensity": intensity_service.is_loaded,
        "cyclone_track": track_service.is_loaded,
        "explainable_ai": xai_service.is_loaded
    }
    all_loaded = all(models_state.values())
    return HealthResponse(
        status="ok",
        service="CycloneAI Backend",
        model_status="active" if all_loaded else "degraded",
        active_models=models_state
    )
