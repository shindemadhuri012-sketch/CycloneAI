"""
Satellite imagery analysis routes.
"""
from fastapi import APIRouter, Depends
from app.schemas.satellite import SatelliteAnalysisRequest, SatelliteAnalysisResponse
from app.services.satellite_service import SatelliteService
from app.api.dependencies import get_satellite_service

router = APIRouter(prefix="/satellite", tags=["Satellite Analysis"])

@router.post("/analyze", response_model=SatelliteAnalysisResponse, summary="Analyze Satellite Image")
async def analyze_satellite(
    request: SatelliteAnalysisRequest,
    service: SatelliteService = Depends(get_satellite_service)
) -> SatelliteAnalysisResponse:
    """
    Submits a satellite image for cyclone identification and spatial feature extraction.
    Currently indicates that the AI model is awaiting training (Phase 4).
    """
    return service.analyze_satellite_image(request)
