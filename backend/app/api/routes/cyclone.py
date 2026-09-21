"""
Cyclone classification routes.
"""
from fastapi import APIRouter, Depends
from app.schemas.cyclone import CycloneClassificationRequest, CycloneClassificationResponse
from app.services.classification_service import ClassificationService
from app.api.dependencies import get_classification_service

router = APIRouter(prefix="/cyclone", tags=["Cyclone Classification"])

@router.post("/classify", response_model=CycloneClassificationResponse, summary="Classify Cyclone Category")
async def classify_cyclone(
    request: CycloneClassificationRequest,
    service: ClassificationService = Depends(get_classification_service)
) -> CycloneClassificationResponse:
    """
    Submits storm observations for category and structural classification.
    Currently indicates that the classification model is awaiting training (Phase 5).
    """
    return service.classify_cyclone(request)
