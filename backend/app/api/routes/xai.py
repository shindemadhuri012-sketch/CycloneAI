"""
Explainable AI (XAI) routes.
Provides transparent model explanations, Dvorak convective saliency,
and physical sensitivity analysis.
"""
from fastapi import APIRouter, Depends
from app.schemas.xai import (
    XAIExplainRequest,
    XAIExplainResponse,
    XAISaliencyRequest,
    XAISaliencyResponse,
    XAISensitivityRequest,
    XAISensitivityResponse,
)
from app.services.xai_service import XAIService
from app.api.dependencies import get_xai_service

router = APIRouter(prefix="/xai", tags=["Explainable AI"])


@router.post("/explain", response_model=XAIExplainResponse, summary="Explain Cyclone Predictions")
async def explain_cyclone_prediction(
    request: XAIExplainRequest,
    service: XAIService = Depends(get_xai_service),
) -> XAIExplainResponse:
    """
    Computes local feature attributions, waterfall steps, counterfactual shifts,
    and natural language synoptic diagnostic briefings for an observation fix.
    """
    return service.explain(request)


@router.post("/saliency", response_model=XAISaliencyResponse, summary="Compute Satellite Convective Saliency")
async def compute_satellite_saliency(
    request: XAISaliencyRequest,
    service: XAIService = Depends(get_xai_service),
) -> XAISaliencyResponse:
    """
    Computes Dvorak BD enhancement zones, 2D convective saliency matrix,
    and eyewall axisymmetry indicators from satellite thermal infrared data.
    """
    return service.compute_saliency(request)


@router.post("/sensitivity", response_model=XAISensitivityResponse, summary="Evaluate Parameter Sensitivity")
async def compute_parameter_sensitivity(
    request: XAISensitivityRequest,
    service: XAIService = Depends(get_xai_service),
) -> XAISensitivityResponse:
    """
    Evaluates 1D/2D partial dependence sensitivity curves for physical meteorological parameters.
    """
    return service.compute_sensitivity(request)
