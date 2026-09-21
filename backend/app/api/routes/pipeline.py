"""
Pipeline orchestration route.
Provides unified single-request inference across all predictive engines.
"""
from fastapi import APIRouter, Depends
from app.schemas.pipeline import PipelineRunRequest, PipelineRunResponse
from app.services.pipeline_service import PipelineOrchestratorService
from app.api.dependencies import get_pipeline_service

router = APIRouter(prefix="/pipeline", tags=["Pipeline Orchestrator"])


@router.post("/run", response_model=PipelineRunResponse, summary="Execute Unified Multi-Model Pipeline")
async def run_unified_pipeline(
    request: PipelineRunRequest,
    service: PipelineOrchestratorService = Depends(get_pipeline_service)
) -> PipelineRunResponse:
    """
    Executes the entire end-to-end analytical cyclone assessment:
    1. Identification (CycloneDetector)
    2. Pattern & Category Classification (CycloneClassifier - 8 IMD categories)
    3. Multi-Horizon Intensity Forecasting (Vmax, Pmin +6h/+12h/+24h & RI Alert)
    4. Multi-Horizon Track Prediction (+6h/+12h/+24h/+48h & 75% uncertainty cones)
    5. Explainable AI (Tree-Path Saabas attribution waterfall, Dvorak BD thermal saliency, synoptic briefing)
    """
    return service.run_pipeline(request)
