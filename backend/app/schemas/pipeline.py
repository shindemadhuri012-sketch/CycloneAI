"""
Schemas for the unified cyclone analysis pipeline.
Orchestrates identification, classification, intensity forecasting,
track trajectory prediction, and explainable AI in a single request.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.track import CoordinatePoint


class PipelineRunRequest(BaseModel):
    storm_id: Optional[str] = Field(None, description="Storm identifier or operational name (e.g. FANI, AMPHAN)")
    lat: float = Field(..., ge=0.0, le=40.0, description="Current storm eye latitude in decimal degrees (0°N - 40°N)")
    lon: float = Field(..., ge=40.0, le=110.0, description="Current storm eye longitude in decimal degrees (40°E - 110°E)")
    central_pres: float = Field(..., ge=850.0, le=1025.0, description="Central sea-level pressure in hPa")
    current_wind_speed_knots: Optional[float] = Field(None, ge=10.0, le=200.0, description="Estimated current sustained wind speed in knots")
    forward_speed: float = Field(15.0, ge=0.0, le=150.0, description="Forward translation speed in km/h")
    bearing: float = Field(315.0, ge=0.0, le=360.0, description="Heading compass direction in degrees (0° - 360°)")
    month: Optional[int] = Field(None, ge=1, le=12, description="Observation calendar month (1 - 12)")
    subbasin: Optional[str] = Field("BB", description="Ocean subbasin: 'BB' (Bay of Bengal) or 'AS' (Arabian Sea)")
    past_track: Optional[List[CoordinatePoint]] = Field(default=[], description="Historical track waypoints if available")


class PipelineRunResponse(BaseModel):
    status: str = Field(default="success", description="Status code of the pipeline execution")
    storm_id: Optional[str] = Field(None, description="Analyzed storm identifier")
    timestamp: str = Field(..., description="Execution timestamp (ISO 8601)")
    execution_time_ms: float = Field(..., description="Wall-clock execution latency in milliseconds")
    
    # 1. Identification
    identification: Dict[str, Any] = Field(..., description="Cyclone detection results and probability")
    
    # 2. Classification
    classification: Dict[str, Any] = Field(..., description="IMD category classification and class probabilities")
    
    # 3. Intensity Forecasting
    intensity: Dict[str, Any] = Field(..., description="Multi-horizon intensity forecasts with 90% bounds and RI status")
    
    # 4. Track Prediction
    track: Dict[str, Any] = Field(..., description="Multi-horizon track waypoints and 75% uncertainty cone polygons")
    
    # 5. Explainable AI
    explainability: Dict[str, Any] = Field(..., description="Feature attribution waterfall, Dvorak metrics, and synoptic briefing")
    
    message: str = Field(
        default="Unified multi-model cyclone pipeline executed successfully across all Phase 4-8 engines.",
        description="Execution message"
    )
