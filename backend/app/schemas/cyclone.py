"""
Schemas for cyclone pattern and category classification endpoints.
"""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


class CycloneClassificationRequest(BaseModel):
    storm_id: Optional[str] = Field(None, description="Storm identifier or observation ID")
    basin: Optional[str] = Field("North Indian Ocean", description="Ocean basin of interest")
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Vortex latitude in decimal degrees")
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Vortex longitude in decimal degrees")
    central_pres: Optional[float] = Field(None, ge=850.0, le=1025.0, description="Central atmospheric pressure in hPa")
    forward_speed: Optional[float] = Field(None, ge=0.0, le=200.0, description="Forward translation speed in km/h")
    bearing: Optional[float] = Field(None, ge=0.0, le=360.0, description="Movement heading direction in degrees")
    month: Optional[int] = Field(None, ge=1, le=12, description="Calendar month of observation (1-12)")
    subbasin: Optional[str] = Field("BB", description="Subbasin identifier ('BB' for Bay of Bengal, 'AS' for Arabian Sea)")


class CycloneClassificationResponse(BaseModel):
    status: str = Field(default="success", description="Status code of classification request")
    service: str = Field(default="Cyclone Classification Service", description="Originating service")
    model_status: str = Field(default="active", description="Current status of the AI model")
    model_connected: bool = Field(default=True, description="Flag indicating if live model is loaded")
    model_version: str = Field(default="1.0.0", description="Loaded model version")
    predicted_category: Optional[str] = Field(None, description="Official IMD category name (e.g. Very Severe Cyclonic Storm)")
    category_code: Optional[str] = Field(None, description="IMD code abbreviation (e.g. VSCS, CS, D)")
    category_index: Optional[int] = Field(None, description="IMD category integer index (0 - 7)")
    confidence: Optional[float] = Field(None, description="Model prediction probability for the top category [0.0 - 1.0]")
    wind_range_kt: Optional[str] = Field(None, description="Sustained surface wind speed range in knots")
    wind_range_kmh: Optional[str] = Field(None, description="Sustained surface wind speed range in km/h")
    damage_potential: Optional[str] = Field(None, description="IMD disaster hazard description")
    all_probabilities: Optional[Dict[str, float]] = Field(None, description="Posterior probabilities across all 8 IMD categories")
    top_categories: Optional[List[Dict[str, Any]]] = Field(None, description="Ranked top probable categories with probability scores")
    top_contributing_features: Optional[List[Dict[str, Any]]] = Field(None, description="Key driving physical features")
    message: str = Field(
        default="Cyclone category classification executed successfully using trained Phase 5 model checkpoint.",
        description="Execution message"
    )
