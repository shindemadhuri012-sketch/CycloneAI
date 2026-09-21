"""
Schemas for satellite image analysis and cyclone identification endpoints.
"""
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


class SatelliteAnalysisRequest(BaseModel):
    image_name: Optional[str] = Field(None, description="Uploaded or reference satellite image identifier")
    sensor: Optional[str] = Field("INSAT-3D", description="Satellite sensor name (e.g. INSAT-3D, INSAT-3DR, GridSat-B1)")
    channel: Optional[str] = Field("TIR-1", description="Spectral channel (e.g. TIR-1, TIR-2, WV, VIS)")
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Vortex latitude in decimal degrees")
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Vortex longitude in decimal degrees")
    central_pres: Optional[float] = Field(None, ge=850.0, le=1025.0, description="Minimum central sea-level pressure in hPa")
    forward_speed: Optional[float] = Field(None, ge=0.0, le=200.0, description="Forward translation speed in km/h")
    bearing: Optional[float] = Field(None, ge=0.0, le=360.0, description="Translation heading angle in degrees")
    month: Optional[int] = Field(None, ge=1, le=12, description="Observation calendar month (1-12)")
    subbasin: Optional[str] = Field("BB", description="Subbasin identifier ('BB' for Bay of Bengal, 'AS' for Arabian Sea)")


class SatelliteAnalysisResponse(BaseModel):
    status: str = Field(default="success", description="Status code of the request")
    service: str = Field(default="Satellite Analysis & Cyclone Identification Service", description="Originating service")
    model_status: str = Field(default="active", description="Current status of the AI model")
    model_connected: bool = Field(default=True, description="Flag indicating if live model is loaded")
    model_version: str = Field(default="1.0.0", description="Loaded model version")
    is_cyclone: Optional[bool] = Field(None, description="Binary prediction: True if Cyclonic Storm present (>= 34 kt)")
    confidence: Optional[float] = Field(None, description="Model prediction confidence score [0.0 - 1.0]")
    probability_cs: Optional[float] = Field(None, description="Posterior calibrated probability of Cyclonic Storm")
    risk_level: Optional[str] = Field(None, description="Operational disaster risk category (Minimal, Low, Moderate, High, Severe)")
    diagnosis: Optional[str] = Field(None, description="Meteorological identification summary")
    detection_result: Optional[str] = Field(None, description="Categorical identification result string")
    convective_features: Optional[Dict[str, Any]] = Field(None, description="Morphological satellite convective metrics")
    top_contributing_features: Optional[List[Dict[str, Any]]] = Field(None, description="Physics feature attribution breakdown")
    gradcam_generated: bool = Field(default=False, description="Whether Grad-CAM XAI map was generated")
    message: str = Field(
        default="Identification inference executed successfully using trained Phase 4 model checkpoint.",
        description="Execution message"
    )
