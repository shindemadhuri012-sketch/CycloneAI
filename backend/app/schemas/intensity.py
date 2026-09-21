"""
Schemas for cyclone intensity prediction and forecasting endpoints.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class IntensityPredictionRequest(BaseModel):
    storm_id: Optional[str] = Field(None, description="Storm identifier or observation ID")
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Vortex latitude in decimal degrees")
    lon: Optional[float] = Field(None, ge=-180.0, le=180.0, description="Vortex longitude in decimal degrees")
    current_wind_speed_knots: Optional[float] = Field(None, ge=10.0, le=200.0, description="Current estimated sustained wind speed in knots")
    central_pressure_hpa: Optional[float] = Field(None, ge=850.0, le=1025.0, description="Current central atmospheric pressure in hPa")
    forward_speed: Optional[float] = Field(None, ge=0.0, le=200.0, description="Forward translation speed in km/h")
    bearing: Optional[float] = Field(None, ge=0.0, le=360.0, description="Movement direction angle in degrees")
    month: Optional[int] = Field(None, ge=1, le=12, description="Observation calendar month (1-12)")
    subbasin: Optional[str] = Field("BB", description="Subbasin: 'BB' for Bay of Bengal, 'AS' for Arabian Sea")


class IntensityPredictionResponse(BaseModel):
    status: str = Field(default="success", description="Status code of intensity prediction request")
    service: str = Field(default="Cyclone Intensity Prediction Service", description="Originating service")
    model_status: str = Field(default="active", description="Current status of the AI model")
    model_connected: bool = Field(default=True, description="Flag indicating if live model is loaded")
    model_version: str = Field(default="1.0.0", description="Loaded model version")
    current_intensity: Optional[Dict[str, Any]] = Field(default=None, description="Current analyzed intensity values")
    forecast_horizons: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Predicted future wind speeds, central pressures, and 90% confidence bounds (+6h, +12h, +24h)"
    )
    rapid_intensification: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Rapid Intensification (RI) risk assessment and probability (dV_24h >= 30 kt)"
    )
    intensity_trend: Optional[str] = Field(default=None, description="Diagnostic trajectory trend: Intensifying, Steady, Weakening")
    message: str = Field(
        default="Multi-horizon intensity forecast generated successfully using trained Phase 6 model checkpoint.",
        description="Execution message"
    )
