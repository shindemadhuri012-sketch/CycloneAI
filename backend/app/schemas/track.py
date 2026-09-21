"""
Schemas for cyclone track trajectory prediction endpoints.
Supports multi-horizon waypoints (+6h, +12h, +24h, +48h), kinematic attributes,
and GeoJSON/Leaflet-compatible uncertainty cone polygons.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CoordinatePoint(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Vortex latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Vortex longitude in decimal degrees")
    timestamp: Optional[str] = Field(None, description="Observation timestamp (ISO 8601)")
    intensity_knots: Optional[float] = Field(None, ge=10.0, le=200.0, description="Sustained wind speed in knots")


class TrackPredictionRequest(BaseModel):
    storm_id: Optional[str] = Field(None, description="Storm identifier or unique observation ID")
    lat: Optional[float] = Field(None, ge=0.0, le=40.0, description="Current storm latitude in decimal degrees")
    lon: Optional[float] = Field(None, ge=40.0, le=110.0, description="Current storm longitude in decimal degrees")
    past_track: Optional[List[CoordinatePoint]] = Field(default=[], description="Historical sequence of vortex fixes")
    forward_speed: Optional[float] = Field(None, ge=0.0, le=150.0, description="Current forward translation speed in km/h")
    bearing: Optional[float] = Field(None, ge=0.0, le=360.0, description="Current compass heading bearing in degrees")
    central_pres: Optional[float] = Field(None, ge=850.0, le=1025.0, description="Central pressure in hPa")
    current_wind_speed_knots: Optional[float] = Field(None, ge=10.0, le=200.0, description="Current wind speed in knots")
    month: Optional[int] = Field(None, ge=1, le=12, description="Observation month (1-12)")
    subbasin: Optional[str] = Field("BB", description="Subbasin: 'BB' for Bay of Bengal, 'AS' for Arabian Sea")


class TrackPredictionResponse(BaseModel):
    status: str = Field(default="success", description="Status code of track prediction request")
    service: str = Field(default="Cyclone Track Prediction Service", description="Originating service")
    model_status: str = Field(default="active", description="Current status of the AI model")
    model_connected: bool = Field(default=True, description="Flag indicating if live model is loaded")
    model_version: str = Field(default="1.0.0", description="Loaded model version")
    initial_position: Optional[Dict[str, float]] = Field(default=None, description="Reference coordinates for forecast")
    predicted_track: List[Dict[str, Any]] = Field(
        default=[],
        description="Predicted coordinates sequence (+6h, +12h, +24h, +48h) with speeds and bearings"
    )
    track_error_cone: List[Dict[str, Any]] = Field(
        default=[],
        description="Uncertainty cone boundary polygon coordinates and radii"
    )
    total_horizons: Optional[int] = Field(default=None, description="Number of predicted horizons")
    message: str = Field(
        default="Multi-horizon track trajectory forecast generated successfully using trained Phase 7 model checkpoint.",
        description="Execution message"
    )
