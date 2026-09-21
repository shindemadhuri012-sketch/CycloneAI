"""
Schemas for CycloneAI Explainable AI (XAI) endpoints.
Covers local feature attribution, waterfall contribution steps,
Dvorak-aligned convective saliency matrices, and marginal sensitivity curves.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class XAIExplainRequest(BaseModel):
    storm_id: Optional[str] = Field(None, description="Storm identifier or unique observation ID")
    lat: Optional[float] = Field(16.0, ge=0.0, le=40.0, description="Vortex latitude in decimal degrees")
    lon: Optional[float] = Field(87.5, ge=40.0, le=110.0, description="Vortex longitude in decimal degrees")
    central_pres: Optional[float] = Field(985.0, ge=850.0, le=1025.0, description="Central pressure in hPa")
    current_wind_speed_knots: Optional[float] = Field(60.0, ge=10.0, le=200.0, description="Current sustained wind speed in knots")
    forward_speed: Optional[float] = Field(16.0, ge=0.0, le=150.0, description="Forward translation speed in km/h")
    bearing: Optional[float] = Field(320.0, ge=0.0, le=360.0, description="Heading movement bearing in degrees")
    month: Optional[int] = Field(10, ge=1, le=12, description="Observation calendar month")
    subbasin: Optional[str] = Field("BB", description="Subbasin: 'BB' for Bay of Bengal, 'AS' for Arabian Sea")


class WaterfallStep(BaseModel):
    feature: str
    feature_value: float
    probability_contribution: Optional[float] = None
    contribution: Optional[float] = None
    direction: str


class XAIExplainResponse(BaseModel):
    status: str = Field(default="success", description="Status code of XAI request")
    service: str = Field(default="CycloneAI Explainable AI Service", description="Originating service")
    model_status: str = Field(default="active", description="Status of explainability engine")
    model_connected: bool = Field(default=True, description="Flag indicating live connection")
    model_version: str = Field(default="1.0.0", description="XAI engine version")
    classification_explanation: Dict[str, Any] = Field(..., description="Local feature attribution for cyclone category")
    intensity_explanation: Dict[str, Any] = Field(..., description="Local feature attribution for 24h intensity forecast")
    track_steering_explanation: Dict[str, Any] = Field(..., description="Kinematic and environmental trajectory steering drivers")
    synoptic_narrative: str = Field(..., description="IMD/RSMC-style natural language diagnostic briefing")
    counterfactual_summary: Dict[str, Any] = Field(..., description="Minimal perturbation required for category shift")
    message: str = Field(default="Comprehensive multi-model XAI explanation generated successfully.")


class XAISaliencyRequest(BaseModel):
    sensor: Optional[str] = Field("INSAT-3D", description="Satellite sensor name")
    channel: Optional[str] = Field("TIR-1", description="Infrared channel name")
    center_row: Optional[int] = Field(None, description="Optional center row index")
    center_col: Optional[int] = Field(None, description="Optional center column index")
    intensity_factor: Optional[float] = Field(0.85, ge=0.1, le=1.5, description="Vortex convective intensity scale")


class XAISaliencyResponse(BaseModel):
    status: str = Field(default="success")
    sensor: str = Field(default="INSAT-3D")
    channel: str = Field(default="TIR-1")
    vortex_center: List[int] = Field(...)
    axisymmetry_score: float = Field(..., ge=0.0, le=1.0)
    eye_contrast_celsius: float = Field(...)
    eyewall_min_temp_celsius: float = Field(...)
    eye_max_temp_celsius: float = Field(...)
    deep_convective_area_km2: float = Field(...)
    dvorak_bd_distribution: List[Dict[str, Any]] = Field(...)
    saliency_grid_shape: List[int] = Field(...)
    saliency_matrix: List[List[float]] = Field(...)
    message: str = Field(default="Dvorak-aligned convective saliency heatmap computed successfully.")


class XAISensitivityRequest(BaseModel):
    parameter: Optional[str] = Field("pressure_deficit_hpa", description="Parameter to vary: 'pressure_deficit_hpa' or 'forward_speed_kmh'")
    lat: Optional[float] = Field(16.0)
    lon: Optional[float] = Field(87.5)
    central_pres: Optional[float] = Field(985.0)
    subbasin: Optional[str] = Field("BB")


class XAISensitivityResponse(BaseModel):
    status: str = Field(default="success")
    parameter: str = Field(...)
    range: List[float] = Field(...)
    sensitivity_curve: List[Dict[str, Any]] = Field(...)
    physical_monotonicity_verified: bool = Field(default=True)
    message: str = Field(default="Physical sensitivity curve computed successfully.")
