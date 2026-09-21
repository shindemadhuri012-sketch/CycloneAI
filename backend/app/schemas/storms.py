"""
Schemas for verified historical storm catalog and observation points.
Powers the Historical Cyclones explorer and 1-click presets.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class StormCatalogItem(BaseModel):
    sid: str = Field(..., description="Unique IBTrACS Storm Identifier")
    name: str = Field(..., description="Official storm name (e.g. FANI, AMPHAN, BIPARJOY)")
    season: int = Field(..., description="Cyclone season year")
    subbasin: str = Field(..., description="Subbasin identifier ('BB' for Bay of Bengal, 'AS' for Arabian Sea)")
    max_grade: str = Field(..., description="Peak IMD intensity grade reached")
    peak_wind_kt: float = Field(..., description="Peak Maximum Sustained Wind Speed in knots")
    min_pressure_hpa: float = Field(..., description="Minimum central barometric pressure reached in hPa")
    start_date: str = Field(..., description="Genesis date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Dissipation date (YYYY-MM-DD)")
    point_count: int = Field(..., description="Number of tracked observation points in IBTrACS")


class StormObservationPoint(BaseModel):
    timestamp: str = Field(..., description="Observation timestamp (ISO 8601)")
    latitude: float = Field(..., description="Storm eye latitude in decimal degrees")
    longitude: float = Field(..., description="Storm eye longitude in decimal degrees")
    central_pres: float = Field(..., description="Central pressure in hPa")
    wind_kt: float = Field(..., description="Maximum sustained wind speed in knots")
    forward_speed: float = Field(..., description="Translation speed in km/h")
    bearing: float = Field(..., description="Compass movement bearing in degrees")
    pressure_deficit: float = Field(..., description="Computed pressure deficit (Pn - Pc) in hPa")
    imd_grade: str = Field(..., description="Recorded IMD category grade")


class StormDetailsResponse(BaseModel):
    sid: str = Field(..., description="Storm ID")
    name: str = Field(..., description="Storm Name")
    season: int = Field(..., description="Season Year")
    subbasin: str = Field(..., description="Subbasin")
    peak_grade: str = Field(..., description="Peak Grade")
    peak_wind_kt: float = Field(..., description="Peak Wind (kt)")
    min_pressure_hpa: float = Field(..., description="Min Pressure (hPa)")
    points: List[StormObservationPoint] = Field(default=[], description="Chronological track observations")


class StormCatalogResponse(BaseModel):
    status: str = Field(default="success")
    total_storms: int = Field(..., description="Total storms matching query")
    storms: List[StormCatalogItem] = Field(default=[], description="List of catalog storms")
