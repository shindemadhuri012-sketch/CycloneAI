"""
Health check schema models.
"""
from typing import Optional, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="operational", description="Operational status of backend")
    service: str = Field(default="CycloneAI Backend", description="Service identifier")
    model_status: str = Field(default="active", description="Status of ML model subsystem")
    active_models: Optional[Dict[str, bool]] = Field(default=None, description="Connection state of individual models")
