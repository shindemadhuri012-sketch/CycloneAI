"""
API dependency injection providers.
Provides singleton instances of all service layer components.
"""
from app.services.satellite_service import satellite_service, SatelliteService
from app.services.classification_service import classification_service, ClassificationService
from app.services.intensity_service import intensity_service, IntensityService
from app.services.track_service import track_service, TrackService
from app.services.xai_service import xai_service, XAIService
from app.services.storm_service import StormCatalogService
from app.services.pipeline_service import PipelineOrchestratorService

# Singletons
storm_service = StormCatalogService()
pipeline_service = PipelineOrchestratorService(
    satellite_service=satellite_service,
    classification_service=classification_service,
    intensity_service=intensity_service,
    track_service=track_service,
    xai_service=xai_service
)

def get_satellite_service() -> SatelliteService:
    return satellite_service

def get_classification_service() -> ClassificationService:
    return classification_service

def get_intensity_service() -> IntensityService:
    return intensity_service

def get_track_service() -> TrackService:
    return track_service

def get_xai_service() -> XAIService:
    return xai_service

def get_storm_service() -> StormCatalogService:
    return storm_service

def get_pipeline_service() -> PipelineOrchestratorService:
    return pipeline_service
