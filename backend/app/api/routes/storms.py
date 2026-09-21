"""
Routes for historical storm archive and real-world presets.
Directly accesses verified North Indian Ocean dataset. Zero synthetic data.
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from app.schemas.storms import StormCatalogResponse, StormDetailsResponse
from app.services.storm_service import StormCatalogService
from app.api.dependencies import get_storm_service

router = APIRouter(prefix="/storms", tags=["Historical Storm Catalog"])


@router.get("/catalog", response_model=StormCatalogResponse, summary="Query Historical Cyclone Catalog")
async def get_cyclone_catalog(
    year: Optional[int] = Query(None, description="Filter by season year (e.g. 2019, 2020)"),
    basin: Optional[str] = Query(None, description="Filter by subbasin ('BB' or 'AS')"),
    search: Optional[str] = Query(None, description="Search storm by name or SID"),
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    service: StormCatalogService = Depends(get_storm_service)
) -> StormCatalogResponse:
    """Returns filtered real-world cyclone catalog from verified IBTrACS NIO records."""
    return service.get_catalog(year=year, basin=basin, search=search, limit=limit)


@router.get("/presets", summary="Get Famous Benchmark Storm Presets")
async def get_storm_presets(
    service: StormCatalogService = Depends(get_storm_service)
) -> List[Dict[str, Any]]:
    """Returns famous real-world benchmark storms (Fani, Amphan, Biparjoy, Mocha, etc.) for 1-click presets."""
    return service.get_preset_storms()


@router.get("/{sid}/points", response_model=StormDetailsResponse, summary="Get Storm Trajectory Points")
async def get_storm_details(
    sid: str,
    service: StormCatalogService = Depends(get_storm_service)
) -> StormDetailsResponse:
    """Returns all chronological observation points for a specific storm."""
    res = service.get_storm_points(sid)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Storm with SID '{sid}' not found in archive.")
    return res
