"""
Service layer for accessing real North Indian Ocean storm archives.
Reads directly from verified tracks_processed.csv (Phase 3 verified dataset).
Zero synthetic or fake cyclone records.
"""
import os
import sys
import logging
from typing import List, Optional, Dict, Any
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.schemas.storms import (
    StormCatalogItem,
    StormObservationPoint,
    StormCatalogResponse,
    StormDetailsResponse,
)

logger = logging.getLogger(__name__)


class StormCatalogService:
    def __init__(self):
        self.tracks_csv_path = os.path.join(PROJECT_ROOT, "data/processed/tracks_processed.csv")
        self.df: Optional[pd.DataFrame] = None
        self.catalog_cache: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Loads and indexes the verified North Indian Ocean dataset."""
        if not os.path.exists(self.tracks_csv_path):
            logger.error(f"tracks_processed.csv not found at {self.tracks_csv_path}")
            return

        try:
            df = pd.read_csv(self.tracks_csv_path, low_memory=False)
            df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
            self.df = df

            # Build aggregated storm catalog for fast retrieval
            catalog = []
            grouped = df.groupby("SID")
            for sid, group in grouped:
                names = group["NAME"].dropna().unique()
                valid_names = [n for n in names if n not in ["UNNAMED", "NOT_NAMED"]]
                name = valid_names[0] if len(valid_names) > 0 else "UNNAMED"
                
                season = int(group["SEASON"].iloc[0]) if "SEASON" in group.columns else 2000
                subbasin = str(group["SUBBASIN"].iloc[0]) if "SUBBASIN" in group.columns else "BB"
                
                # Peak wind and min pressure
                peak_wind = float(group["EFFECTIVE_WIND_KT"].max()) if "EFFECTIVE_WIND_KT" in group.columns else 30.0
                min_pres = float(group["CENTRAL_PRES_HPA"].min()) if "CENTRAL_PRES_HPA" in group.columns else 1000.0
                
                grades = group["IMD_GRADE"].dropna().unique().tolist()
                peak_grade = grades[-1] if len(grades) > 0 else "LOW"
                
                start_date = str(group["ISO_TIME"].min().date()) if not group["ISO_TIME"].isna().all() else "N/A"
                end_date = str(group["ISO_TIME"].max().date()) if not group["ISO_TIME"].isna().all() else "N/A"
                point_count = len(group)

                catalog.append({
                    "sid": str(sid),
                    "name": name,
                    "season": season,
                    "subbasin": subbasin,
                    "max_grade": peak_grade,
                    "peak_wind_kt": round(peak_wind, 1),
                    "min_pressure_hpa": round(min_pres, 1),
                    "start_date": start_date,
                    "end_date": end_date,
                    "point_count": point_count
                })

            # Sort catalog by season descending, peak wind descending
            catalog.sort(key=lambda x: (x["season"], x["peak_wind_kt"]), reverse=True)
            self.catalog_cache = catalog
            logger.info(f"StormCatalogService loaded {len(self.catalog_cache)} unique storms ({len(df)} points).")
        except Exception as e:
            logger.error(f"Failed to load storm catalog: {e}")

    def get_catalog(
        self,
        year: Optional[int] = None,
        basin: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50
    ) -> StormCatalogResponse:
        """Filters catalog by year, basin, and name query."""
        results = self.catalog_cache

        if year is not None:
            results = [s for s in results if s["season"] == year]

        if basin is not None and basin != "All":
            results = [s for s in results if s["subbasin"].upper() == basin.upper()]

        if search:
            query = search.strip().upper()
            results = [s for s in results if query in s["name"].upper() or query in s["sid"].upper()]

        total = len(results)
        sliced = results[:limit]

        items = [StormCatalogItem(**s) for s in sliced]
        return StormCatalogResponse(status="success", total_storms=total, storms=items)

    def get_storm_points(self, sid: str) -> Optional[StormDetailsResponse]:
        """Returns all chronological observation points for a given storm."""
        if self.df is None:
            return None

        storm_df = self.df[self.df["SID"] == sid].sort_values("ISO_TIME")
        if storm_df.empty:
            return None

        meta = next((s for s in self.catalog_cache if s["sid"] == sid), None)
        name = meta["name"] if meta else "STORM"
        season = meta["season"] if meta else int(storm_df["SEASON"].iloc[0])
        subbasin = meta["subbasin"] if meta else str(storm_df["SUBBASIN"].iloc[0])
        peak_grade = meta["max_grade"] if meta else "LOW"
        peak_wind = meta["peak_wind_kt"] if meta else float(storm_df["EFFECTIVE_WIND_KT"].max())
        min_pres = meta["min_pressure_hpa"] if meta else float(storm_df["CENTRAL_PRES_HPA"].min())

        points = []
        for _, row in storm_df.iterrows():
            points.append(StormObservationPoint(
                timestamp=str(row["ISO_TIME"]),
                latitude=round(float(row["LAT"]), 3),
                longitude=round(float(row["LON"]), 3),
                central_pres=round(float(row.get("CENTRAL_PRES_HPA", 1000.0)), 1),
                wind_kt=round(float(row.get("EFFECTIVE_WIND_KT", 30.0)), 1),
                forward_speed=round(float(row.get("FORWARD_SPEED_KMH", 15.0)), 1),
                bearing=round(float(row.get("BEARING_DEG", 315.0)), 1),
                pressure_deficit=round(float(row.get("PRESSURE_DEFICIT", 10.0)), 1),
                imd_grade=str(row.get("IMD_GRADE", "D"))
            ))

        return StormDetailsResponse(
            sid=sid,
            name=name,
            season=season,
            subbasin=subbasin,
            peak_grade=peak_grade,
            peak_wind_kt=peak_wind,
            min_pressure_hpa=min_pres,
            points=points
        )

    def get_preset_storms(self) -> List[Dict[str, Any]]:
        """Returns famous benchmark storms for 1-click presets in the frontend."""
        presets = []
        target_names = ["FANI", "AMPHAN", "BIPARJOY", "MOCHA", "REMAL", "TAUKTAE", "HUDHUD", "DANA"]
        for target in target_names:
            match = next((s for s in self.catalog_cache if s["name"].upper() == target), None)
            if match and self.df is not None:
                # Get a peak observation point for this storm
                pts = self.df[self.df["SID"] == match["sid"]].sort_values("EFFECTIVE_WIND_KT", ascending=False)
                if not pts.empty:
                    peak_pt = pts.iloc[0]
                    presets.append({
                        "sid": match["sid"],
                        "name": match["name"],
                        "season": match["season"],
                        "subbasin": match["subbasin"],
                        "peak_grade": match["max_grade"],
                        "peak_wind_kt": match["peak_wind_kt"],
                        "observation": {
                            "lat": round(float(peak_pt["LAT"]), 2),
                            "lon": round(float(peak_pt["LON"]), 2),
                            "central_pres": round(float(peak_pt["CENTRAL_PRES_HPA"]), 1),
                            "current_wind_speed_knots": round(float(peak_pt["EFFECTIVE_WIND_KT"]), 1),
                            "forward_speed": round(float(peak_pt["FORWARD_SPEED_KMH"]), 1),
                            "bearing": round(float(peak_pt["BEARING_DEG"]), 1),
                            "month": int(peak_pt["ISO_TIME"].month) if pd.notnull(peak_pt["ISO_TIME"]) else 5,
                            "subbasin": str(peak_pt["SUBBASIN"]) if pd.notnull(peak_pt["SUBBASIN"]) else "BB"
                        }
                    })
        return presets
