"""
Multi-Modal Temporal & Spatial Synchronization
Bridges high-dimensional satellite imagery tensors with low-dimensional time-series track vectors.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

def align_satellite_with_tracks(
    satellite_metadata: pd.DataFrame,
    tracks_df: pd.DataFrame,
    temporal_tolerance_minutes: int = 45,
    spatial_tolerance_degrees: float = 0.50
) -> pd.DataFrame:
    """
    Synchronizes satellite imagery observation frames with verified best-track ground truth.
    
    Args:
        satellite_metadata: DataFrame containing columns ['image_id', 'sid', 'timestamp', 'center_lat', 'center_lon']
        tracks_df: Processed best-track DataFrame with ['SID', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'NEWDELHI_GRADE', ...]
        temporal_tolerance_minutes: Maximum allowable time difference between satellite scan and track point.
        spatial_tolerance_degrees: Maximum allowable distance between image center and tracked vortex eye.
        
    Returns:
        Synchronized DataFrame containing merged image identifiers and ground-truth metrics.
    """
    sat = satellite_metadata.copy()
    trk = tracks_df.copy()

    sat["timestamp"] = pd.to_datetime(sat["timestamp"])
    trk["ISO_TIME"] = pd.to_datetime(trk["ISO_TIME"])

    aligned_records = []

    # Group by Storm Identifier (SID)
    trk_grouped = {sid: group for sid, group in trk.groupby("SID")}

    for _, sat_row in sat.iterrows():
        sid = sat_row.get("sid")
        if sid not in trk_grouped:
            continue

        storm_tracks = trk_grouped[sid]
        scan_time = sat_row["timestamp"]

        # Calculate time delta in minutes
        time_deltas = np.abs((storm_tracks["ISO_TIME"] - scan_time).dt.total_seconds()) / 60.0
        nearest_idx = time_deltas.idxmin()
        min_dt = time_deltas.loc[nearest_idx]

        if min_dt <= temporal_tolerance_minutes:
            track_row = storm_tracks.loc[nearest_idx]

            # Verify spatial distance if image center is available
            spatial_match = True
            if "center_lat" in sat_row and "center_lon" in sat_row:
                d_lat = abs(sat_row["center_lat"] - track_row["LAT"])
                d_lon = abs(sat_row["center_lon"] - track_row["LON"])
                if d_lat > spatial_tolerance_degrees or d_lon > spatial_tolerance_degrees:
                    spatial_match = False

            if spatial_match:
                merged = {
                    "image_id": sat_row.get("image_id"),
                    "sid": sid,
                    "satellite_time": scan_time.isoformat(),
                    "track_time": track_row["ISO_TIME"].isoformat(),
                    "dt_minutes": round(float(min_dt), 1),
                    "lat": float(track_row["LAT"]),
                    "lon": float(track_row["LON"]),
                    "wind_knots": float(track_row["WMO_WIND"]) if pd.notna(track_row.get("WMO_WIND")) else np.nan,
                    "imd_wind": float(track_row["NEWDELHI_WIND"]) if pd.notna(track_row.get("NEWDELHI_WIND")) else np.nan,
                    "imd_grade": track_row.get("NEWDELHI_GRADE", "UNKNOWN")
                }
                aligned_records.append(merged)

    return pd.DataFrame(aligned_records)
