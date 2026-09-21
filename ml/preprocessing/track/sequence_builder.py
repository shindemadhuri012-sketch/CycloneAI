"""
Sliding Window Sequence Generator for Spatio-Temporal Trajectory & Intensity Forecasting
Constructs (X, y) sequence pairs per storm, strictly preventing cross-storm sequence contamination.
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from .coordinate_transforms import normalize_coordinates

def generate_storm_trajectory_sequences(
    df: pd.DataFrame,
    history_steps: int = 4,   # 4 steps * 6h = 24h past trajectory
    forecast_steps: int = 4   # 4 steps * 6h = 24h future forecast
) -> List[Dict[str, np.ndarray]]:
    """
    Constructs multi-step input and target sequences for neural trajectory and intensity forecasting.
    
    Returns a list of sample dictionaries:
    {
        "sid": str,
        "input_coords": [history_steps, 2] (norm_lat, norm_lon),
        "input_kinematics": [history_steps, 4] (delta_lat, delta_lon, speed_kmh_norm, bearing_norm),
        "input_intensity": [history_steps, 2] (wind_norm, pres_deficit_norm),
        "target_coords": [forecast_steps, 2] (raw lat, lon targets for loss calculation),
        "target_wind": [forecast_steps] (raw wind knots targets)
    }
    """
    samples = []
    df = df.copy().sort_values(["SID", "ISO_TIME"])

    for sid, storm_group in df.groupby("SID", sort=False):
        n_obs = len(storm_group)
        required_len = history_steps + forecast_steps
        if n_obs < required_len:
            continue  # Storm track too short for full window

        lats = storm_group["LAT"].values
        lons = storm_group["LON"].values
        winds = storm_group["WMO_WIND"].values
        winds_imd = storm_group["NEWDELHI_WIND"].values if "NEWDELHI_WIND" in storm_group.columns else winds

        # Prefer IMD wind where available, fallback to WMO
        effective_winds = np.where(np.isnan(winds_imd), winds, winds_imd)

        # Skip storms with missing coordinates
        if np.isnan(lats).any() or np.isnan(lons).any():
            # Interpolate small gaps or filter
            valid_mask = ~np.isnan(lats) & ~np.isnan(lons)
            if valid_mask.sum() < required_len:
                continue
            storm_group = storm_group[valid_mask].copy()
            lats = storm_group["LAT"].values
            lons = storm_group["LON"].values
            effective_winds = effective_winds[valid_mask]
            n_obs = len(storm_group)
            if n_obs < required_len:
                continue

        # Extract normalized coordinates
        norm_lats, norm_lons = normalize_coordinates(lats, lons)

        # Kinematic fields
        d_lats = storm_group["DELTA_LAT"].values if "DELTA_LAT" in storm_group.columns else np.zeros_like(lats)
        d_lons = storm_group["DELTA_LON"].values if "DELTA_LON" in storm_group.columns else np.zeros_like(lons)
        speeds = storm_group["FORWARD_SPEED_KMH"].values if "FORWARD_SPEED_KMH" in storm_group.columns else np.zeros_like(lats)
        bearings = storm_group["BEARING_DEG"].values if "BEARING_DEG" in storm_group.columns else np.zeros_like(lats)

        # Normalization
        norm_speeds = np.clip(speeds / 100.0, 0.0, 1.0)
        norm_bearings = bearings / 360.0
        norm_winds = np.clip(np.nan_to_num(effective_winds, nan=25.0) / 150.0, 0.0, 1.0)
        pres_deficits = storm_group["PRESSURE_DEFICIT"].values if "PRESSURE_DEFICIT" in storm_group.columns else np.zeros_like(lats)
        norm_pres_deficits = np.clip(np.nan_to_num(pres_deficits, nan=10.0) / 100.0, 0.0, 1.0)

        # Slide window across the storm's lifespan
        for i in range(n_obs - required_len + 1):
            # Input slice
            in_slice = slice(i, i + history_steps)
            # Output target slice
            out_slice = slice(i + history_steps, i + required_len)

            input_coords = np.stack([norm_lats[in_slice], norm_lons[in_slice]], axis=-1).astype(np.float32)
            input_kinematics = np.stack([
                d_lats[in_slice],
                d_lons[in_slice],
                norm_speeds[in_slice],
                norm_bearings[in_slice]
            ], axis=-1).astype(np.float32)
            input_intensity = np.stack([
                norm_winds[in_slice],
                norm_pres_deficits[in_slice]
            ], axis=-1).astype(np.float32)

            target_coords = np.stack([lats[out_slice], lons[out_slice]], axis=-1).astype(np.float32)
            target_wind = effective_winds[out_slice].astype(np.float32)

            samples.append({
                "sid": sid,
                "input_coords": input_coords,
                "input_kinematics": input_kinematics,
                "input_intensity": input_intensity,
                "target_coords": target_coords,
                "target_wind": target_wind
            })

    return samples
