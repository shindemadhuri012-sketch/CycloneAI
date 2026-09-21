"""
Geospatial Coordinate Transformations & Kinematic Feature Extraction
Computes translation speeds, heading bearings, displacement vectors, and bounding normalizations.
"""
from typing import Tuple
import numpy as np
import pandas as pd

# Synoptic domain boundaries for the North Indian Ocean basin (Bay of Bengal & Arabian Sea)
NIO_BOUNDS = {
    "lat_min": 0.0,
    "lat_max": 35.0,
    "lon_min": 45.0,
    "lon_max": 105.0
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle geodesic distance in kilometers between two points on Earth.
    """
    r_earth = 6371.0  # Earth's mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r_earth * c)

def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the initial compass heading bearing (0° to 360°) from point 1 to point 2.
    0° = North, 90° = East, 180° = South, 270° = West.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return float(bearing_deg)

def compute_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sequential kinematic features (displacement, forward speed, bearing) per storm (SID).
    """
    df = df.copy()
    df.sort_values(by=["SID", "ISO_TIME"], inplace=True)

    # Initialize feature columns
    df["DELTA_LAT"] = 0.0
    df["DELTA_LON"] = 0.0
    df["DIST_KM"] = 0.0
    df["DT_HOURS"] = 0.0
    df["FORWARD_SPEED_KMH"] = 0.0
    df["BEARING_DEG"] = 0.0

    processed_rows = []
    for sid, group in df.groupby("SID", sort=False):
        group = group.copy()
        lats = group["LAT"].values
        lons = group["LON"].values
        times = pd.to_datetime(group["ISO_TIME"]).values

        n = len(group)
        if n > 1:
            delta_lats = np.zeros(n, dtype=np.float32)
            delta_lons = np.zeros(n, dtype=np.float32)
            dist_km = np.zeros(n, dtype=np.float32)
            dt_hours = np.zeros(n, dtype=np.float32)
            speed_kmh = np.zeros(n, dtype=np.float32)
            bearings = np.zeros(n, dtype=np.float32)

            for i in range(1, n):
                delta_lats[i] = lats[i] - lats[i - 1]
                delta_lons[i] = lons[i] - lons[i - 1]

                # Time step in hours
                dt = (times[i] - times[i - 1]) / np.timedelta64(1, "h")
                dt_hours[i] = max(dt, 0.1)  # avoid div by zero

                # Distance and speed
                d = haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
                dist_km[i] = d
                speed_kmh[i] = min(d / dt_hours[i], 120.0)  # Clamped to physical storm translation limits

                # Direction
                bearings[i] = compute_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])

            group["DELTA_LAT"] = delta_lats
            group["DELTA_LON"] = delta_lons
            group["DIST_KM"] = dist_km
            group["DT_HOURS"] = dt_hours
            group["FORWARD_SPEED_KMH"] = speed_kmh
            group["BEARING_DEG"] = bearings

        processed_rows.append(group)

    return pd.concat(processed_rows, ignore_index=True)

def normalize_coordinates(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Min-max normalizes coordinates to [0.0, 1.0] within the North Indian Ocean basin bounds.
    """
    norm_lat = (lats - NIO_BOUNDS["lat_min"]) / (NIO_BOUNDS["lat_max"] - NIO_BOUNDS["lat_min"])
    norm_lon = (lons - NIO_BOUNDS["lon_min"]) / (NIO_BOUNDS["lon_max"] - NIO_BOUNDS["lon_min"])
    return np.clip(norm_lat, 0.0, 1.0), np.clip(norm_lon, 0.0, 1.0)

def denormalize_coordinates(norm_lat: np.ndarray, norm_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverts min-max coordinate normalization back to true decimal degrees.
    """
    lats = norm_lat * (NIO_BOUNDS["lat_max"] - NIO_BOUNDS["lat_min"]) + NIO_BOUNDS["lat_min"]
    lons = norm_lon * (NIO_BOUNDS["lon_max"] - NIO_BOUNDS["lon_min"]) + NIO_BOUNDS["lon_min"]
    return lats, lons
