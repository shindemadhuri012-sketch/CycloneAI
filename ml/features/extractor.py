"""
Feature extraction pipeline for tropical cyclone identification.
Transforms raw atmospheric, kinematic, and spatial observations into
physically grounded feature matrices for machine learning models.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = [
    "lat",
    "lon",
    "forward_speed",
    "dist_km",
    "dt_hours",
    "bearing_sin",
    "bearing_cos",
    "coriolis_f",
    "pressure_deficit",
    "central_pres",
    "month_sin",
    "month_cos",
    "day_of_year",
    "is_bob",
]

# Physical Earth angular rotation rate (rad/s)
EARTH_OMEGA = 7.2921e-5
# Environmental baseline sea-level pressure in the North Indian Ocean tropics (hPa)
BASE_PRESSURE_HPA = 1010.0


class CycloneFeatureExtractor:
    """
    Extracts physically grounded features for tropical cyclone identification.
    Handles DataFrame batches (training/evaluation) as well as single observation dicts (real-time API).
    """

    def __init__(self):
        self.feature_names = FEATURE_NAMES

    def compute_coriolis(self, lat: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Computes planetary vorticity Coriolis parameter:
        f = 2 * Omega * sin(latitude)
        Scaled by 1e4 for numerical stability.
        """
        rad_lat = np.radians(lat)
        return 2.0 * EARTH_OMEGA * np.sin(rad_lat) * 1e4

    def extract_from_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Extracts features and targets from a preprocessed manifest DataFrame.

        Returns:
            X: DataFrame with 14 engineered features.
            y_cs: Binary Series (1 if Cyclonic Storm >= 34 kt / CS+, 0 otherwise).
            y_dep: Binary Series (1 if Depression >= 17 kt / D+, 0 otherwise).
        """
        features = pd.DataFrame(index=df.index)

        # Spatial coordinates
        features["lat"] = df["LAT"].astype(float)
        features["lon"] = df["LON"].astype(float)

        # Kinematics
        features["forward_speed"] = df["FORWARD_SPEED_KMH"].fillna(0.0).clip(lower=0.0, upper=150.0).astype(float)
        features["dist_km"] = df["DIST_KM"].fillna(0.0).clip(lower=0.0, upper=1000.0).astype(float)
        features["dt_hours"] = df["DT_HOURS"].fillna(3.0).clip(lower=0.5, upper=24.0).astype(float)

        # Bearing cyclical decomposition
        bearing = df["BEARING_DEG"].fillna(0.0)
        rad_bearing = np.radians(bearing)
        features["bearing_sin"] = np.sin(rad_bearing).astype(float)
        features["bearing_cos"] = np.cos(rad_bearing).astype(float)

        # Planetary vorticity
        features["coriolis_f"] = self.compute_coriolis(features["lat"]).astype(float)

        # Thermodynamics
        features["pressure_deficit"] = df["PRESSURE_DEFICIT"].fillna(0.0).clip(lower=0.0, upper=150.0).astype(float)
        features["central_pres"] = df["CENTRAL_PRES_HPA"].fillna(BASE_PRESSURE_HPA).clip(lower=850.0, upper=1020.0).astype(float)

        # Temporal / Seasonal cyclical features
        iso_time = pd.to_datetime(df["ISO_TIME"])
        month = iso_time.dt.month.fillna(10)
        features["month_sin"] = np.sin(2.0 * np.pi * month / 12.0).astype(float)
        features["month_cos"] = np.cos(2.0 * np.pi * month / 12.0).astype(float)
        features["day_of_year"] = (iso_time.dt.dayofyear.fillna(280) / 365.25).astype(float)

        # Ocean sub-basin (Bay of Bengal = 1, Arabian Sea / other = 0)
        features["is_bob"] = (df["SUBBASIN"].fillna("BB") == "BB").astype(float)

        # Target 1: Cyclonic Storm (>= 34 kt or IMD grade CS, SCS, VSCS, ESCS, SuCS)
        cs_grades = {"CS", "SCS", "VSCS", "ESCS", "SuCS", "SUCS", "SCS(H)"}
        is_cs = (df["IMD_GRADE"].isin(cs_grades)) | (df["EFFECTIVE_WIND_KT"].fillna(0.0) >= 34.0)
        y_cs = is_cs.astype(int)

        # Target 2: Depression or above (>= 17 kt, all except LOW / L)
        is_dep = (~df["IMD_GRADE"].isin(["LOW", "L"])) & (df["IMD_GRADE"].notna())
        y_dep = is_dep.astype(int)

        return features[FEATURE_NAMES], y_cs, y_dep

    def extract_from_dict(self, obs: Dict[str, Any]) -> pd.DataFrame:
        """
        Extracts features from a single observation dictionary for real-time inference.
        """
        lat = float(obs.get("lat", 15.0))
        lon = float(obs.get("lon", 88.0))
        speed = float(obs.get("forward_speed", 15.0))
        dist = float(obs.get("dist_km", 45.0))
        dt = float(obs.get("dt_hours", 3.0))
        bearing = float(obs.get("bearing", 315.0))
        central_pres = float(obs.get("central_pres", 1000.0))
        deficit = float(obs.get("pressure_deficit", max(0.0, BASE_PRESSURE_HPA - central_pres)))
        month = int(obs.get("month", 10))
        day_of_year = float(obs.get("day_of_year", 290.0)) / 365.25
        subbasin = str(obs.get("subbasin", "BB")).upper()
        is_bob = 1.0 if ("BB" in subbasin or "BENGAL" in subbasin) else 0.0

        rad_bearing = np.radians(bearing)
        coriolis = float(self.compute_coriolis(lat))

        data = {
            "lat": [lat],
            "lon": [lon],
            "forward_speed": [min(150.0, max(0.0, speed))],
            "dist_km": [min(1000.0, max(0.0, dist))],
            "dt_hours": [min(24.0, max(0.5, dt))],
            "bearing_sin": [np.sin(rad_bearing)],
            "bearing_cos": [np.cos(rad_bearing)],
            "coriolis_f": [coriolis],
            "pressure_deficit": [min(150.0, max(0.0, deficit))],
            "central_pres": [min(1020.0, max(850.0, central_pres))],
            "month_sin": [np.sin(2.0 * np.pi * month / 12.0)],
            "month_cos": [np.cos(2.0 * np.pi * month / 12.0)],
            "day_of_year": [day_of_year],
            "is_bob": [is_bob],
        }
        return pd.DataFrame(data)[FEATURE_NAMES]
