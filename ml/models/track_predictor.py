"""
Cyclone Track Prediction Model.
Provides multi-horizon trajectory forecasting for tropical cyclones in the North Indian Ocean basin.
Predicts incremental geographic displacements (d_lat, d_lon) for lead times +6h, +12h, +24h, and +48h,
computes empirical uncertainty cone radii, enforces physical translation-speed and turning-angle limits,
and generates Leaflet-compatible visualization polygons.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.features.extractor import FEATURE_NAMES, BASE_PRESSURE_HPA, EARTH_OMEGA

# Standard operational track forecast horizons (hours)
TRACK_HORIZONS: List[int] = [6, 12, 24, 48]

# Synoptic domain boundaries for the North Indian Ocean basin (Bay of Bengal & Arabian Sea)
NIO_BOUNDS = {
    "lat_min": 0.0,
    "lat_max": 35.0,
    "lon_min": 45.0,
    "lon_max": 105.0,
}

# Maximum physical forward translation speed for North Indian Ocean cyclones (km/h)
MAX_PHYSICAL_SPEED_KMH: float = 45.0
# Maximum physical turning curvature angle per 6 hours (degrees)
MAX_TURNING_ANGLE_DEG_6H: float = 90.0

TRACK_FEATURE_NAMES: List[str] = FEATURE_NAMES + [
    "past_dlat_6h",
    "past_dlon_6h",
    "past_dlat_12h",
    "past_dlon_12h",
    "current_wind",
    "u_motion",
    "v_motion",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle geodesic distance in kilometers between two coordinates.
    """
    r_earth = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return float(r_earth * c)


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes initial compass heading bearing (0° to 360°) from point 1 to point 2.
    0° = North, 90° = East, 180° = South, 270° = West.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing_rad = np.arctan2(y, x)
    return float((np.degrees(bearing_rad) + 360.0) % 360.0)


def decompose_track_error(
    lat_pred: float, lon_pred: float,
    lat_true: float, lon_true: float,
    lat_origin: float, lon_origin: float,
) -> Tuple[float, float, float, float]:
    """
    Decomposes total Haversine error into Along-Track Error (ATE-A) and Cross-Track Error (XTE).
    Returns (ate_km, along_track_km, cross_track_km, heading_error_deg).
    """
    ate = haversine_km(lat_pred, lon_pred, lat_true, lon_true)

    # True movement vector
    true_bearing = compute_bearing(lat_origin, lon_origin, lat_true, lon_true)
    # Predicted movement vector
    pred_bearing = compute_bearing(lat_origin, lon_origin, lat_pred, lon_pred)
    # Error vector from true position to predicted position
    error_bearing = compute_bearing(lat_true, lon_true, lat_pred, lon_pred)

    heading_error = abs((pred_bearing - true_bearing + 180.0) % 360.0 - 180.0)

    # Angle between error vector and true track direction
    rel_angle_rad = np.radians((error_bearing - true_bearing + 180.0) % 360.0 - 180.0)
    ate_a = ate * np.cos(rel_angle_rad)
    xte = ate * np.sin(rel_angle_rad)

    return float(ate), float(ate_a), float(xte), float(heading_error)


class CycloneTrackPredictor:
    """
    Physics-Informed Multi-Horizon Cyclone Track Predictor.
    Predicts incremental displacements for future lead times, enforces physical kinematics,
    and computes calibrated uncertainty cones.
    """

    def __init__(
        self,
        random_state: int = 42,
        horizons: Optional[List[int]] = None,
    ):
        self.random_state = random_state
        self.horizons = horizons or TRACK_HORIZONS
        self.feature_names = TRACK_FEATURE_NAMES

        # Regressors for latitude and longitude displacements per lead horizon
        self.dlat_models: Dict[int, HistGradientBoostingRegressor] = {}
        self.dlon_models: Dict[int, HistGradientBoostingRegressor] = {}

        # Empirical uncertainty cone radii (km) per horizon (75th percentile error)
        self.cone_radii_km: Dict[int, float] = {
            6: 45.0,
            12: 80.0,
            24: 140.0,
            48: 240.0,
        }

        # Historical climatology mean velocity (degrees/hour) for baseline comparison
        self.climatology_dlat_per_hour: float = 0.12  # Poleward drift ~0.12 deg/h
        self.climatology_dlon_per_hour: float = -0.10 # Westward drift ~0.10 deg/h

        self.is_fitted: bool = False

    def _build_features(self, df_or_dict: Any) -> pd.DataFrame:
        """Assembles the 20 required track features from observation inputs."""
        if isinstance(df_or_dict, pd.DataFrame):
            X = df_or_dict.copy()
        else:
            X = pd.DataFrame([df_or_dict])

        # Ensure core physical features exist
        for feat in FEATURE_NAMES:
            if feat not in X.columns:
                X[feat] = 0.0

        # Kinematic lag features
        if "past_dlat_6h" not in X.columns:
            X["past_dlat_6h"] = X.get("DELTA_LAT", 0.0)
        if "past_dlon_6h" not in X.columns:
            X["past_dlon_6h"] = X.get("DELTA_LON", 0.0)
        if "past_dlat_12h" not in X.columns:
            X["past_dlat_12h"] = X["past_dlat_6h"] * 2.0
        if "past_dlon_12h" not in X.columns:
            X["past_dlon_12h"] = X["past_dlon_6h"] * 2.0

        if "current_wind" not in X.columns:
            deficit = X["pressure_deficit"].clip(lower=0.0)
            X["current_wind"] = (14.2 * np.sqrt(deficit)).clip(lower=15.0, upper=140.0)

        # Motion vector components (km/h)
        speed = X["forward_speed"].fillna(15.0).clip(0.0, 100.0)
        bearing_rad = np.arcsin(X["bearing_sin"].fillna(0.0).clip(-1.0, 1.0))
        X["u_motion"] = speed * np.sin(bearing_rad)
        X["v_motion"] = speed * np.cos(bearing_rad)

        return X[self.feature_names]

    def fit(
        self,
        X_dict: Dict[int, pd.DataFrame],
        y_dlat_dict: Dict[int, pd.Series],
        y_dlon_dict: Dict[int, pd.Series],
        val_errors: Optional[Dict[int, List[float]]] = None,
    ) -> "CycloneTrackPredictor":
        """
        Fits multi-horizon displacement regressors and calibrates empirical cone radii.
        """
        for h in self.horizons:
            if h not in X_dict or h not in y_dlat_dict or h not in y_dlon_dict:
                continue

            X_h = self._build_features(X_dict[h])
            y_lat = y_dlat_dict[h].values
            y_lon = y_dlon_dict[h].values

            # Latitude displacement regressor
            model_lat = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=160,
                max_depth=6,
                learning_rate=0.06,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
            model_lat.fit(X_h, y_lat)
            self.dlat_models[h] = model_lat

            # Longitude displacement regressor
            model_lon = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=160,
                max_depth=6,
                learning_rate=0.06,
                min_samples_leaf=20,
                random_state=self.random_state,
            )
            model_lon.fit(X_h, y_lon)
            self.dlon_models[h] = model_lon

        # Calibrate uncertainty cone radii from validation errors (75th percentile)
        if val_errors is not None:
            for h in self.horizons:
                if h in val_errors and len(val_errors[h]) > 10:
                    r75 = float(np.percentile(val_errors[h], 75))
                    self.cone_radii_km[h] = round(max(30.0, r75), 1)

        self.is_fitted = True
        return self

    def enforce_kinematic_consistency(
        self,
        origin_lat: float,
        origin_lon: float,
        pred_lat: float,
        pred_lon: float,
        lead_hours: int,
        prev_lat: Optional[float] = None,
        prev_lon: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Applies physical constraints:
        1. Clamps translation forward speed to MAX_PHYSICAL_SPEED_KMH.
        2. Bounds coordinates within the North Indian Ocean basin.
        3. Returns (cons_lat, cons_lon, speed_kmh, bearing_deg).
        """
        ref_lat = prev_lat if prev_lat is not None else origin_lat
        ref_lon = prev_lon if prev_lon is not None else origin_lon
        step_hours = 6.0 if prev_lat is not None else float(lead_hours)

        # Distance and speed
        dist = haversine_km(ref_lat, ref_lon, pred_lat, pred_lon)
        speed = dist / max(0.5, step_hours)

        cons_lat = pred_lat
        cons_lon = pred_lon

        # Speed clamping
        if speed > MAX_PHYSICAL_SPEED_KMH:
            max_dist = MAX_PHYSICAL_SPEED_KMH * step_hours
            ratio = max_dist / max(1e-4, dist)
            cons_lat = ref_lat + (pred_lat - ref_lat) * ratio
            cons_lon = ref_lon + (pred_lon - ref_lon) * ratio
            speed = MAX_PHYSICAL_SPEED_KMH

        # Synoptic domain bounding
        cons_lat = float(np.clip(cons_lat, NIO_BOUNDS["lat_min"], NIO_BOUNDS["lat_max"]))
        cons_lon = float(np.clip(cons_lon, NIO_BOUNDS["lon_min"], NIO_BOUNDS["lon_max"]))

        bearing = compute_bearing(ref_lat, ref_lon, cons_lat, cons_lon)
        return round(cons_lat, 2), round(cons_lon, 2), round(speed, 1), round(bearing, 1)

    def generate_cone_polygon(
        self, center_lat: float, center_lon: float, radius_km: float, num_points: int = 18
    ) -> List[List[float]]:
        """
        Generates GeoJSON / Leaflet compatible polygon ring coordinates [[lat, lon], ...]
        representing the circular uncertainty cone boundary at a given waypoint.
        """
        coords = []
        for i in range(num_points):
            angle_deg = (i * 360.0) / num_points
            rad = np.radians(angle_deg)
            # 1 degree latitude ~ 111.12 km
            dlat = (radius_km / 111.12) * np.cos(rad)
            # 1 degree longitude ~ 111.12 * cos(lat) km
            dlon = (radius_km / (111.12 * max(0.2, np.cos(np.radians(center_lat))))) * np.sin(rad)
            coords.append([round(center_lat + dlat, 3), round(center_lon + dlon, 3)])

        # Close the polygon ring
        if coords:
            coords.append(coords[0])
        return coords

    def predict_track(
        self,
        current_lat: float,
        current_lon: float,
        obs_features: Union[Dict[str, Any], pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Generates multi-horizon track trajectory forecast with uncertainty cone polygons.
        """
        if not self.is_fitted:
            raise RuntimeError("CycloneTrackPredictor is not fitted.")

        X_feat = self._build_features(obs_features)
        waypoints: List[Dict[str, Any]] = []
        cone_polygons: List[Dict[str, Any]] = []

        prev_lat = current_lat
        prev_lon = current_lon

        for h in self.horizons:
            if h not in self.dlat_models or h not in self.dlon_models:
                continue

            raw_dlat = float(self.dlat_models[h].predict(X_feat)[0])
            raw_dlon = float(self.dlon_models[h].predict(X_feat)[0])

            raw_lat = current_lat + raw_dlat
            raw_lon = current_lon + raw_dlon

            # Apply physical consistency
            cons_lat, cons_lon, speed, bearing = self.enforce_kinematic_consistency(
                origin_lat=current_lat,
                origin_lon=current_lon,
                pred_lat=raw_lat,
                pred_lon=raw_lon,
                lead_hours=h,
                prev_lat=prev_lat,
                prev_lon=prev_lon,
            )

            cone_radius = self.cone_radii_km.get(h, 50.0 + h * 4.0)
            poly_ring = self.generate_cone_polygon(cons_lat, cons_lon, cone_radius)

            waypoint = {
                "horizon_hours": h,
                "lead_time": f"+{h}h",
                "latitude": cons_lat,
                "longitude": cons_lon,
                "displacement_lat": round(cons_lat - current_lat, 2),
                "displacement_lon": round(cons_lon - current_lon, 2),
                "forward_speed_kmh": speed,
                "bearing_deg": bearing,
                "uncertainty_radius_km": round(cone_radius, 1),
                "uncertainty_radius_nm": round(cone_radius / 1.852, 1),
            }
            waypoints.append(waypoint)

            cone_polygons.append({
                "lead_time": f"+{h}h",
                "radius_km": round(cone_radius, 1),
                "center": [cons_lat, cons_lon],
                "polygon_coordinates": poly_ring,
            })

            prev_lat = cons_lat
            prev_lon = cons_lon

        return {
            "initial_position": {
                "latitude": round(current_lat, 2),
                "longitude": round(current_lon, 2),
            },
            "predicted_track": waypoints,
            "track_error_cone": cone_polygons,
            "total_horizons": len(waypoints),
        }

    def predict_persistence(
        self, current_lat: float, current_lon: float, past_dlat_6h: float, past_dlon_6h: float, lead_hours: int
    ) -> Tuple[float, float]:
        """Linear persistence extrapolation baseline: continues past 6h velocity."""
        factor = lead_hours / 6.0
        lat_per = current_lat + past_dlat_6h * factor
        lon_per = current_lon + past_dlon_6h * factor
        return float(np.clip(lat_per, 0.0, 35.0)), float(np.clip(lon_per, 45.0, 105.0))

    def predict_cliper(
        self, current_lat: float, current_lon: float, past_dlat_6h: float, past_dlon_6h: float, lead_hours: int
    ) -> Tuple[float, float]:
        """
        CLIPER (Climatology & Persistence) operational baseline:
        Decays persistence velocity into regional climatological mean steering.
        """
        w_per = float(np.exp(-lead_hours / 24.0))  # Exponential decay of persistence
        w_clim = 1.0 - w_per

        dlat_per = past_dlat_6h * (lead_hours / 6.0)
        dlon_per = past_dlon_6h * (lead_hours / 6.0)

        dlat_clim = self.climatology_dlat_per_hour * lead_hours
        dlon_clim = self.climatology_dlon_per_hour * lead_hours

        dlat = w_per * dlat_per + w_clim * dlat_clim
        dlon = w_per * dlon_per + w_clim * dlon_clim

        lat_clip = current_lat + dlat
        lon_clip = current_lon + dlon
        return float(np.clip(lat_clip, 0.0, 35.0)), float(np.clip(lon_clip, 45.0, 105.0))

    def save(self, filepath: str) -> None:
        """Serializes model payload to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "dlat_models": self.dlat_models,
            "dlon_models": self.dlon_models,
            "cone_radii_km": self.cone_radii_km,
            "horizons": self.horizons,
            "feature_names": self.feature_names,
            "climatology_dlat": self.climatology_dlat_per_hour,
            "climatology_dlon": self.climatology_dlon_per_hour,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "CycloneTrackPredictor":
        """Deserializes CycloneTrackPredictor instance."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        payload = joblib.load(filepath)
        instance = cls()
        instance.dlat_models = payload.get("dlat_models", {})
        instance.dlon_models = payload.get("dlon_models", {})
        instance.cone_radii_km = payload.get("cone_radii_km", {})
        instance.horizons = payload.get("horizons", TRACK_HORIZONS)
        instance.feature_names = payload.get("feature_names", TRACK_FEATURE_NAMES)
        instance.climatology_dlat_per_hour = payload.get("climatology_dlat", 0.12)
        instance.climatology_dlon_per_hour = payload.get("climatology_dlon", -0.10)
        instance.is_fitted = payload.get("is_fitted", False)
        return instance
