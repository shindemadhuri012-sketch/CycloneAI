"""
Master Training Pipeline for Phase 7: Cyclone Track Prediction Model.
Extracts intra-storm trajectory displacement targets across lead horizons (+6h, +12h, +24h, +48h),
fits multi-horizon coordinate regressors, calibrates empirical uncertainty cone radii on validation data,
and serializes the trained checkpoint to ml/models/saved/cyclone_track.joblib.
"""
import os
import sys
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.extractor import CycloneFeatureExtractor, BASE_PRESSURE_HPA
from ml.models.track_predictor import CycloneTrackPredictor, TRACK_HORIZONS, haversine_km

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_track_dataset(
    df: pd.DataFrame, horizons: List[int] = TRACK_HORIZONS
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.Series], Dict[int, pd.Series]]:
    """
    Extracts multi-horizon features and displacement targets strictly within storm lifespans.
    """
    extractor = CycloneFeatureExtractor()
    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])

    X_dict: Dict[int, List[Dict]] = {h: [] for h in horizons}
    y_lat_dict: Dict[int, List[float]] = {h: [] for h in horizons}
    y_lon_dict: Dict[int, List[float]] = {h: [] for h in horizons}

    for sid, group in df.groupby("SID", sort=False):
        group = group.sort_values("ISO_TIME").reset_index(drop=True)
        times = group["ISO_TIME"].values
        lats = group["LAT"].values
        lons = group["LON"].values
        n = len(group)

        for i in range(n):
            t_i = times[i]
            row_i = group.iloc[i].to_dict()

            # Past 6h and 12h kinematic vectors (look backward)
            past_dlat_6h = float(row_i.get("DELTA_LAT", 0.0))
            past_dlon_6h = float(row_i.get("DELTA_LON", 0.0))
            past_dlat_12h = past_dlat_6h * 2.0
            past_dlon_12h = past_dlon_6h * 2.0

            for k in range(i - 1, -1, -1):
                dt_back = (t_i - times[k]) / np.timedelta64(1, "h")
                if abs(dt_back - 6.0) <= 1.5:
                    past_dlat_6h = float(lats[i] - lats[k])
                    past_dlon_6h = float(lons[i] - lons[k])
                elif abs(dt_back - 12.0) <= 2.0:
                    past_dlat_12h = float(lats[i] - lats[k])
                    past_dlon_12h = float(lons[i] - lons[k])
                elif dt_back > 14.0:
                    break

            # Build base observation dictionary
            obs_dict = {
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "forward_speed": float(row_i.get("FORWARD_SPEED_KMH", 15.0)),
                "dist_km": float(row_i.get("DIST_KM", 45.0)),
                "dt_hours": float(row_i.get("DT_HOURS", 3.0)),
                "bearing": float(row_i.get("BEARING_DEG", 315.0)),
                "central_pres": float(row_i.get("CENTRAL_PRES_HPA", 1000.0)),
                "pressure_deficit": float(row_i.get("PRESSURE_DEFICIT", 10.0)),
                "month": int(pd.to_datetime(row_i.get("ISO_TIME")).month),
                "day_of_year": float(pd.to_datetime(row_i.get("ISO_TIME")).dayofyear),
                "subbasin": str(row_i.get("SUBBASIN", "BB")),
            }

            feat_df = extractor.extract_from_dict(obs_dict)
            feat_dict = feat_df.iloc[0].to_dict()

            current_wind = float(row_i.get("EFFECTIVE_WIND_KT", 0.0))
            if np.isnan(current_wind) or current_wind <= 0:
                deficit = max(0.0, BASE_PRESSURE_HPA - obs_dict["central_pres"])
                current_wind = 14.2 * np.sqrt(deficit)

            feat_dict["past_dlat_6h"] = past_dlat_6h
            feat_dict["past_dlon_6h"] = past_dlon_6h
            feat_dict["past_dlat_12h"] = past_dlat_12h
            feat_dict["past_dlon_12h"] = past_dlon_12h
            feat_dict["current_wind"] = current_wind

            # Forward look for future lead horizons
            for h in horizons:
                for j in range(i + 1, n):
                    dt_fwd = (times[j] - t_i) / np.timedelta64(1, "h")
                    if abs(dt_fwd - h) <= 1.5:
                        dlat = float(lats[j] - lats[i])
                        dlon = float(lons[j] - lons[i])
                        X_dict[h].append(feat_dict)
                        y_lat_dict[h].append(dlat)
                        y_lon_dict[h].append(dlon)
                        break
                    elif dt_fwd > h + 1.5:
                        break

    X_out = {h: pd.DataFrame(X_dict[h]) for h in horizons}
    y_lat_out = {h: pd.Series(y_lat_dict[h]) for h in horizons}
    y_lon_out = {h: pd.Series(y_lon_dict[h]) for h in horizons}
    return X_out, y_lat_out, y_lon_out


def train_track_model():
    """Main training routine for CycloneTrackPredictor."""
    logger.info("Starting Phase 7 Cyclone Track Prediction Model Training Pipeline")

    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")
    val_path = os.path.join(PROJECT_ROOT, "data/processed/val_manifest.csv")

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} points across {train_df['SID'].nunique()} historical storms")

    logger.info("Extracting intra-storm trajectory displacement samples for training...")
    X_train, y_lat_train, y_lon_train = extract_track_dataset(train_df)

    for h in TRACK_HORIZONS:
        logger.info(f"Horizon +{h}h: {len(X_train[h])} training pairs assembled")

    # Load validation data for empirical uncertainty cone calibration
    logger.info(f"Loading validation data from {val_path}")
    val_df = pd.read_csv(val_path)
    X_val, y_lat_val, y_lon_val = extract_track_dataset(val_df)

    # Initialize model
    predictor = CycloneTrackPredictor(random_state=42)

    # Preliminary fit
    logger.info("Fitting multi-horizon displacement regressors (HistGradientBoosting)...")
    predictor.fit(X_train, y_lat_train, y_lon_train)

    # Compute validation track errors to calibrate empirical 75th percentile cone radii
    logger.info("Evaluating on validation split to calibrate uncertainty cone radii (75th percentile)...")
    val_errors: Dict[int, List[float]] = {h: [] for h in TRACK_HORIZONS}

    for h in TRACK_HORIZONS:
        if len(X_val[h]) == 0:
            continue
        X_h = predictor._build_features(X_val[h])
        pred_dlats = predictor.dlat_models[h].predict(X_h)
        pred_dlons = predictor.dlon_models[h].predict(X_h)

        orig_lats = X_val[h]["lat"].values
        orig_lons = X_val[h]["lon"].values
        true_dlats = y_lat_val[h].values
        true_dlons = y_lon_val[h].values

        for k in range(len(X_h)):
            t_lat = orig_lats[k] + true_dlats[k]
            t_lon = orig_lons[k] + true_dlons[k]
            p_lat = orig_lats[k] + pred_dlats[k]
            p_lon = orig_lons[k] + pred_dlons[k]

            err = haversine_km(p_lat, p_lon, t_lat, t_lon)
            val_errors[h].append(err)

        mean_ate = float(np.mean(val_errors[h]))
        r75 = float(np.percentile(val_errors[h], 75))
        logger.info(f"Validation +{h}h: ATE = {mean_ate:.2f} km | 75th percentile radius = {r75:.2f} km")

    # Re-calibrate cone radii in predictor
    for h in TRACK_HORIZONS:
        if len(val_errors[h]) > 0:
            predictor.cone_radii_km[h] = round(float(np.percentile(val_errors[h], 75)), 1)

    # Save trained checkpoint
    save_dir = os.path.join(PROJECT_ROOT, "ml/models/saved")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "cyclone_track.joblib")
    predictor.save(checkpoint_path)
    logger.info(f"Successfully saved CycloneTrackPredictor to {checkpoint_path}")

    # Save training metadata
    metadata_path = os.path.join(save_dir, "track_metadata.json")
    val_ate_dict = {f"+{h}h_ate_km": round(float(np.mean(val_errors[h])), 2) for h in TRACK_HORIZONS}
    metadata = {
        "model_version": "1.0.0",
        "random_state": 42,
        "horizons": TRACK_HORIZONS,
        "training_samples_per_horizon": {f"+{h}h": len(X_train[h]) for h in TRACK_HORIZONS},
        "validation_samples_per_horizon": {f"+{h}h": len(X_val[h]) for h in TRACK_HORIZONS},
        "validation_ate_km": val_ate_dict,
        "calibrated_cone_radii_km": predictor.cone_radii_km,
        "feature_names": predictor.feature_names,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved track training metadata to {metadata_path}")


if __name__ == "__main__":
    train_track_model()
