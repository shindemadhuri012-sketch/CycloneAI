"""
Master Training Script for Phase 6: Cyclone Intensity Prediction Model.
Constructs multi-horizon intra-storm training samples, fits gradient boosted regressors
with Huber loss, estimates quantile uncertainty bounds, trains the Rapid Intensification (RI)
classifier, and exports model checkpoints.
"""
import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.extractor import CycloneFeatureExtractor, BASE_PRESSURE_HPA
from ml.models.intensity_predictor import (
    CycloneIntensityPredictor,
    FORECAST_HORIZONS,
    RI_THRESHOLD_KT,
    IMD_COEFF,
)


def build_intensity_dataset(manifest_df: pd.DataFrame):
    """
    Constructs multi-horizon training targets strictly intra-storm.
    Matches observations at t + 6h, t + 12h, t + 24h within the same storm SID.
    """
    df = manifest_df.copy()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # 1. Base Feature Extraction
    extractor = CycloneFeatureExtractor()
    base_features, _, _ = extractor.extract_from_dataframe(df)

    # 2. Impute current wind where missing using official IMD pressure-wind formula
    deficit = df["PRESSURE_DEFICIT"].fillna(0.0).clip(lower=0.0)
    theo_wind = IMD_COEFF * np.sqrt(deficit)
    current_wind = df["EFFECTIVE_WIND_KT"].fillna(theo_wind).clip(lower=15.0, upper=140.0)

    base_features["current_wind"] = current_wind.values
    base_features["central_pres"] = df["CENTRAL_PRES_HPA"].fillna(BASE_PRESSURE_HPA).clip(lower=850.0, upper=1020.0).values
    base_features["pressure_deficit"] = deficit.values

    # Index by (SID, ISO_TIME) for fast O(1) intra-storm temporal lookups
    df_indexed = df.copy()
    df_indexed["_calc_wind"] = current_wind
    df_indexed["_calc_pres"] = base_features["central_pres"]
    df_indexed["_idx"] = df_indexed.index
    lookup_map = df_indexed.set_index(["SID", "ISO_TIME"])

    # 3. Dynamic Past 6-Hour Trends (dV/dt, dP/dt)
    past_times = df["ISO_TIME"] - pd.Timedelta(hours=6)
    past_keys = list(zip(df["SID"], past_times))
    past_dv = []
    past_dp = []

    for i, key in enumerate(past_keys):
        if key in lookup_map.index:
            row = lookup_map.loc[key]
            # In case of duplicate timestamps in raw data, take the first row
            prev_w = float(row["_calc_wind"].iloc[0] if isinstance(row, pd.DataFrame) else row["_calc_wind"])
            prev_p = float(row["_calc_pres"].iloc[0] if isinstance(row, pd.DataFrame) else row["_calc_pres"])
            past_dv.append(current_wind.iloc[i] - prev_w)
            past_dp.append(base_features["central_pres"].iloc[i] - prev_p)
        else:
            past_dv.append(0.0)
            past_dp.append(0.0)

    base_features["past_dv_6h"] = np.array(past_dv, dtype=np.float32)
    base_features["past_dp_6h"] = np.array(past_dp, dtype=np.float32)

    theo_w = np.maximum(10.0, IMD_COEFF * np.sqrt(np.maximum(0.1, BASE_PRESSURE_HPA - base_features["central_pres"])))
    base_features["intensity_balance_ratio"] = (base_features["current_wind"] / theo_w).clip(0.4, 2.5)

    # 4. Multi-Horizon Targets (+0h, +6h, +12h, +24h)
    X_dict: Dict[int, pd.DataFrame] = {}
    y_wind_dict: Dict[int, pd.Series] = {}
    y_pres_dict: Dict[int, pd.Series] = {}
    pers_wind_dict: Dict[int, pd.Series] = {}
    pers_pres_dict: Dict[int, pd.Series] = {}

    for h in FORECAST_HORIZONS:
        if h == 0:
            X_dict[0] = base_features.copy()
            y_wind_dict[0] = current_wind.copy()
            y_pres_dict[0] = base_features["central_pres"].copy()
            pers_wind_dict[0] = current_wind.copy()
            pers_pres_dict[0] = base_features["central_pres"].copy()
            continue

        target_times = df["ISO_TIME"] + pd.Timedelta(hours=h)
        target_keys = list(zip(df["SID"], target_times))

        valid_indices = []
        target_winds = []
        target_pressures = []

        for i, key in enumerate(target_keys):
            if key in lookup_map.index:
                row = lookup_map.loc[key]
                fut_w = float(row["_calc_wind"].iloc[0] if isinstance(row, pd.DataFrame) else row["_calc_wind"])
                fut_p = float(row["_calc_pres"].iloc[0] if isinstance(row, pd.DataFrame) else row["_calc_pres"])
                valid_indices.append(i)
                target_winds.append(fut_w)
                target_pressures.append(fut_p)

        X_dict[h] = base_features.iloc[valid_indices].reset_index(drop=True)
        y_wind_dict[h] = pd.Series(target_winds, dtype=np.float32)
        y_pres_dict[h] = pd.Series(target_pressures, dtype=np.float32)
        pers_wind_dict[h] = current_wind.iloc[valid_indices].reset_index(drop=True)
        pers_pres_dict[h] = base_features["central_pres"].iloc[valid_indices].reset_index(drop=True)

    # 5. Rapid Intensification (RI) Target (delta V_24h >= 30 kt)
    delta_w_24 = y_wind_dict[24] - pers_wind_dict[24]
    y_ri = (delta_w_24 >= RI_THRESHOLD_KT).astype(int)
    X_ri = X_dict[24].copy()

    return X_dict, y_wind_dict, y_pres_dict, pers_wind_dict, pers_pres_dict, X_ri, y_ri


def run_training():
    print("=" * 75)
    print("CycloneAI — Phase 6: Cyclone Intensity Prediction Model Training")
    print("=" * 75)

    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")
    val_path = os.path.join(PROJECT_ROOT, "data/processed/val_manifest.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Processed manifests not found.")

    print(f"\n[1/5] Ingesting manifests and constructing intra-storm temporal horizons...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_tr_d, y_w_tr, y_p_tr, _, _, X_ri_tr, y_ri_tr = build_intensity_dataset(train_df)
    X_val_d, y_w_val, y_p_val, p_w_val, p_p_val, X_ri_val, y_ri_val = build_intensity_dataset(val_df)

    print(f"  Training Storms:   {train_df['SID'].nunique()} storms ({len(train_df):,d} total track points)")
    for h in [6, 12, 24]:
        print(f"    Horizon +{h:2d}h samples: {len(y_w_tr[h]):,d} train | {len(y_w_val[h]):,d} val")
    print(f"  Rapid Intensification (+24h RI >= 30 kt): {y_ri_tr.sum():,d} / {len(y_ri_tr):,d} in train ({y_ri_tr.mean()*100:.1f}%)")

    print(f"\n[2/5] Fitting Multi-Horizon Regressors & Quantile Uncertainty Estimators...")
    predictor = CycloneIntensityPredictor(random_state=42)
    predictor.fit(X_tr_d, y_w_tr, y_p_tr, y_ri=y_ri_tr, X_ri=X_ri_tr)

    print(f"\n[3/5] Evaluating Multi-Horizon Forecasts against Persistence Baselines...")
    val_metrics = {}

    print("\n" + "=" * 70)
    print("   VALIDATION BENCHMARK RESULTS ACROSS FORECAST HORIZONS")
    print("=" * 70)

    for h in [6, 12, 24]:
        X_h = predictor._build_features(X_val_d[h])
        pred_w = predictor.wind_models[h].predict(X_h)
        pred_p = predictor.pres_models[h].predict(X_h)

        true_w = y_w_val[h].values
        true_p = y_p_val[h].values
        pers_w = p_w_val[h].values
        pers_p = p_p_val[h].values

        # Model metrics
        w_mae = float(mean_absolute_error(true_w, pred_w))
        w_rmse = float(root_mean_squared_error(true_w, pred_w))
        w_r2 = float(r2_score(true_w, pred_w))

        p_mae = float(mean_absolute_error(true_p, pred_p))
        p_rmse = float(root_mean_squared_error(true_p, pred_p))
        p_r2 = float(r2_score(true_p, pred_p))

        # Persistence baseline metrics
        w_mae_pers = float(mean_absolute_error(true_w, pers_w))
        p_mae_pers = float(mean_absolute_error(true_p, pers_p))

        # Quantile coverage check (alpha=0.10, 0.90)
        q10_w = predictor.wind_q10[h].predict(X_h)
        q90_w = predictor.wind_q90[h].predict(X_h)
        w_coverage = float(np.mean((true_w >= q10_w) & (true_w <= q90_w)))

        w_skill = ((w_mae_pers - w_mae) / w_mae_pers) * 100.0

        val_metrics[f"+{h}h"] = {
            "wind_mae": round(w_mae, 2),
            "wind_rmse": round(w_rmse, 2),
            "wind_r2": round(w_r2, 3),
            "persistence_wind_mae": round(w_mae_pers, 2),
            "wind_skill_gain_pct": round(w_skill, 1),
            "wind_90pct_interval_coverage": round(w_coverage * 100, 1),
            "pressure_mae": round(p_mae, 2),
            "pressure_rmse": round(p_rmse, 2),
            "pressure_r2": round(p_r2, 3),
            "persistence_pressure_mae": round(p_mae_pers, 2),
        }

        print(f"\n--- Lead Horizon: +{h} Hours ({len(true_w):,d} validation points) ---")
        print(f"  Wind Speed (knots):    MAE = {w_mae:.2f} kt | RMSE = {w_rmse:.2f} kt | R2 = {w_r2:.3f}")
        print(f"    Persistence Baseline: MAE = {w_mae_pers:.2f} kt -> Model Skill Gain: +{w_skill:.1f}%")
        print(f"    90% Uncertainty Bound Coverage: {w_coverage*100:.1f}% of true observations")
        print(f"  Central Pressure (hPa): MAE = {p_mae:.2f} hPa | RMSE = {p_rmse:.2f} hPa | R2 = {p_r2:.3f}")

    # RI Classifier Validation
    X_ri_v = predictor._build_features(X_ri_val)
    ri_probs = predictor.ri_model.predict_proba(X_ri_v)[:, 1]
    ri_preds = (ri_probs >= 0.040).astype(int)

    ri_auc = float(roc_auc_score(y_ri_val, ri_probs))
    from sklearn.metrics import precision_score, recall_score, f1_score
    ri_prec = float(precision_score(y_ri_val, ri_preds, zero_division=0))
    ri_rec = float(recall_score(y_ri_val, ri_preds, zero_division=0))
    ri_f1 = float(f1_score(y_ri_val, ri_preds, zero_division=0))

    print("\n" + "=" * 70)
    print("   RAPID INTENSIFICATION (RI) DETECTION ON VALIDATION SPLIT")
    print("=" * 70)
    print(f"  Actual RI Events (dV_24h >= 30 kt): {y_ri_val.sum()} / {len(y_ri_val)}")
    print(f"  ROC-AUC Score:      {ri_auc:.4f}")
    print(f"  Sensitivity (Recall): {ri_rec*100:.1f}% (Operational threshold p >= 0.040)")
    print(f"  Precision:          {ri_prec*100:.1f}%")
    print(f"  F1-Score:           {ri_f1:.4f}")

    val_metrics["rapid_intensification"] = {
        "roc_auc": round(ri_auc, 4),
        "recall": round(ri_rec, 4),
        "precision": round(ri_prec, 4),
        "f1_score": round(ri_f1, 4),
        "actual_ri_events": int(y_ri_val.sum()),
    }

    print(f"\n[4/5] Serializing model and metadata to disk...")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "ml/models/saved")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "cyclone_intensity.joblib")
    meta_path = os.path.join(checkpoint_dir, "intensity_metadata.json")

    predictor.save(model_path)

    metadata = {
        "model_name": "CycloneIntensityPredictor",
        "version": "1.0.0",
        "phase": "Phase 6 - Cyclone Intensity Prediction",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizons": FORECAST_HORIZONS,
        "ri_threshold_kt": RI_THRESHOLD_KT,
        "validation_metrics": val_metrics,
        "features": predictor.feature_names,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model saved to:    {model_path} ({os.path.getsize(model_path):,d} bytes)")
    print(f"  Metadata saved to: {meta_path}")
    print("=" * 75)
    print("Phase 6 Intensity Model Training Successfully Completed!")
    print("=" * 75)
    return predictor, val_metrics


if __name__ == "__main__":
    run_training()
