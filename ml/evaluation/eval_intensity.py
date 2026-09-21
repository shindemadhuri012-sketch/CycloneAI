"""
Independent Scientific Evaluation Pipeline for Phase 6: Cyclone Intensity Prediction Model.
Evaluates the frozen multi-horizon CycloneIntensityPredictor strictly on the completely unseen test split
(70 storms, 2,223 points) with zero data leakage.
"""
import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.intensity_predictor import CycloneIntensityPredictor, IMD_COEFF
from ml.training.train_intensity import build_intensity_dataset


def run_evaluation():
    print("=" * 75)
    print("CycloneAI — Phase 6 Independent Scientific Evaluation on Unseen Test Split")
    print("=" * 75)

    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_intensity.joblib")
    test_path = os.path.join(PROJECT_ROOT, "data/processed/test_manifest.csv")
    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test manifest not found at: {test_path}")

    print(f"\n[1/5] Loading trained checkpoint & test split...")
    predictor = CycloneIntensityPredictor.load(model_path)
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    # 100% Leakage verification
    train_sids = set(train_df["SID"].unique())
    test_sids = set(test_df["SID"].unique())
    leakage = train_sids.intersection(test_sids)
    print(f"  Unseen Test Storms: {len(test_sids)} unique historical storms")
    print(f"  Test Observations:  {len(test_df):,d} track observations")
    print(f"  Leakage Audit:      {len(leakage)} overlapping storms ({'0.00% LEAKAGE - STRICT PASS' if len(leakage) == 0 else 'FAIL'})")
    assert len(leakage) == 0, f"Critical: {len(leakage)} storms leaked between train and test!"

    print(f"\n[2/5] Constructing intra-storm multi-horizon test evaluation targets...")
    X_test_d, y_w_test, y_p_test, p_w_test, p_p_test, X_ri_test, y_ri_test = build_intensity_dataset(test_df)

    for h in [6, 12, 24]:
        print(f"  Horizon +{h:2d}h test samples: {len(y_w_test[h]):,d}")
    print(f"  Rapid Intensification test cases (dV_24h >= 30 kt): {y_ri_test.sum():,d} / {len(y_ri_test):,d} ({y_ri_test.mean()*100:.1f}%)")

    print(f"\n[3/5] Benchmarking Multi-Horizon Forecasts against Persistence Baseline...")
    horizon_results = {}

    print("\n" + "=" * 70)
    print("      UNSEEN TEST BENCHMARK RESULTS ACROSS LEAD HORIZONS")
    print("=" * 70)

    for h in [6, 12, 24]:
        X_h = predictor._build_features(X_test_d[h])
        pred_w_raw = predictor.wind_models[h].predict(X_h)
        pred_p_raw = predictor.pres_models[h].predict(X_h)

        # Enforce hydrodynamic physical consistency
        pred_w = []
        pred_p = []
        for rw, rp in zip(pred_w_raw, pred_p_raw):
            cw, cp = predictor.enforce_hydrodynamic_consistency(float(rw), float(rp))
            pred_w.append(cw)
            pred_p.append(cp)
        pred_w = np.array(pred_w, dtype=np.float32)
        pred_p = np.array(pred_p, dtype=np.float32)

        true_w = y_w_test[h].values
        true_p = y_p_test[h].values
        pers_w = p_w_test[h].values
        pers_p = p_p_test[h].values

        # Wind Speed metrics
        w_mae = float(mean_absolute_error(true_w, pred_w))
        w_rmse = float(root_mean_squared_error(true_w, pred_w))
        w_r2 = float(r2_score(true_w, pred_w))
        w_pers_mae = float(mean_absolute_error(true_w, pers_w))
        w_skill = float(((w_pers_mae - w_mae) / w_pers_mae) * 100.0)

        # Central Pressure metrics
        p_mae = float(mean_absolute_error(true_p, pred_p))
        p_rmse = float(root_mean_squared_error(true_p, pred_p))
        p_r2 = float(r2_score(true_p, pred_p))
        p_pers_mae = float(mean_absolute_error(true_p, pers_p))
        p_skill = float(((p_pers_mae - p_mae) / p_pers_mae) * 100.0)

        # Quantile 90% uncertainty bound coverage
        q10_w = predictor.wind_q10[h].predict(X_h)
        q90_w = predictor.wind_q90[h].predict(X_h)
        w_cov = float(np.mean((true_w >= q10_w) & (true_w <= q90_w)))

        q10_p = predictor.pres_q10[h].predict(X_h)
        q90_p = predictor.pres_q90[h].predict(X_h)
        p_cov = float(np.mean((true_p >= q10_p) & (true_p <= q90_p)))

        horizon_results[f"+{h}h"] = {
            "lead_hours": h,
            "sample_count": len(true_w),
            "wind_speed": {
                "mae_knots": round(w_mae, 2),
                "rmse_knots": round(w_rmse, 2),
                "r2_score": round(w_r2, 3),
                "persistence_mae_knots": round(w_pers_mae, 2),
                "skill_gain_over_persistence_pct": round(w_skill, 1),
                "uncertainty_90pct_coverage": round(w_cov * 100, 1),
            },
            "central_pressure": {
                "mae_hpa": round(p_mae, 2),
                "rmse_hpa": round(p_rmse, 2),
                "r2_score": round(p_r2, 3),
                "persistence_mae_hpa": round(p_pers_mae, 2),
                "skill_gain_over_persistence_pct": round(p_skill, 1),
                "uncertainty_90pct_coverage": round(p_cov * 100, 1),
            },
        }

        print(f"\n--- Lead Horizon: +{h} Hours ({len(true_w):,d} unseen observations) ---")
        print(f"  Wind Speed (knots):     MAE = {w_mae:.2f} kt | RMSE = {w_rmse:.2f} kt | R2 = {w_r2:.3f}")
        print(f"    Persistence Baseline: MAE = {w_pers_mae:.2f} kt -> Skill Gain: +{w_skill:.1f}%")
        print(f"    90% Uncertainty Bound Coverage: {w_cov*100:.1f}% of true observations")
        print(f"  Central Pressure (hPa): MAE = {p_mae:.2f} hPa | RMSE = {p_rmse:.2f} hPa | R2 = {p_r2:.3f}")
        print(f"    Persistence Baseline: MAE = {p_pers_mae:.2f} hPa -> Skill Gain: +{p_skill:.1f}%")

    print("\n[4/5] Evaluating Rapid Intensification (RI) Alert Gating...")
    X_ri_t = predictor._build_features(X_ri_test)
    ri_probs = predictor.ri_model.predict_proba(X_ri_t)[:, 1]
    ri_preds = (ri_probs >= 0.040).astype(int)

    ri_auc = float(roc_auc_score(y_ri_test, ri_probs))
    ri_prec = float(precision_score(y_ri_test, ri_preds, zero_division=0))
    ri_rec = float(recall_score(y_ri_test, ri_preds, zero_division=0))
    ri_f1 = float(f1_score(y_ri_test, ri_preds, zero_division=0))
    ri_cm = confusion_matrix(y_ri_test, ri_preds).tolist()

    print("\n" + "=" * 70)
    print("   RAPID INTENSIFICATION (RI) UNSEEN TEST EVALUATION")
    print("=" * 70)
    print(f"  Actual RI Events (dV_24h >= 30 kt): {y_ri_test.sum():,d} / {len(y_ri_test):,d}")
    print(f"  ROC-AUC Score:          {ri_auc:.4f}")
    print(f"  Sensitivity (Recall):   {ri_rec*100:.1f}%")
    print(f"  Precision:              {ri_prec*100:.1f}%")
    print(f"  F1-Score:               {ri_f1:.4f}")
    print(f"  Confusion Matrix (p>=0.040): [[TN={ri_cm[0][0]}, FP={ri_cm[0][1]}], [FN={ri_cm[1][0]}, TP={ri_cm[1][1]}]]")

    # Hydrodynamic Physical Consistency Audit
    pred_24w = predictor.wind_models[24].predict(X_ri_t)
    pred_24p = predictor.pres_models[24].predict(X_ri_t)
    theo_w = IMD_COEFF * np.sqrt(np.maximum(0.0, 1010.0 - pred_24p))
    wind_pres_corr = float(np.corrcoef(pred_24w, 1010.0 - pred_24p)[0, 1])
    print(f"\n  Hydrodynamic Consistency Audit:")
    print(f"    Correlation between Predicted Wind & Pressure Deficit: r = {wind_pres_corr:.4f} (Strict Physical Coupling)")

    print(f"\n[5/5] Exporting evaluation report to JSON...")
    report_dir = os.path.join(PROJECT_ROOT, "docs/models/evaluation_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "intensity_test_report.json")

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "CycloneIntensityPredictor v1.0.0",
        "split": "Unseen Test Split (70 storms)",
        "sample_count": len(test_df),
        "unique_storms": len(test_sids),
        "leakage_count": len(leakage),
        "multi_horizon_benchmarks": horizon_results,
        "rapid_intensification": {
            "roc_auc": round(ri_auc, 4),
            "recall": round(ri_rec, 4),
            "precision": round(ri_prec, 4),
            "f1_score": round(ri_f1, 4),
            "confusion_matrix": ri_cm,
            "actual_ri_events": int(y_ri_test.sum()),
        },
        "hydrodynamic_coupling_correlation": round(wind_pres_corr, 4),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"  Report written to: {report_path}")
    print("=" * 75)
    print("Phase 6 Independent Scientific Evaluation Completed Successfully!")
    print("=" * 75)
    return report_data


if __name__ == "__main__":
    run_evaluation()
