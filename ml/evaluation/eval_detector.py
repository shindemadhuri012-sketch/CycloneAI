"""
Scientific Evaluation Pipeline for Phase 4: Cyclone Identification Model.
Evaluates the frozen CycloneDetector model strictly on the completely unseen test split
(70 storms, 2,223 points) with zero data leakage.
"""
import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.extractor import CycloneFeatureExtractor
from ml.models.detector import CycloneDetector


def run_evaluation():
    print("=" * 70)
    print("CycloneAI — Independent Scientific Evaluation on Unseen Test Split")
    print("=" * 70)

    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_detector.joblib")
    test_path = os.path.join(PROJECT_ROOT, "data/processed/test_manifest.csv")
    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test manifest not found at: {test_path}")

    print(f"\n[1/5] Loading trained checkpoint & test split...")
    detector = CycloneDetector.load(model_path)
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

    print(f"\n[2/5] Extracting test features...")
    extractor = CycloneFeatureExtractor()
    X_test, y_test_cs, y_test_dep = extractor.extract_from_dataframe(test_df)

    pos_count = int(y_test_cs.sum())
    neg_count = int((y_test_cs == 0).sum())
    print(f"  Test Positive (Cyclonic Storm >= 34 kt): {pos_count:,d} ({pos_count/len(y_test_cs)*100:.1f}%)")
    print(f"  Test Negative (Sub-cyclonic / Low):      {neg_count:,d} ({neg_count/len(y_test_cs)*100:.1f}%)")

    print(f"\n[3/5] Computing quantitative metrics...")
    test_metrics = detector.evaluate(X_test, y_test_cs, split_name="Unseen Test Split")

    print("\n" + "=" * 50)
    print("      UNSEEN TEST BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  ROC-AUC Score:      {test_metrics['roc_auc']:.4f}")
    print(f"  Brier Score:        {test_metrics['brier_score']:.4f}")
    print("\n  -- Standard Decision Threshold (p >= 0.50) --")
    std = test_metrics["standard_threshold"]
    print(f"  Accuracy:           {std['accuracy']*100:.2f}%")
    print(f"  Precision:          {std['precision']:.4f}")
    print(f"  Recall:             {std['recall']:.4f}")
    print(f"  F1-Score:           {std['f1_score']:.4f}")
    print(f"  Confusion Matrix:   [[TN={std['confusion_matrix'][0][0]}, FP={std['confusion_matrix'][0][1]}], [FN={std['confusion_matrix'][1][0]}, TP={std['confusion_matrix'][1][1]}]]")

    print("\n  -- Operational Disaster Response Threshold (p >= 0.35) --")
    ops = test_metrics["operational_threshold"]
    print(f"  Accuracy:           {ops['accuracy']*100:.2f}%")
    print(f"  Precision:          {ops['precision']:.4f}")
    print(f"  Recall (Safety):    {ops['recall']:.4f}  (High sensitivity to avoid missing cyclones)")
    print(f"  F1-Score:           {ops['f1_score']:.4f}")
    print(f"  Confusion Matrix:   [[TN={ops['confusion_matrix'][0][0]}, FP={ops['confusion_matrix'][0][1]}], [FN={ops['confusion_matrix'][1][0]}, TP={ops['confusion_matrix'][1][1]}]]")

    print(f"\n[4/5] Computing Storm-Level Lifecycle Identification...")
    # Add predictions back to test_df to assess storm-level detection
    probs = detector.predict_proba(X_test)[:, 1]
    test_df_eval = test_df.copy()
    test_df_eval["PRED_PROB"] = probs
    test_df_eval["IS_CS_TRUE"] = y_test_cs
    test_df_eval["PRED_CS_STD"] = (probs >= 0.50).astype(int)
    test_df_eval["PRED_CS_OPS"] = (probs >= 0.35).astype(int)

    storm_summary = []
    for sid, group in test_df_eval.groupby("SID"):
        has_actual_cs = bool(group["IS_CS_TRUE"].max() == 1)
        detected_std = bool(group["PRED_CS_STD"].max() == 1)
        detected_ops = bool(group["PRED_CS_OPS"].max() == 1)
        max_prob = float(group["PRED_PROB"].max())
        name = str(group["NAME"].iloc[0]) if "NAME" in group else "UNNAMED"
        season = int(group["SEASON"].iloc[0])
        storm_summary.append({
            "sid": sid,
            "name": name,
            "season": season,
            "has_actual_cs": has_actual_cs,
            "detected_std": detected_std,
            "detected_ops": detected_ops,
            "max_prob": max_prob,
            "point_count": len(group),
        })

    storm_df = pd.DataFrame(storm_summary)
    actual_cs_storms = storm_df[storm_df["has_actual_cs"]]
    actual_non_cs_storms = storm_df[~storm_df["has_actual_cs"]]

    cs_detected_std_count = int(actual_cs_storms["detected_std"].sum())
    cs_detected_ops_count = int(actual_cs_storms["detected_ops"].sum())
    total_cs_storms = len(actual_cs_storms)

    print(f"  Total Test Storms with CS Stage:     {total_cs_storms}")
    print(f"  Storms Correctly Identified (std):   {cs_detected_std_count} / {total_cs_storms} ({cs_detected_std_count/total_cs_storms*100:.1f}%)")
    print(f"  Storms Correctly Identified (ops):   {cs_detected_ops_count} / {total_cs_storms} ({cs_detected_ops_count/total_cs_storms*100:.1f}%)")

    # Sub-basin performance
    bob_mask = test_df_eval["SUBBASIN"] == "BB"
    as_mask = test_df_eval["SUBBASIN"] == "AS"
    bob_auc = float(roc_auc_score(y_test_cs[bob_mask], probs[bob_mask])) if y_test_cs[bob_mask].nunique() > 1 else 0.0
    as_auc = float(roc_auc_score(y_test_cs[as_mask], probs[as_mask])) if y_test_cs[as_mask].nunique() > 1 else 0.0
    print(f"\n  Sub-basin ROC-AUC:")
    print(f"    Bay of Bengal (BB):  {bob_auc:.4f} ({bob_mask.sum():,d} obs)")
    print(f"    Arabian Sea (AS):    {as_auc:.4f} ({as_mask.sum():,d} obs)")

    print(f"\n[5/5] Exporting evaluation report to JSON...")
    report_dir = os.path.join(PROJECT_ROOT, "docs/models/evaluation_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "detector_test_report.json")

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "CycloneDetector v1.0.0",
        "split": "Unseen Test Split (70 storms)",
        "sample_count": len(test_df),
        "unique_storms": len(test_sids),
        "leakage_count": len(leakage),
        "test_metrics": test_metrics,
        "subbasin_auc": {
            "bay_of_bengal": round(bob_auc, 4),
            "arabian_sea": round(as_auc, 4),
        },
        "storm_level_metrics": {
            "total_cyclonic_storms": total_cs_storms,
            "detected_standard": cs_detected_std_count,
            "detected_operational": cs_detected_ops_count,
            "storm_recall_standard": round(cs_detected_std_count / total_cs_storms, 4),
            "storm_recall_operational": round(cs_detected_ops_count / total_cs_storms, 4),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"  Report written to: {report_path}")
    print("=" * 70)
    print("Scientific Evaluation Successfully Finished!")
    print("=" * 70)
    return report_data


if __name__ == "__main__":
    run_evaluation()
