"""
Scientific Evaluation Pipeline for Phase 5: Cyclone Pattern & Category Classification.
Evaluates the frozen CycloneClassifier model strictly on the completely unseen test split
(70 storms, 2,223 points) with zero data leakage.
"""
import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.extractor import CycloneFeatureExtractor
from ml.models.classifier import (
    CycloneClassifier,
    IMD_CATEGORIES,
    IMD_GRADE_TO_INDEX,
    CATEGORY_NAMES,
)


def run_evaluation():
    print("=" * 75)
    print("CycloneAI — Phase 5 Independent Scientific Evaluation on Unseen Test Split")
    print("=" * 75)

    model_path = os.path.join(PROJECT_ROOT, "ml/models/saved/cyclone_classifier.joblib")
    test_path = os.path.join(PROJECT_ROOT, "data/processed/test_manifest.csv")
    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test manifest missing: {test_path}")

    print(f"\n[1/5] Loading trained checkpoint & test split...")
    classifier = CycloneClassifier.load(model_path)
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

    print(f"\n[2/5] Extracting test features & mapping 8 IMD categories...")
    extractor = CycloneFeatureExtractor()
    X_test, _, _ = extractor.extract_from_dataframe(test_df)
    y_test = test_df["IMD_GRADE"].map(IMD_GRADE_TO_INDEX).fillna(0).astype(int)

    print("\n  Class Distribution in Unseen Test Split:")
    test_dist = y_test.value_counts().sort_index()
    for idx, count in test_dist.items():
        cat = IMD_CATEGORIES[int(idx)]
        pct = (count / len(y_test)) * 100
        print(f"    [{cat['code']:<5}] {cat['name']:<32}: {count:>4,d} ({pct:>5.1f}%) | Wind: {cat['wind_kt']:>7} kt")

    print(f"\n[3/5] Computing multi-class metrics on unseen test data...")
    test_metrics = classifier.evaluate(X_test, y_test, split_name="Unseen Test Split")

    print("\n" + "=" * 60)
    print("       UNSEEN TEST BENCHMARK RESULTS (70 STORMS)")
    print("=" * 60)
    print(f"  Top-1 Exact Match Accuracy:           {test_metrics['top1_exact_accuracy']*100:.2f}%")
    print(f"  Adjacent Category Match (|diff| <= 1): {test_metrics['adjacent_accuracy']*100:.2f}%")
    print(f"  Top-2 Categorical Accuracy:           {test_metrics['top2_accuracy']*100:.2f}%")
    print(f"  Mean Absolute Category Error (MACE):  {test_metrics['mean_absolute_category_error']:.3f} steps")
    print(f"  Macro-Averaged F1-Score:              {test_metrics['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted-Averaged F1-Score:           {test_metrics['weighted_avg']['f1_score']:.4f}")

    print("\n  Per-Class Performance on Unseen Test Split:")
    print("  " + "-" * 72)
    print(f"  {'Code':<5} {'Category Name':<32} {'Prec':<7} {'Recall':<7} {'F1':<7} {'Support':<7}")
    print("  " + "-" * 72)
    for row in test_metrics["per_class"]:
        print(f"  {row['code']:<5} {row['name']:<32} {row['precision']:<7.3f} {row['recall']:<7.3f} {row['f1_score']:<7.3f} {row['support']:<7d}")
    print("  " + "-" * 72)

    print("\n[4/5] Computing Sub-basin Performance Breakdown...")
    probs = classifier.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    bob_mask = test_df["SUBBASIN"] == "BB"
    as_mask = test_df["SUBBASIN"] == "AS"

    bob_acc = float(np.mean(preds[bob_mask] == y_test[bob_mask]))
    bob_adj = float(np.mean(np.abs(preds[bob_mask] - y_test[bob_mask]) <= 1))
    bob_mace = float(np.mean(np.abs(preds[bob_mask] - y_test[bob_mask])))

    as_acc = float(np.mean(preds[as_mask] == y_test[as_mask]))
    as_adj = float(np.mean(np.abs(preds[as_mask] - y_test[as_mask]) <= 1))
    as_mace = float(np.mean(np.abs(preds[as_mask] - y_test[as_mask])))

    print(f"  Bay of Bengal (BB - {bob_mask.sum():,d} obs): Top-1 Acc={bob_acc*100:.2f}%, Adjacent Acc={bob_adj*100:.2f}%, MACE={bob_mace:.3f}")
    print(f"  Arabian Sea   (AS - {as_mask.sum():,d} obs): Top-1 Acc={as_acc*100:.2f}%, Adjacent Acc={as_adj*100:.2f}%, MACE={as_mace:.3f}")

    print(f"\n[5/5] Exporting evaluation report to JSON...")
    report_dir = os.path.join(PROJECT_ROOT, "docs/models/evaluation_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "classifier_test_report.json")

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "CycloneClassifier v1.0.0",
        "split": "Unseen Test Split (70 storms)",
        "sample_count": len(test_df),
        "unique_storms": len(test_sids),
        "leakage_count": len(leakage),
        "test_metrics": test_metrics,
        "subbasin_breakdown": {
            "bay_of_bengal": {
                "sample_count": int(bob_mask.sum()),
                "top1_accuracy": round(bob_acc, 4),
                "adjacent_accuracy": round(bob_adj, 4),
                "mace": round(bob_mace, 4),
            },
            "arabian_sea": {
                "sample_count": int(as_mask.sum()),
                "top1_accuracy": round(as_acc, 4),
                "adjacent_accuracy": round(as_adj, 4),
                "mace": round(as_mace, 4),
            },
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"  Report written to: {report_path}")
    print("=" * 75)
    print("Phase 5 Scientific Evaluation Finished Successfully!")
    print("=" * 75)
    return report_data


if __name__ == "__main__":
    run_evaluation()
