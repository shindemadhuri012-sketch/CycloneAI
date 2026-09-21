"""
Master Training Script for Phase 5: Cyclone Pattern & Category Classification Model.
Trains the CycloneClassifier multi-class ensemble on real preprocessed manifests,
evaluates on validation data across all 8 IMD categories, and exports model checkpoints.
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


def run_training():
    print("=" * 75)
    print("CycloneAI — Phase 5: Cyclone Pattern & Category Classification Model Training")
    print("=" * 75)

    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")
    val_path = os.path.join(PROJECT_ROOT, "data/processed/val_manifest.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Manifests missing:\n  {train_path}\n  {val_path}")

    print(f"\n[1/5] Loading real preprocessed manifests...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"  Training Split:   {len(train_df):,d} observations across {train_df['SID'].nunique()} unique storms")
    print(f"  Validation Split: {len(val_df):,d} observations across {val_df['SID'].nunique()} unique storms")

    print(f"\n[2/5] Extracting features and mapping 8 IMD category targets...")
    extractor = CycloneFeatureExtractor()
    X_train, _, _ = extractor.extract_from_dataframe(train_df)
    X_val, _, _ = extractor.extract_from_dataframe(val_df)

    y_train = train_df["IMD_GRADE"].map(IMD_GRADE_TO_INDEX).fillna(0).astype(int)
    y_val = val_df["IMD_GRADE"].map(IMD_GRADE_TO_INDEX).fillna(0).astype(int)

    print("\n  Class Distribution in Training Split:")
    train_dist = y_train.value_counts().sort_index()
    for idx, count in train_dist.items():
        cat = IMD_CATEGORIES[int(idx)]
        pct = (count / len(y_train)) * 100
        print(f"    [{cat['code']:<5}] {cat['name']:<32}: {count:>5,d} ({pct:>5.1f}%) | Wind: {cat['wind_kt']:>7} kt")

    print(f"\n[3/5] Fitting CycloneClassifier (Balanced RF + Weighted HistGradientBoosting)...")
    classifier = CycloneClassifier(
        n_estimators=160,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    print(f"\n[4/5] Evaluating on Training and Validation splits...")
    train_metrics = classifier.evaluate(X_train, y_train, split_name="Train")
    val_metrics = classifier.evaluate(X_val, y_val, split_name="Validation")

    print("\n" + "=" * 55)
    print("        VALIDATION SET MULTI-CLASS PERFORMANCE")
    print("=" * 55)
    print(f"  Top-1 Exact Match Accuracy:           {val_metrics['top1_exact_accuracy']*100:.2f}%")
    print(f"  Adjacent Category Match (|diff| <= 1): {val_metrics['adjacent_accuracy']*100:.2f}%")
    print(f"  Top-2 Categorical Accuracy:           {val_metrics['top2_accuracy']*100:.2f}%")
    print(f"  Mean Absolute Category Error (MACE):  {val_metrics['mean_absolute_category_error']:.3f} steps")
    print(f"  Macro-Averaged F1-Score:              {val_metrics['macro_avg']['f1_score']:.4f}")
    print(f"  Weighted-Averaged F1-Score:           {val_metrics['weighted_avg']['f1_score']:.4f}")

    print("\n  Per-Class Performance on Validation Split:")
    print("  " + "-" * 72)
    print(f"  {'Code':<5} {'Category Name':<32} {'Prec':<7} {'Recall':<7} {'F1':<7} {'Support':<7}")
    print("  " + "-" * 72)
    for row in val_metrics["per_class"]:
        print(f"  {row['code']:<5} {row['name']:<32} {row['precision']:<7.3f} {row['recall']:<7.3f} {row['f1_score']:<7.3f} {row['support']:<7d}")
    print("  " + "-" * 72)

    print("\n--- Physical Feature Importances ---")
    sorted_feats = sorted(classifier.feature_importances_.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_feats:
        bar = "#" * int(imp * 40)
        print(f"  {feat:<18} {imp:.4f}  |{bar}")

    print(f"\n[5/5] Serializing model and metadata to disk...")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "ml/models/saved")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "cyclone_classifier.joblib")
    meta_path = os.path.join(checkpoint_dir, "classifier_metadata.json")

    classifier.save(model_path)

    metadata = {
        "model_name": "CycloneClassifier",
        "version": "1.0.0",
        "phase": "Phase 5 - Cyclone Classification",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(train_df),
        "training_unique_storms": int(train_df["SID"].nunique()),
        "validation_samples": len(val_df),
        "validation_unique_storms": int(val_df["SID"].nunique()),
        "categories": IMD_CATEGORIES,
        "features": extractor.feature_names,
        "hyperparameters": {
            "n_estimators": classifier.n_estimators,
            "max_depth": classifier.max_depth,
            "min_samples_split": classifier.min_samples_split,
            "random_state": classifier.random_state,
        },
        "validation_metrics": val_metrics,
        "feature_importances": classifier.feature_importances_,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model saved to:    {model_path} ({os.path.getsize(model_path):,d} bytes)")
    print(f"  Metadata saved to: {meta_path}")
    print("=" * 75)
    print("Phase 5 Model Training Successfully Completed!")
    print("=" * 75)
    return classifier, val_metrics


if __name__ == "__main__":
    run_training()
