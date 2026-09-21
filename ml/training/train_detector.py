"""
Master Training Script for Phase 4: Cyclone Identification Model.
Trains the CycloneDetector ensemble on real preprocessed manifests,
calibrates posterior probabilities on validation data, and exports model checkpoints.
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
from ml.models.detector import CycloneDetector


def run_training():
    print("=" * 70)
    print("CycloneAI — Phase 4: Cyclone Identification Model Training")
    print("=" * 70)

    train_path = os.path.join(PROJECT_ROOT, "data/processed/train_manifest.csv")
    val_path = os.path.join(PROJECT_ROOT, "data/processed/val_manifest.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Preprocessed manifests not found. Expected:\n  {train_path}\n  {val_path}"
        )

    print(f"\n[1/5] Loading real preprocessed manifests...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"  Training Split:   {len(train_df):,d} observations across {train_df['SID'].nunique()} unique storms")
    print(f"  Validation Split: {len(val_df):,d} observations across {val_df['SID'].nunique()} unique storms")

    print(f"\n[2/5] Extracting physics-grounded features & targets...")
    extractor = CycloneFeatureExtractor()
    X_train, y_train_cs, y_train_dep = extractor.extract_from_dataframe(train_df)
    X_val, y_val_cs, y_val_dep = extractor.extract_from_dataframe(val_df)

    print(f"  Feature count: {X_train.shape[1]} features")
    print(f"  Features: {', '.join(extractor.feature_names)}")
    print(f"  Train Target: {y_train_cs.sum():,d} Cyclonic Storms ({y_train_cs.mean()*100:.1f}%), {(y_train_cs==0).sum():,d} Non-Cyclonic ({(y_train_cs==0).mean()*100:.1f}%)")
    print(f"  Val Target:   {y_val_cs.sum():,d} Cyclonic Storms ({y_val_cs.mean()*100:.1f}%), {(y_val_cs==0).sum():,d} Non-Cyclonic ({(y_val_cs==0).mean()*100:.1f}%)")

    print(f"\n[3/5] Fitting CycloneDetector & calibrating posterior probabilities...")
    detector = CycloneDetector(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42,
        operational_threshold=0.35,
        standard_threshold=0.50,
    )
    detector.fit(X_train, y_train_cs, X_val, y_val_cs)

    print(f"\n[4/5] Evaluating on Training and Validation splits...")
    train_metrics = detector.evaluate(X_train, y_train_cs, split_name="Train")
    val_metrics = detector.evaluate(X_val, y_val_cs, split_name="Validation")

    print("\n--- Training Set Performance ---")
    print(f"  ROC-AUC:            {train_metrics['roc_auc']:.4f}")
    print(f"  Brier Score:        {train_metrics['brier_score']:.4f}")
    print(f"  Standard (p>=0.50): Accuracy={train_metrics['standard_threshold']['accuracy']*100:.2f}%, Precision={train_metrics['standard_threshold']['precision']:.4f}, Recall={train_metrics['standard_threshold']['recall']:.4f}, F1={train_metrics['standard_threshold']['f1_score']:.4f}")
    print(f"  Operational (p>=0.35): Accuracy={train_metrics['operational_threshold']['accuracy']*100:.2f}%, Recall={train_metrics['operational_threshold']['recall']:.4f}")

    print("\n--- Validation Set Performance ---")
    print(f"  ROC-AUC:            {val_metrics['roc_auc']:.4f}")
    print(f"  Brier Score:        {val_metrics['brier_score']:.4f}")
    print(f"  Standard (p>=0.50): Accuracy={val_metrics['standard_threshold']['accuracy']*100:.2f}%, Precision={val_metrics['standard_threshold']['precision']:.4f}, Recall={val_metrics['standard_threshold']['recall']:.4f}, F1={val_metrics['standard_threshold']['f1_score']:.4f}")
    print(f"  Standard Confusion Matrix:\n    {val_metrics['standard_threshold']['confusion_matrix']}")
    print(f"  Operational (p>=0.35): Accuracy={val_metrics['operational_threshold']['accuracy']*100:.2f}%, Precision={val_metrics['operational_threshold']['precision']:.4f}, Recall={val_metrics['operational_threshold']['recall']:.4f}, F1={val_metrics['operational_threshold']['f1_score']:.4f}")
    print(f"  Operational Confusion Matrix:\n    {val_metrics['operational_threshold']['confusion_matrix']}")

    print("\n--- Top Physical Feature Importances ---")
    sorted_feats = sorted(detector.feature_importances_.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_feats:
        bar = "#" * int(imp * 40)
        print(f"  {feat:<18} {imp:.4f}  |{bar}")

    print(f"\n[5/5] Serializing model and metadata to disk...")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "ml/models/saved")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "cyclone_detector.joblib")
    meta_path = os.path.join(checkpoint_dir, "detector_metadata.json")

    detector.save(model_path)

    metadata = {
        "model_name": "CycloneDetector",
        "version": "1.0.0",
        "phase": "Phase 4 - Cyclone Identification",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(train_df),
        "training_unique_storms": int(train_df["SID"].nunique()),
        "validation_samples": len(val_df),
        "validation_unique_storms": int(val_df["SID"].nunique()),
        "features": extractor.feature_names,
        "hyperparameters": {
            "n_estimators": detector.n_estimators,
            "max_depth": detector.max_depth,
            "learning_rate": detector.learning_rate,
            "subsample": detector.subsample,
            "random_state": detector.random_state,
            "operational_threshold": detector.operational_threshold,
            "standard_threshold": detector.standard_threshold,
        },
        "validation_metrics": val_metrics,
        "feature_importances": detector.feature_importances_,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Model saved to:    {model_path} ({os.path.getsize(model_path):,d} bytes)")
    print(f"  Metadata saved to: {meta_path}")
    print("=" * 70)
    print("Model Training Successfully Completed!")
    print("=" * 70)
    return detector, val_metrics


if __name__ == "__main__":
    run_training()
