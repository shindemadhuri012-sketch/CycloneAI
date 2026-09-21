"""
Cyclone Identification Model.
Encapsulates an ensemble gradient boosted decision tree classifier with probability calibration
and configurable operational risk thresholds for tropical cyclone detection.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report

from ml.features.extractor import FEATURE_NAMES


class CycloneDetector:
    """
    Binary Tropical Cyclone Identification Model.
    Predicts presence of an organized Cyclonic Storm (V >= 34 kt / IMD CS+).
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        random_state: int = 42,
        operational_threshold: float = 0.35,
        standard_threshold: float = 0.50,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.operational_threshold = operational_threshold
        self.standard_threshold = standard_threshold

        self.feature_names: List[str] = FEATURE_NAMES
        self.base_model: Optional[GradientBoostingClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.is_fitted: bool = False
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "CycloneDetector":
        """
        Fits the gradient boosted base model and calibrates probability outputs on validation data.
        """
        # Ensure column alignment
        X_in = X[self.feature_names]

        self.base_model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self.base_model.fit(X_in, y)

        # Store feature importances
        raw_importances = self.base_model.feature_importances_
        self.feature_importances_ = {
            name: float(imp) for name, imp in zip(self.feature_names, raw_importances)
        }

        # Calibrate probabilities using validation set if provided; otherwise 5-fold CV
        if X_val is not None and y_val is not None:
            n_train = len(X_in)
            n_val = len(X_val)
            X_combined = pd.concat([X_in, X_val[self.feature_names]], ignore_index=True)
            y_combined = pd.concat([y, y_val], ignore_index=True)
            custom_cv = [(np.arange(n_train), np.arange(n_train, n_train + n_val))]
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.base_model,
                method="sigmoid",
                cv=custom_cv
            )
            self.calibrated_model.fit(X_combined, y_combined)
        else:
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.base_model,
                method="sigmoid",
                cv=5
            )
            self.calibrated_model.fit(X_in, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns calibrated class probabilities: shape (N, 2) where col 1 is P(Cyclone).
        """
        if not self.is_fitted:
            raise RuntimeError("CycloneDetector model is not fitted yet.")
        X_in = X[self.feature_names]
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_in)
        return self.base_model.predict_proba(X_in)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary presence (1 = Cyclone, 0 = Non-Cyclone / Disturbance).
        Uses standard_threshold (0.50) if threshold is None.
        """
        thresh = threshold if threshold is not None else self.standard_threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

    def predict_operational(self, X: pd.DataFrame) -> np.ndarray:
        """
        High-sensitivity disaster response prediction using operational_threshold (0.35).
        Minimizes false negatives for coastal protection.
        """
        return self.predict(X, threshold=self.operational_threshold)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "Test") -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics on a designated split.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot evaluate an unfitted model.")

        probs = self.predict_proba(X)[:, 1]
        preds_std = self.predict(X, threshold=self.standard_threshold)
        preds_ops = self.predict(X, threshold=self.operational_threshold)

        auc = float(roc_auc_score(y, probs))
        brier = float(brier_score_loss(y, probs))

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
        acc_std = float(accuracy_score(y, preds_std))
        acc_ops = float(accuracy_score(y, preds_ops))

        prec_std, rec_std, f1_std, _ = precision_recall_fscore_support(y, preds_std, average="binary")
        prec_ops, rec_ops, f1_ops, _ = precision_recall_fscore_support(y, preds_ops, average="binary")

        cm_std = confusion_matrix(y, preds_std).tolist()
        cm_ops = confusion_matrix(y, preds_ops).tolist()

        return {
            "split": split_name,
            "sample_count": len(y),
            "cyclone_positive_rate": float(y.mean()),
            "roc_auc": auc,
            "brier_score": brier,
            "standard_threshold": {
                "threshold": self.standard_threshold,
                "accuracy": acc_std,
                "precision": float(prec_std),
                "recall": float(rec_std),
                "f1_score": float(f1_std),
                "confusion_matrix": cm_std,
            },
            "operational_threshold": {
                "threshold": self.operational_threshold,
                "accuracy": acc_ops,
                "precision": float(prec_ops),
                "recall": float(rec_ops),
                "f1_score": float(f1_ops),
                "confusion_matrix": cm_ops,
            },
            "feature_importances": self.feature_importances_,
        }

    def explain_sample(self, row: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates feature attribution and diagnostic insight for a single sample.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explaining.")

        X_in = row[self.feature_names].iloc[0:1]
        prob = float(self.predict_proba(X_in)[0, 1])

        # Top driving features based on global importance and local deviations
        attributions = []
        for feat in self.feature_names:
            val = float(X_in[feat].iloc[0])
            imp = self.feature_importances_.get(feat, 0.0)
            attributions.append({
                "feature": feat,
                "value": round(val, 3),
                "importance_weight": round(imp, 4),
            })

        attributions.sort(key=lambda x: x["importance_weight"], reverse=True)

        # Risk classification
        if prob >= 0.70:
            risk = "Severe"
        elif prob >= 0.50:
            risk = "High"
        elif prob >= 0.35:
            risk = "Moderate (Disaster Alert Threshold Exceeded)"
        else:
            risk = "Low"

        return {
            "cyclone_probability": round(prob, 4),
            "is_cyclone_standard": prob >= self.standard_threshold,
            "is_cyclone_operational": prob >= self.operational_threshold,
            "risk_level": risk,
            "top_features": attributions[:6],
        }

    def save(self, filepath: str) -> None:
        """
        Serializes model instance and metadata to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "base_model": self.base_model,
            "calibrated_model": self.calibrated_model,
            "feature_importances": self.feature_importances_,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "random_state": self.random_state,
                "operational_threshold": self.operational_threshold,
                "standard_threshold": self.standard_threshold,
            },
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "CycloneDetector":
        """
        Deserializes a saved CycloneDetector instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")

        payload = joblib.load(filepath)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.base_model = payload.get("base_model")
        instance.calibrated_model = payload.get("calibrated_model")
        instance.feature_importances_ = payload.get("feature_importances", {})
        instance.feature_names = payload.get("feature_names", FEATURE_NAMES)
        instance.is_fitted = payload.get("is_fitted", False)
        return instance
