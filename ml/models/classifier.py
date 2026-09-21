"""
Cyclone Pattern & Category Classification Model.
Classifies tropical cyclone systems into the 8 official India Meteorological Department (IMD)
intensity categories using a physics-grounded, class-balanced ensemble.
"""
from typing import Dict, List, Optional, Tuple, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from ml.features.extractor import FEATURE_NAMES

# Official IMD Tropical Cyclone Classification Scale
IMD_CATEGORIES: List[Dict[str, Any]] = [
    {
        "index": 0,
        "code": "LOW",
        "name": "Low Pressure Area",
        "wind_kt": "< 17",
        "wind_kmh": "< 31",
        "damage_potential": "Minimal. Scattered cloudiness and localized rainfall.",
    },
    {
        "index": 1,
        "code": "D",
        "name": "Depression",
        "wind_kt": "17 - 27",
        "wind_kmh": "31 - 49",
        "damage_potential": "Minor. Rough seas, squally winds, localized coastal inundation.",
    },
    {
        "index": 2,
        "code": "DD",
        "name": "Deep Depression",
        "wind_kt": "28 - 33",
        "wind_kmh": "50 - 61",
        "damage_potential": "Moderate. Heavy rain, high seas; fishermen advised not to venture into sea.",
    },
    {
        "index": 3,
        "code": "CS",
        "name": "Cyclonic Storm",
        "wind_kt": "34 - 47",
        "wind_kmh": "62 - 88",
        "damage_potential": "Moderate to High. Gale-force winds; damage to thatched huts, breaking of tree branches.",
    },
    {
        "index": 4,
        "code": "SCS",
        "name": "Severe Cyclonic Storm",
        "wind_kt": "48 - 63",
        "wind_kmh": "89 - 117",
        "damage_potential": "High. Uprooting of large trees, extensive damage to kutcha houses, storm surge 1.0 - 1.5m.",
    },
    {
        "index": 5,
        "code": "VSCS",
        "name": "Very Severe Cyclonic Storm",
        "wind_kt": "64 - 89",
        "wind_kmh": "118 - 165",
        "damage_potential": "Very High. Extensive damage to structures, power/communication disruption, storm surge 2.0 - 2.5m.",
    },
    {
        "index": 6,
        "code": "ESCS",
        "name": "Extremely Severe Cyclonic Storm",
        "wind_kt": "90 - 119",
        "wind_kmh": "166 - 220",
        "damage_potential": "Devastating. Catastrophic damage, widespread building destruction, storm surge > 3.0m.",
    },
    {
        "index": 7,
        "code": "SuCS",
        "name": "Super Cyclonic Storm",
        "wind_kt": ">= 120",
        "wind_kmh": ">= 221",
        "damage_potential": "Total Catastrophe. Complete annihilation of infrastructure, extreme storm surge > 5.0m.",
    },
]

# Mapping from raw best-track / IMD strings to integer index (0 - 7)
IMD_GRADE_TO_INDEX: Dict[str, int] = {
    "LOW": 0, "L": 0,
    "D": 1,
    "DD": 2,
    "CS": 3,
    "SCS": 4, "SCS(H)": 4,
    "VSCS": 5,
    "ESCS": 6,
    "SuCS": 7, "SUCS": 7,
}

CATEGORY_NAMES: List[str] = [cat["name"] for cat in IMD_CATEGORIES]
CATEGORY_CODES: List[str] = [cat["code"] for cat in IMD_CATEGORIES]


class CycloneClassifier:
    """
    Multi-class Tropical Cyclone Category Classifier.
    Predicts probability distribution across the 8 IMD intensity categories.
    """

    def __init__(
        self,
        n_estimators: int = 160,
        max_depth: int = 12,
        min_samples_split: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.feature_names: List[str] = FEATURE_NAMES
        self.num_classes: int = len(IMD_CATEGORIES)
        self.rf_model: Optional[RandomForestClassifier] = None
        self.hgb_model: Optional[HistGradientBoostingClassifier] = None
        self.is_fitted: bool = False
        self.feature_importances_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CycloneClassifier":
        """
        Fits class-balanced ensemble combining Random Forest with balanced subsampling
        and sample-weighted HistGradientBoosting.
        """
        X_in = X[self.feature_names]

        # 1. Random Forest with balanced subsample weighting
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_model.fit(X_in, y)

        # 2. HistGradientBoosting with inverse class frequency sample weights
        sample_weights = compute_sample_weight("balanced", y)
        self.hgb_model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            learning_rate=0.08,
            random_state=self.random_state,
        )
        self.hgb_model.fit(X_in, y, sample_weight=sample_weights)

        # 3. Store feature importances from the Random Forest component
        raw_importances = self.rf_model.feature_importances_
        self.feature_importances_ = {
            name: float(imp) for name, imp in zip(self.feature_names, raw_importances)
        }

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns blended posterior probability distribution across all 8 IMD categories.
        Ensemble weights: 60% Balanced RF + 40% Weighted HistGradientBoosting.
        Shape: (N, 8) where each row sums to 1.0.
        """
        if not self.is_fitted:
            raise RuntimeError("CycloneClassifier model is not fitted yet.")

        X_in = X[self.feature_names]
        rf_probs = self.rf_model.predict_proba(X_in)
        hgb_probs = self.hgb_model.predict_proba(X_in)

        # Ensure all 8 classes are represented in probability array
        full_rf_probs = np.zeros((len(X_in), self.num_classes), dtype=np.float32)
        full_hgb_probs = np.zeros((len(X_in), self.num_classes), dtype=np.float32)

        for i, cls in enumerate(self.rf_model.classes_):
            if cls < self.num_classes:
                full_rf_probs[:, int(cls)] = rf_probs[:, i]

        for i, cls in enumerate(self.hgb_model.classes_):
            if cls < self.num_classes:
                full_hgb_probs[:, int(cls)] = hgb_probs[:, i]

        blended = 0.60 * full_rf_probs + 0.40 * full_hgb_probs
        # Re-normalize rows to ensure exact sum to 1.0
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return blended / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns argmax predicted category index (0 to 7).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "Test") -> Dict[str, Any]:
        """
        Computes comprehensive multi-class evaluation metrics, including per-class
        metrics for all 8 categories, Top-1 accuracy, adjacent match accuracy, and MACE.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot evaluate unfitted classifier.")

        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        y_arr = np.asarray(y)

        # Exact and Ordinal Proximity Metrics
        abs_diff = np.abs(preds - y_arr)
        exact_acc = float(np.mean(abs_diff == 0))
        adjacent_acc = float(np.mean(abs_diff <= 1))
        mace = float(np.mean(abs_diff))

        # Top-2 Accuracy (True class in top 2 predicted probabilities)
        top2_preds = np.argsort(probs, axis=1)[:, -2:]
        top2_acc = float(np.mean([y_arr[i] in top2_preds[i] for i in range(len(y_arr))]))

        # Per-class metrics
        report_dict = classification_report(
            y_arr,
            preds,
            labels=list(range(self.num_classes)),
            target_names=CATEGORY_NAMES,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_arr,
            preds,
            labels=list(range(self.num_classes)),
        ).tolist()

        per_class = []
        for i, cat in enumerate(IMD_CATEGORIES):
            cat_name = cat["name"]
            metrics = report_dict.get(cat_name, {})
            per_class.append({
                "index": i,
                "code": cat["code"],
                "name": cat_name,
                "precision": round(float(metrics.get("precision", 0.0)), 4),
                "recall": round(float(metrics.get("recall", 0.0)), 4),
                "f1_score": round(float(metrics.get("f1-score", 0.0)), 4),
                "support": int(metrics.get("support", 0)),
            })

        return {
            "split": split_name,
            "sample_count": len(y),
            "top1_exact_accuracy": round(exact_acc, 4),
            "adjacent_accuracy": round(adjacent_acc, 4),
            "top2_accuracy": round(top2_acc, 4),
            "mean_absolute_category_error": round(mace, 4),
            "macro_avg": {
                "precision": round(float(report_dict["macro avg"]["precision"]), 4),
                "recall": round(float(report_dict["macro avg"]["recall"]), 4),
                "f1_score": round(float(report_dict["macro avg"]["f1-score"]), 4),
            },
            "weighted_avg": {
                "precision": round(float(report_dict["weighted avg"]["precision"]), 4),
                "recall": round(float(report_dict["weighted avg"]["recall"]), 4),
                "f1_score": round(float(report_dict["weighted avg"]["f1-score"]), 4),
            },
            "per_class": per_class,
            "confusion_matrix": cm,
            "feature_importances": self.feature_importances_,
        }

    def explain_sample(self, row: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates comprehensive multi-class attribution and diagnostic profile for a single sample.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before explaining.")

        X_in = row[self.feature_names].iloc[0:1]
        probs = self.predict_proba(X_in)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Sort categories by probability
        ranked_indices = np.argsort(probs)[::-1]
        top_categories = []
        for idx in ranked_indices[:3]:
            cat = IMD_CATEGORIES[idx]
            top_categories.append({
                "index": int(idx),
                "code": cat["code"],
                "name": cat["name"],
                "probability": round(float(probs[idx]), 4),
                "wind_kt": cat["wind_kt"],
                "wind_kmh": cat["wind_kmh"],
            })

        cat_meta = IMD_CATEGORIES[pred_idx]

        # Top driving features
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

        return {
            "predicted_category": f"{cat_meta['name']} ({cat_meta['code']})",
            "category_code": cat_meta["code"],
            "category_index": pred_idx,
            "confidence": round(confidence, 4),
            "wind_range_kt": cat_meta["wind_kt"],
            "wind_range_kmh": cat_meta["wind_kmh"],
            "damage_potential": cat_meta["damage_potential"],
            "top_categories": top_categories,
            "all_probabilities": {
                cat["code"]: round(float(probs[i]), 4)
                for i, cat in enumerate(IMD_CATEGORIES)
            },
            "top_features": attributions[:6],
        }

    def save(self, filepath: str) -> None:
        """
        Serializes model instance to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "rf_model": self.rf_model,
            "hgb_model": self.hgb_model,
            "feature_importances": self.feature_importances_,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "CycloneClassifier":
        """
        Deserializes a saved CycloneClassifier instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")

        payload = joblib.load(filepath)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.rf_model = payload.get("rf_model")
        instance.hgb_model = payload.get("hgb_model")
        instance.feature_importances_ = payload.get("feature_importances", {})
        instance.feature_names = payload.get("feature_names", FEATURE_NAMES)
        instance.is_fitted = payload.get("is_fitted", False)
        return instance
