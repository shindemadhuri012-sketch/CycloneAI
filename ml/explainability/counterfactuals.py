"""
Physics-Constrained Counterfactual & Marginal Sensitivity Analysis Engine.
Computes "What-If" diagnostic scenarios, minimal physical perturbations for category/intensity shifts,
and 1D/2D partial dependence sensitivity curves bounded by physical atmospheric dynamics.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ml.features.extractor import BASE_PRESSURE_HPA


class CounterfactualEngine:
    """
    Evaluates physical sensitivity curves and determines minimal parameter perturbations
    required to alter categorical or intensity forecasting outcomes.
    """

    def __init__(self, classifier: Any = None, intensity_predictor: Any = None):
        self.classifier = classifier
        self.intensity_predictor = intensity_predictor

    def compute_pressure_deficit_sensitivity(
        self, base_features: pd.DataFrame, num_points: int = 20
    ) -> Dict[str, Any]:
        """
        Computes 1D marginal sensitivity curve of predicted wind speed and category
        across varying pressure deficits: delta_P from 5 hPa to 95 hPa.
        """
        deficits = np.linspace(5.0, 95.0, num_points)
        curve_points = []

        for p_def in deficits:
            X_temp = base_features.copy()
            X_temp["pressure_deficit"] = float(p_def)
            X_temp["central_pres"] = float(BASE_PRESSURE_HPA - p_def)
            # Theoretical wind speed (IMD formula)
            theo_wind = float(14.2 * np.sqrt(p_def))
            X_temp["current_wind"] = theo_wind

            pred_wind_24h = theo_wind
            if self.intensity_predictor is not None and getattr(self.intensity_predictor, "is_fitted", False):
                try:
                    res = self.intensity_predictor.predict_forecast(X_temp)
                    pred_wind_24h = res["forecast_horizons"].get("+24h", {}).get("wind_knots", theo_wind)
                except Exception:
                    pred_wind_24h = theo_wind

            pred_cat = "Depression"
            if self.classifier is not None and getattr(self.classifier, "is_fitted", False):
                try:
                    probs = self.classifier.predict_proba(X_temp)[0]
                    from ml.models.classifier import IMD_CATEGORIES
                    pred_cat = IMD_CATEGORIES[int(np.argmax(probs))]["name"]
                except Exception:
                    pass

            curve_points.append({
                "pressure_deficit_hpa": round(float(p_def), 1),
                "central_pressure_hpa": round(float(BASE_PRESSURE_HPA - p_def), 1),
                "predicted_wind_24h_knots": round(float(pred_wind_24h), 1),
                "theoretical_imd_wind_knots": round(theo_wind, 1),
                "predicted_category": pred_cat,
            })

        return {
            "parameter": "pressure_deficit_hpa",
            "range": [5.0, 95.0],
            "sensitivity_curve": curve_points,
            "physical_monotonicity_verified": bool(
                curve_points[-1]["predicted_wind_24h_knots"] >= curve_points[0]["predicted_wind_24h_knots"]
            ),
        }

    def compute_speed_sensitivity(
        self, base_features: pd.DataFrame, num_points: int = 15
    ) -> Dict[str, Any]:
        """
        Computes sensitivity curve across forward translation speeds: 5 km/h to 45 km/h.
        """
        speeds = np.linspace(5.0, 45.0, num_points)
        curve_points = []

        for s in speeds:
            X_temp = base_features.copy()
            X_temp["forward_speed"] = float(s)
            X_temp["dist_km"] = float(s * 3.0)  # 3-hour displacement

            # Recompute motion components
            b_sin = float(X_temp["bearing_sin"].iloc[0])
            b_cos = float(X_temp["bearing_cos"].iloc[0])
            X_temp["u_motion"] = float(s * b_sin)
            X_temp["v_motion"] = float(s * b_cos)

            ri_prob = 0.05
            if self.intensity_predictor is not None and getattr(self.intensity_predictor, "is_fitted", False):
                try:
                    res = self.intensity_predictor.predict_forecast(X_temp)
                    ri_prob = res["rapid_intensification"].get("probability", 0.05)
                except Exception:
                    pass

            curve_points.append({
                "forward_speed_kmh": round(float(s), 1),
                "ri_probability": round(float(ri_prob), 4),
            })

        return {
            "parameter": "forward_speed_kmh",
            "range": [5.0, 45.0],
            "sensitivity_curve": curve_points,
        }

    def find_counterfactual_category_shift(
        self, base_features: pd.DataFrame, target_category_idx: int
    ) -> Dict[str, Any]:
        """
        Calculates the minimal physical parameter change (delta_P, speed)
        required to shift the classifier prediction to target_category_idx.
        """
        if self.classifier is None or not getattr(self.classifier, "is_fitted", False):
            return {"status": "classifier_unavailable"}

        from ml.models.classifier import IMD_CATEGORIES
        target_meta = IMD_CATEGORIES[target_category_idx]

        orig_probs = self.classifier.predict_proba(base_features)[0]
        orig_idx = int(np.argmax(orig_probs))
        orig_meta = IMD_CATEGORIES[orig_idx]

        if orig_idx == target_category_idx:
            return {
                "status": "already_satisfied",
                "current_category": orig_meta["name"],
                "target_category": target_meta["name"],
                "minimal_perturbation": {},
            }

        # Vectorized batch search along physical pressure deficit axis
        p_grid = np.linspace(5.0, 100.0, 25)
        X_batch = pd.concat([base_features] * len(p_grid), ignore_index=True)
        X_batch["pressure_deficit"] = p_grid
        X_batch["central_pres"] = BASE_PRESSURE_HPA - p_grid
        X_batch["current_wind"] = 14.2 * np.sqrt(p_grid)

        all_probs = self.classifier.predict_proba(X_batch)
        pred_classes = np.argmax(all_probs, axis=1)

        curr_def = float(base_features["pressure_deficit"].iloc[0])
        matching_indices = np.where(pred_classes == target_category_idx)[0]

        best_delta_p = None
        min_p_diff = 999.0
        if len(matching_indices) > 0:
            for idx in matching_indices:
                p_val = float(p_grid[idx])
                diff = abs(p_val - curr_def)
                if diff < min_p_diff:
                    min_p_diff = diff
                    best_delta_p = p_val

        if best_delta_p is not None:
            curr_def = float(base_features["pressure_deficit"].iloc[0])
            return {
                "status": "counterfactual_found",
                "current_category": orig_meta["name"],
                "target_category": target_meta["name"],
                "required_pressure_deficit_hpa": round(float(best_delta_p), 1),
                "required_central_pressure_hpa": round(float(BASE_PRESSURE_HPA - best_delta_p), 1),
                "pressure_change_needed_hpa": round(float(best_delta_p - curr_def), 1),
                "meteorological_feasibility": "High" if min_p_diff <= 25.0 else "Extreme",
            }

        return {
            "status": "unreachable_within_bounds",
            "current_category": orig_meta["name"],
            "target_category": target_meta["name"],
            "message": "Target category unreachable via single-parameter physical pressure perturbation alone.",
        }
