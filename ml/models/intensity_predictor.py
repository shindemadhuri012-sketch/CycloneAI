"""
Cyclone Intensity Prediction Model.
Provides multi-horizon regression for Maximum Sustained Wind Speed (Vmax) and Minimum Central Pressure (Pmin),
quantile-based 90% uncertainty estimation, Rapid Intensification (RI) alert gating,
and IMD empirical hydrodynamic physical consistency enforcement.
"""
from typing import Dict, List, Optional, Tuple, Any
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score

from ml.features.extractor import FEATURE_NAMES, BASE_PRESSURE_HPA

# Lead forecast horizons (hours)
FORECAST_HORIZONS: List[int] = [0, 6, 12, 24]
# Official IMD/NHC Rapid Intensification threshold: wind speed increase >= 30 kt in 24 hours
RI_THRESHOLD_KT: float = 30.0
# IMD empirical wind-pressure coefficient (Mishra & Gupta): Vmax = 14.2 * sqrt(1010 - Pmin)
IMD_COEFF: float = 14.2

INTENSITY_FEATURE_NAMES: List[str] = FEATURE_NAMES + [
    "current_wind",
    "past_dv_6h",
    "past_dp_6h",
    "intensity_balance_ratio",
]


class CycloneIntensityPredictor:
    """
    Multi-horizon Cyclone Intensity Predictor with Rapid Intensification detection
    and quantile uncertainty estimation.
    """

    def __init__(
        self,
        random_state: int = 42,
        ri_threshold_kt: float = RI_THRESHOLD_KT,
    ):
        self.random_state = random_state
        self.ri_threshold_kt = ri_threshold_kt
        self.horizons = FORECAST_HORIZONS
        self.feature_names = INTENSITY_FEATURE_NAMES

        # Forecasters per horizon: {h: model}
        self.wind_models: Dict[int, HistGradientBoostingRegressor] = {}
        self.pres_models: Dict[int, HistGradientBoostingRegressor] = {}

        # Quantile forecasters for uncertainty bounds (alpha=0.10, alpha=0.90)
        self.wind_q10: Dict[int, HistGradientBoostingRegressor] = {}
        self.wind_q90: Dict[int, HistGradientBoostingRegressor] = {}
        self.pres_q10: Dict[int, HistGradientBoostingRegressor] = {}
        self.pres_q90: Dict[int, HistGradientBoostingRegressor] = {}

        # Rapid Intensification (RI) Classifier
        self.ri_model: Optional[CalibratedClassifierCV] = None
        self.is_fitted: bool = False

    def _build_features(self, df_or_dict: Any) -> pd.DataFrame:
        """Ensures all features are present with appropriate defaults."""
        if isinstance(df_or_dict, pd.DataFrame):
            X = df_or_dict.copy()
        else:
            X = pd.DataFrame([df_or_dict])

        # Core physical features
        for feat in FEATURE_NAMES:
            if feat not in X.columns:
                X[feat] = 0.0

        # Intensity specific features
        if "current_wind" not in X.columns:
            # Impute from central pressure deficit using IMD formula if wind is missing
            deficit = X["pressure_deficit"].clip(lower=0.0)
            X["current_wind"] = (IMD_COEFF * np.sqrt(deficit)).clip(lower=15.0, upper=140.0)

        if "past_dv_6h" not in X.columns:
            X["past_dv_6h"] = 0.0
        if "past_dp_6h" not in X.columns:
            X["past_dp_6h"] = 0.0

        deficit = np.maximum(0.1, BASE_PRESSURE_HPA - X["central_pres"])
        theoretical_wind = IMD_COEFF * np.sqrt(deficit)
        X["intensity_balance_ratio"] = (X["current_wind"] / np.maximum(10.0, theoretical_wind)).clip(0.4, 2.5)

        return X[self.feature_names]

    def fit(
        self,
        X_dict: Dict[int, pd.DataFrame],
        y_wind_dict: Dict[int, pd.Series],
        y_pres_dict: Dict[int, pd.Series],
        y_ri: Optional[pd.Series] = None,
        X_ri: Optional[pd.DataFrame] = None,
    ) -> "CycloneIntensityPredictor":
        """
        Fits multi-horizon regression models, quantile uncertainty models, and RI classifier.
        """
        for h in self.horizons:
            if h not in X_dict or h not in y_wind_dict or h not in y_pres_dict:
                continue

            X_h = self._build_features(X_dict[h])
            y_w = y_wind_dict[h].values
            y_p = y_pres_dict[h].values

            # 1. Median / Mean wind forecaster (squared error loss)
            w_model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=150,
                max_depth=7,
                learning_rate=0.07,
                random_state=self.random_state,
            )
            w_model.fit(X_h, y_w)
            self.wind_models[h] = w_model

            # Quantile wind bounds
            w_q10 = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.10,
                max_iter=120, max_depth=6, learning_rate=0.08,
                random_state=self.random_state,
            )
            w_q10.fit(X_h, y_w)
            self.wind_q10[h] = w_q10

            w_q90 = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.90,
                max_iter=120, max_depth=6, learning_rate=0.08,
                random_state=self.random_state,
            )
            w_q90.fit(X_h, y_w)
            self.wind_q90[h] = w_q90

            # 2. Pressure forecaster (squared error loss)
            p_model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=150,
                max_depth=7,
                learning_rate=0.07,
                random_state=self.random_state,
            )
            p_model.fit(X_h, y_p)
            self.pres_models[h] = p_model

            # Quantile pressure bounds
            p_q10 = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.10,
                max_iter=120, max_depth=6, learning_rate=0.08,
                random_state=self.random_state,
            )
            p_q10.fit(X_h, y_p)
            self.pres_q10[h] = p_q10

            p_q90 = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.90,
                max_iter=120, max_depth=6, learning_rate=0.08,
                random_state=self.random_state,
            )
            p_q90.fit(X_h, y_p)
            self.pres_q90[h] = p_q90

        # 3. Rapid Intensification (RI) Classifier
        if X_ri is not None and y_ri is not None:
            X_ri_feat = self._build_features(X_ri)
            base_ri = GradientBoostingClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.random_state,
            )
            self.ri_model = CalibratedClassifierCV(
                estimator=base_ri,
                method="sigmoid",
                cv=3,
            )
            self.ri_model.fit(X_ri_feat, y_ri)

        self.is_fitted = True
        return self

    def enforce_hydrodynamic_consistency(
        self, wind_kt: float, pres_hpa: float
    ) -> Tuple[float, float]:
        """
        Applies physical consistency checks based on the empirical IMD pressure-wind relation.
        Ensures predicted wind speed and predicted central pressure remain coupled.
        """
        # Wind cannot exceed theoretical gradient limit for the pressure deficit:
        # V_limit = 14.2 * sqrt(max(0, 1012 - P)) + 15
        deficit = max(0.0, BASE_PRESSURE_HPA - pres_hpa)
        theoretical_wind = IMD_COEFF * np.sqrt(deficit) if deficit > 0 else 15.0

        # Bound wind within +/- 20 kt of hydrodynamic curve
        upper_wind_limit = theoretical_wind + 22.0
        lower_wind_limit = max(15.0, theoretical_wind - 22.0)

        cons_wind = float(np.clip(wind_kt, lower_wind_limit, upper_wind_limit))

        # Reciprocally bound central pressure
        # P_theoretical = 1010 - (V / 14.2)^2
        pres_theoretical = BASE_PRESSURE_HPA - ((cons_wind / IMD_COEFF) ** 2)
        upper_pres_limit = min(1015.0, pres_theoretical + 18.0)
        lower_pres_limit = max(870.0, pres_theoretical - 18.0)

        cons_pres = float(np.clip(pres_hpa, lower_pres_limit, upper_pres_limit))
        return round(cons_wind, 1), round(cons_pres, 1)

    def predict_forecast(self, row: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates multi-horizon intensity forecast with uncertainty bounds and RI alert.
        """
        if not self.is_fitted:
            raise RuntimeError("CycloneIntensityPredictor is not fitted.")

        X_feat = self._build_features(row)
        horizons_out: Dict[str, Any] = {}

        current_w = float(row.get("current_wind", [35.0])[0]) if "current_wind" in row else 35.0
        current_p = float(row.get("central_pres", [995.0])[0]) if "central_pres" in row else 995.0

        for h in [6, 12, 24]:
            if h not in self.wind_models or h not in self.pres_models:
                continue

            raw_w = float(self.wind_models[h].predict(X_feat)[0])
            raw_p = float(self.pres_models[h].predict(X_feat)[0])

            # Hydrodynamic consistency check
            cons_w, cons_p = self.enforce_hydrodynamic_consistency(raw_w, raw_p)

            # Quantile bounds
            w_q10 = float(self.wind_q10[h].predict(X_feat)[0])
            w_q90 = float(self.wind_q90[h].predict(X_feat)[0])
            p_q10 = float(self.pres_q10[h].predict(X_feat)[0])
            p_q90 = float(self.pres_q90[h].predict(X_feat)[0])

            # Monotonic quantile bound sorting
            w_low, w_high = min(w_q10, cons_w), max(w_q90, cons_w)
            p_low, p_high = min(p_q10, cons_p), max(p_q90, cons_p)

            horizons_out[f"+{h}h"] = {
                "horizon_hours": h,
                "wind_knots": cons_w,
                "wind_kmh": round(cons_w * 1.852, 1),
                "pressure_hpa": cons_p,
                "pressure_deficit_hpa": round(max(0.0, BASE_PRESSURE_HPA - cons_p), 1),
                "wind_bounds_90": {
                    "low_knots": round(w_low, 1),
                    "high_knots": round(w_high, 1),
                },
                "pressure_bounds_90": {
                    "low_hpa": round(p_low, 1),
                    "high_hpa": round(p_high, 1),
                },
            }

        # Rapid Intensification (RI) Prediction
        ri_prob = 0.0
        is_ri = False
        if self.ri_model is not None:
            ri_probs = self.ri_model.predict_proba(X_feat)[0]
            ri_prob = float(ri_probs[1]) if len(ri_probs) > 1 else 0.0
            is_ri = bool(ri_prob >= 0.040)  # High-sensitivity operational RI threshold calibrated to 3.4% base rate
        else:
            # Fallback heuristic from +24h predicted delta
            pred_24w = horizons_out.get("+24h", {}).get("wind_knots", current_w)
            delta_24 = pred_24w - current_w
            is_ri = bool(delta_24 >= self.ri_threshold_kt)
            ri_prob = float(min(1.0, max(0.0, delta_24 / self.ri_threshold_kt)))

        # 24h wind change and risk level
        pred_24w = horizons_out.get("+24h", {}).get("wind_knots", current_w)
        delta_24 = round(pred_24w - current_w, 1)

        if is_ri or delta_24 >= 30.0:
            ri_risk = "Critical (Rapid Intensification Detected)"
            trend = "Rapidly Intensifying"
        elif delta_24 >= 15.0:
            ri_risk = "High"
            trend = "Steadily Intensifying"
        elif delta_24 >= -5.0:
            ri_risk = "Moderate"
            trend = "Near Steady State"
        else:
            ri_risk = "Low"
            trend = "Weakening / Landfall Dissipation"

        return {
            "current_intensity": {
                "wind_knots": round(current_w, 1),
                "wind_kmh": round(current_w * 1.852, 1),
                "pressure_hpa": round(current_p, 1),
                "pressure_deficit_hpa": round(max(0.0, BASE_PRESSURE_HPA - current_p), 1),
            },
            "forecast_horizons": horizons_out,
            "rapid_intensification": {
                "is_alert_active": is_ri,
                "probability": round(ri_prob, 4),
                "threshold_knots_24h": self.ri_threshold_kt,
                "predicted_24h_wind_change_knots": delta_24,
                "risk_level": ri_risk,
            },
            "intensity_trend": trend,
        }

    def save(self, filepath: str) -> None:
        """Serializes model payload to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "wind_models": self.wind_models,
            "pres_models": self.pres_models,
            "wind_q10": self.wind_q10,
            "wind_q90": self.wind_q90,
            "pres_q10": self.pres_q10,
            "pres_q90": self.pres_q90,
            "ri_model": self.ri_model,
            "horizons": self.horizons,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, filepath, compress=3)

    @classmethod
    def load(cls, filepath: str) -> "CycloneIntensityPredictor":
        """Deserializes a saved CycloneIntensityPredictor instance."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        payload = joblib.load(filepath)
        instance = cls()
        instance.wind_models = payload.get("wind_models", {})
        instance.pres_models = payload.get("pres_models", {})
        instance.wind_q10 = payload.get("wind_q10", {})
        instance.wind_q90 = payload.get("wind_q90", {})
        instance.pres_q10 = payload.get("pres_q10", {})
        instance.pres_q90 = payload.get("pres_q90", {})
        instance.ri_model = payload.get("ri_model")
        instance.horizons = payload.get("horizons", FORECAST_HORIZONS)
        instance.feature_names = payload.get("feature_names", INTENSITY_FEATURE_NAMES)
        instance.is_fitted = payload.get("is_fitted", False)
        return instance
