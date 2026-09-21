# ML Evaluation Module (`ml/evaluation/`)

## Purpose
Provides standardized, objective metric calculation scripts aligned with meteorological operational standards.

## Evaluated Metrics
1. **Detection & Classification**:
   - Precision, Recall, F1 Score, Balanced Accuracy.
   - Confusion Matrix mapping IMD categories (Depression, Deep Depression, CS, SCS, VSCS, ESCS, SuCS).
2. **Intensity Prediction**:
   - Mean Absolute Error (MAE in knots / hPa).
   - Root Mean Square Error (RMSE in knots / hPa).
   - Bias analysis across intensity bins (under-estimation of rapid intensification).
3. **Track Prediction**:
   - Average Track Error (ATE in kilometers) across 6h, 12h, 24h, and 48h lead times.
   - Cross-track and along-track error decomposition.

*Zero metric values are fabricated. Metrics will be calculated after model training.*
