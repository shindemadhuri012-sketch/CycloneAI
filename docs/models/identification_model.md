# CycloneAI — Phase 4: Cyclone Identification Model Technical Report

**Project**: CycloneAI — Intelligent Tropical Cyclone Analysis & Prediction System  
**Problem Statement**: *"To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data."*  
**Milestone**: Phase 4 — Cyclone Identification Model  
**Status**: **COMPLETED & VERIFIED**  
**Date**: September 2026  

---

## 1. Executive Summary

Phase 4 delivers the foundational AI detection engine of the CycloneAI platform: the **Cyclone Identification Model** (`CycloneDetector`). The model addresses the primary meteorological question: **Is an organized Tropical Cyclone present?**

Trained strictly on real, quality-audited North Indian Ocean historical cyclone tracks from Phase 3, the model operates with **0.00% storm-level data leakage** across training, validation, and testing splits.

### Key Benchmark Highlights on Completely Unseen Test Split (70 Storms, 2,223 Observations)
* **ROC-AUC Score**: **0.9319**
* **Brier Calibration Score**: **0.0683** (outstanding probability reliability)
* **Standard Threshold ($p \ge 0.50$)**:
  * **Accuracy**: **91.09%**
  * **Precision**: **0.8059**
  * **Recall**: **0.7575**
  * **F1-Score**: **0.7810**
* **Operational Disaster Response Threshold ($p \ge 0.35$)**:
  * **Accuracy**: **89.61%**
  * **Recall (Sensitivity)**: **0.8026** (high sensitivity to minimize missed coastal threats)
  * **F1-Score**: **0.7640**
* **Storm-Level Lifecycle Detection**: **21 of 23 cyclonic storms (91.3%)** correctly identified during their lifecycle.
* **Sub-Basin Generalization**:
  * Bay of Bengal (BB): **0.9264 ROC-AUC** (1,653 obs)
  * Arabian Sea (AS): **0.9537 ROC-AUC** (570 obs)

---

## 2. Problem Formulation & Meteorological Standards

In international and regional tropical meteorology:
* **World Meteorological Organization (WMO)** and **India Meteorological Department (IMD)** define a system as a **Cyclonic Storm (CS)** when maximum sustained 3-minute surface winds reach or exceed **34 knots** ($62\text{ km/h}$ / $17.5\text{ m/s}$).
* At 34 knots, gale-force winds are initiated, the system receives an official regional cyclone name, and coastal disaster warnings are hoisted.
* Below 34 knots, systems are classified as **Depression** (17–27 kt), **Deep Depression** (28–33 kt), or **Low Pressure Area** (< 17 kt).

### Target Definition ($y \in \{0, 1\}$)
* **Positive Class ($y = 1$)**: Cyclonic Storm and above (Observed sustained wind $V_{\max} \ge 34.0\text{ kt}$ or IMD category in `{"CS", "SCS", "VSCS", "ESCS", "SuCS"}`).
* **Negative Class ($y = 0$)**: Sub-cyclonic disturbance, depression, deep depression, or low-pressure area ($V_{\max} < 34.0\text{ kt}$, IMD category `LOW`, `D`, `DD`).

---

## 3. Physical Feature Engineering

The feature extractor ([`ml/features/extractor.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/ml/features/extractor.py)) derives 14 physically consistent features:

| Feature Name | Type | Physical Dimension | Meteorological Rationale |
| :--- | :--- | :--- | :--- |
| `pressure_deficit` | Thermodynamic | $\text{hPa}$ | $\Delta P = 1010.0 - P_{\min}$. Primary hydrostatic driver of gradient wind balance. |
| `central_pres` | Thermodynamic | $\text{hPa}$ | Minimum central sea-level pressure at vortex core ($850 - 1020\text{ hPa}$). |
| `lat` | Spatial | Degrees N | Latitude coordinates. Controls thermal boundary conditions and SST. |
| `lon` | Spatial | Degrees E | Longitude coordinates in North Indian Ocean ($45^\circ\text{E} - 105^\circ\text{E}$). |
| `coriolis_f` | Planetary Dynamic | $\text{s}^{-1} \times 10^4$ | $f = 2\Omega \sin(\phi)$. Essential for cyclonic vorticity; near-zero at equator ($< 5^\circ\text{N}$). |
| `forward_speed` | Kinematic | $\text{km/h}$ | Vortex translation speed ($0 - 150\text{ km/h}$). Fast motion inhibits vertical stacking. |
| `dist_km` | Kinematic | $\text{km}$ | Step displacement distance traversed in the observation interval. |
| `dt_hours` | Temporal | Hours | Time interval between consecutive track fixes ($0.5 - 24.0\text{ h}$). |
| `bearing_sin` | Kinematic | $[-1.0, 1.0]$ | $\sin(\theta \times \pi / 180)$. Directional orientation component. |
| `bearing_cos` | Kinematic | $[-1.0, 1.0]$ | $\cos(\theta \times \pi / 180)$. North-South directional component. |
| `month_sin` | Climatological | $[-1.0, 1.0]$ | $\sin(2\pi \times \text{month} / 12)$. Captures bimodal North Indian Ocean seasonality. |
| `month_cos` | Climatological | $[-1.0, 1.0]$ | $\cos(2\pi \times \text{month} / 12)$. Distinguishes pre-monsoon (May) & post-monsoon (Oct-Nov). |
| `day_of_year` | Climatological | $[0.0, 1.0]$ | Normalized day-of-year ($t_{\text{doy}} / 365.25$). |
| `is_bob` | Geographic | Binary $\{0, 1\}$ | Bay of Bengal flag (accounts for ~80% of North Indian Ocean cyclogenesis). |

---

## 4. Model Architecture & Calibration

### Core Architecture: `CycloneDetector`
* **Base Learner**: Tuned Gradient Boosted Decision Trees (`GradientBoostingClassifier`):
  * `n_estimators = 150`
  * `max_depth = 5`
  * `learning_rate = 0.08`
  * `subsample = 0.85`
* **Probability Calibration**: Sigmoid Platt scaling via `CalibratedClassifierCV` using cross-validation over the training manifest, ensuring that raw margin scores are mapped to true empirical posterior probabilities $P(y=1|x) \in [0.0, 1.0]$.
* **Decision Policies**:
  1. **Standard Policy ($p \ge 0.50$)**: Maximizes overall balanced accuracy and F1 score for scientific benchmarking.
  2. **Operational Disaster Policy ($p \ge 0.35$)**: High-sensitivity threshold configured for disaster management authorities to minimize False Negatives (missed cyclones).

### Global Physical Feature Importances
From tree splits across 150 gradient-boosted stages:

```
  Feature             Importance   Contribution Bar
  -------------------------------------------------------------
  pressure_deficit    0.2722       |############################
  central_pres        0.2416       |########################
  day_of_year         0.1584       |################
  lat                 0.1001       |##########
  coriolis_f          0.0955       |##########
  lon                 0.0686       |#######
  bearing_sin         0.0195       |##
  bearing_cos         0.0115       |#
  forward_speed       0.0103       |#
  dist_km             0.0096       |#
  month_cos           0.0070       |#
  month_sin           0.0039       |
  is_bob              0.0008       |
  dt_hours            0.0008       |
```

**Physical Interpretation**:
Thermodynamic central pressure deficit ($\Delta P$) and central pressure ($P_{\min}$) account for over **51.3%** of the total predictive power. Climatological seasonality (`day_of_year`) and planetary coordinates (`lat`, `coriolis_f`, `lon`) account for **42.3%**, consistent with North Indian Ocean cyclogenesis climatology.

---

## 5. Quantitative Benchmark Results

### Dataset Splits (Strictly Split by Storm SID — 0% Leakage)
* **Training Set**: 10,557 observations across 323 historical cyclones (25.2% CS positive)
* **Validation Set**: 2,321 observations across 69 historical cyclones (28.6% CS positive)
* **Unseen Test Set**: 2,223 observations across 70 completely unseen cyclones (21.0% CS positive)

### Detailed Quantitative Comparison

| Metric | Training Set | Validation Set | Unseen Test Set (Held-Out) |
| :--- | :---: | :---: | :---: |
| **Sample Count** | 10,557 | 2,321 | 2,223 |
| **Unique Storms** | 323 | 69 | 70 |
| **ROC-AUC** | **0.9979** | **0.9494** | **0.9319** |
| **Brier Score** | **0.0219** | **0.0681** | **0.0683** |
| **Standard Accuracy ($p \ge 0.50$)** | 98.21% | 91.04% | **91.09%** |
| **Standard Precision** | 0.9743 | 0.8862 | **0.8059** |
| **Standard Recall** | 0.9541 | 0.7873 | **0.7575** |
| **Standard F1-Score** | 0.9641 | 0.8339 | **0.7810** |
| **Operational Accuracy ($p \ge 0.35$)** | 97.46% | 91.34% | **89.61%** |
| **Operational Recall** | 0.9876 | 0.8567 | **0.8026** |
| **Operational F1-Score** | 0.9502 | 0.8497 | **0.7640** |

### Unseen Test Confusion Matrices

#### Standard Threshold ($p \ge 0.50$)
```
                   Predicted Non-Cyclone    Predicted Cyclonic Storm
Actual Non-Cyclone        1,672 (TN)                  85 (FP)
Actual Cyclonic Storm       113 (FN)                 353 (TP)
```
* Specificity (True Negative Rate): **95.16%**
* Precision: **80.59%**
* Overall Accuracy: **91.09%**

#### Operational Disaster Alert Threshold ($p \ge 0.35$)
```
                   Predicted Non-Cyclone    Predicted Cyclonic Storm
Actual Non-Cyclone        1,618 (TN)                 139 (FP)
Actual Cyclonic Storm        92 (FN)                 374 (TP)
```
* Sensitivity (Recall): **80.26%** (captures 374 of 466 cyclone points; cuts false negatives by 18.6%)

---

## 6. Satellite Convective Pattern Analysis

In addition to kinematic and thermodynamic track observations, the system incorporates the **Satellite Convective Pattern Analyzer** ([`ml/models/satellite_detector.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/ml/models/satellite_detector.py)):
* **Input**: 2D calibrated infrared brightness temperature arrays ($T_b \in [170\text{ K}, 320\text{ K}]$).
* **Deep Convection Fraction**: Calculates percentage of pixels with $T_b \le 210.0\text{ K}$ ($-63.15^\circ\text{C}$, the WMO benchmark for vigorous tropical updrafts).
* **Azimuthal Symmetry Score**: Computes standard deviation across 8 angular octants to quantify axisymmetric eyewall curvature.
* **Eye / Warm Core Detector**: Evaluates radial contrast $\Delta T = T_{\text{center}} - T_{\text{ring}}$ to detect central eye formation within cold convective canopies.
* **Composite Organization Score**: Blends deep convection coverage, cloud-top temperatures, and azimuthal symmetry into an overall organization metric $[0.0, 1.0]$.

---

## 7. Backend API Integration

The trained model is directly wired into the FastAPI backend service:
* **Route**: `POST /api/satellite/analyze`
* **Service**: [`backend/app/services/satellite_service.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/backend/app/services/satellite_service.py)
* **Schemas**: [`backend/app/schemas/satellite.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/backend/app/schemas/satellite.py)

### Sample Request
```json
{
  "sensor": "INSAT-3D",
  "channel": "TIR-1",
  "lat": 16.5,
  "lon": 86.2,
  "central_pres": 975.0,
  "forward_speed": 18.0,
  "bearing": 320.0,
  "month": 10,
  "subbasin": "BB"
}
```

### Live Model Response
```json
{
  "status": "success",
  "service": "Satellite Analysis & Cyclone Identification Service",
  "model_status": "active",
  "model_connected": true,
  "model_version": "1.0.0",
  "is_cyclone": true,
  "confidence": 0.8924,
  "probability_cs": 0.8924,
  "risk_level": "Severe",
  "diagnosis": "Organized Cyclonic Storm identified (P = 89.2%). Sustained winds >= 34 kt indicated with severe coastal threat.",
  "detection_result": "CYCLONIC_STORM",
  "top_contributing_features": [
    {"feature": "pressure_deficit", "value": 35.0, "importance_weight": 0.2722},
    {"feature": "central_pres", "value": 975.0, "importance_weight": 0.2416},
    {"feature": "day_of_year", "value": 0.794, "importance_weight": 0.1584},
    {"feature": "lat", "value": 16.5, "importance_weight": 0.1001},
    {"feature": "coriolis_f", "value": 0.414, "importance_weight": 0.0955},
    {"feature": "lon", "value": 86.2, "importance_weight": 0.0686}
  ],
  "gradcam_generated": false,
  "message": "Identification inference executed successfully using trained Phase 4 model checkpoint."
}
```

---

## 8. Reproducibility & Model Artifacts

All training artifacts, configurations, and test reports are preserved:
1. **Model Checkpoint**: `ml/models/saved/cyclone_detector.joblib` (489 KB, MD5 verified)
2. **Metadata & Hyperparameters**: `ml/models/saved/detector_metadata.json`
3. **Independent Test Evaluation Report**: `docs/models/evaluation_reports/detector_test_report.json`
4. **Master Training CLI**: `python ml/training/train_detector.py`
5. **Independent Evaluation CLI**: `python ml/evaluation/eval_detector.py`
6. **Automated QA Test Suite**: `pytest tests/test_identification.py -v` (7 passed in 7.05s)
7. **Full System Test Suite**: `pytest backend/tests tests -v` (13 passed in 7.24s)
