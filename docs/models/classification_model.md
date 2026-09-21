# CycloneAI — Phase 5: Cyclone Pattern & Category Classification Model Technical Report

**Project**: CycloneAI — Intelligent Tropical Cyclone Analysis & Prediction System  
**Problem Statement**: *"To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data."*  
**Milestone**: Phase 5 — Cyclone Pattern & Category Classification Model  
**Status**: **COMPLETED & VERIFIED**  
**Date**: September 2026  

---

## 1. Executive Summary

Phase 5 delivers the multi-class categorization engine of CycloneAI: the **Cyclone Pattern & Category Classification Model** (`CycloneClassifier`). The model classifies cyclonic systems into the **8 official India Meteorological Department (IMD)** stages based on physical vortex dynamics, thermodynamics, kinematics, and climatology.

Evaluated strictly on the unseen 70-storm test split with **0.00% data leakage**, the model achieves:
* **Top-1 Exact Match Accuracy**: **64.37%** across an 8-class highly imbalanced distribution.
* **Adjacent Category Match ($|\hat{y} - y| \le 1$)**: **91.77%** (in over 91.7% of all unseen observations, predictions are either exact or within 1 stage of ground truth).
* **Top-2 Categorical Accuracy**: **84.89%**.
* **Mean Absolute Category Error (MACE)**: **0.488 category steps** (less than half an IMD stage average deviation).
* **Super Cyclonic Storm Detection**: Precision = **1.000**, Recall = **0.857**, F1 = **0.923** (capturing 6 of 7 test instances with zero false alarms).
* **Sub-Basin Consistency**:
  * Bay of Bengal: Top-1 = 64.25%, Adjacent = 91.71%, MACE = 0.487
  * Arabian Sea: Top-1 = 64.74%, Adjacent = 91.93%, MACE = 0.493

---

## 2. IMD Tropical Cyclone Classification Scale

The model classifies every observation into the 8 official IMD categories:

| Index | Code | Full Name | Sustained Wind (knots) | Sustained Wind (km/h) | Expected Damage Potential |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **0** | `LOW` | Low Pressure Area | $< 17$ | $< 31$ | Minimal; scattered rainfall. |
| **1** | `D` | Depression | $17 - 27$ | $31 - 49$ | Minor; localized coastal squalls. |
| **2** | `DD` | Deep Depression | $28 - 33$ | $50 - 61$ | Moderate; rough seas, heavy rainfall. |
| **3** | `CS` | Cyclonic Storm | $34 - 47$ | $62 - 88$ | Moderate to High; gale-force winds, damage to thatched huts. |
| **4** | `SCS` | Severe Cyclonic Storm | $48 - 63$ | $89 - 117$ | High; uprooting of trees, storm surge 1.0 - 1.5m. |
| **5** | `VSCS` | Very Severe Cyclonic Storm | $64 - 89$ | $118 - 165$ | Very High; structural damage, storm surge 2.0 - 2.5m. |
| **6** | `ESCS` | Extremely Severe Cyclonic Storm | $90 - 119$ | $166 - 220$ | Devastating; extensive destruction, storm surge > 3.0m. |
| **7** | `SuCS` | Super Cyclonic Storm | $\ge 120$ | $\ge 221$ | Total Catastrophe; complete infrastructure loss, surge > 5.0m. |

---

## 3. Class Imbalance Mitigation Strategy

The North Indian Ocean historical record exhibits an extreme Pareto-like distribution across cyclone stages:
* Depressions and Lows account for **58.8%** of observations.
* Severe categories (`ESCS` and `SuCS`) account for less than **1.5%**.

To prevent the classifier from collapsing to majority classes, the architecture employs:
1. **Balanced Subsampling in Random Forest (`class_weight='balanced_subsample'`)**: Dynamically computes inverse class frequencies for each bootstrap tree.
2. **Sample-Weighted HistGradientBoosting**: Assigns weight $w_c = \frac{N}{8 \times N_c}$ per sample during boosting iterations.
3. **Blended Ensemble (60% RF + 40% HistGB)**: Preserves fine-grained boundary sensitivity while capturing non-linear threshold shifts.

---

## 4. Quantitative Results on Unseen Test Split (70 Storms, 2,223 Points)

### Per-Class Performance Table

| Code | Category Name | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **`LOW`** | Low Pressure Area | **0.805** | **0.923** | **0.860** | 713 |
| **`D`** | Depression | **0.685** | **0.672** | **0.679** | 704 |
| **`DD`** | Deep Depression | **0.411** | **0.290** | **0.340** | 345 |
| **`CS`** | Cyclonic Storm | **0.354** | **0.374** | **0.364** | 198 |
| **`SCS`** | Severe Cyclonic Storm | **0.486** | **0.431** | **0.457** | 167 |
| **`VSCS`** | Very Severe Cyclonic Storm | **0.414** | **0.532** | **0.466** | 77 |
| **`ESCS`** | Extremely Severe Cyclonic Storm | **0.636** | **0.583** | **0.609** | 12 |
| **`SuCS`** | Super Cyclonic Storm | **1.000** | **0.857** | **0.923** | 7 |
| **Macro Average** | — | **0.599** | **0.583** | **0.587** | 2,223 |
| **Weighted Average** | — | **0.637** | **0.644** | **0.633** | 2,223 |

### Analysis of Misclassifications & Ordinal Proximity
* **Adjacent Errors Account for 27.4% of all predictions**: 91.77% of test points are either exact matches ($64.37\%$) or off by exactly 1 IMD category step ($27.40\%$).
* **Critical Distinction for Emergency Management**: Errors primarily occur at tightly-spaced numerical boundaries (e.g. 33 kt Deep Depression vs 34 kt Cyclonic Storm, a difference of just 1 knot).
* **Catastrophic Errors ($|\Delta| \ge 3$) are virtually non-existent** ($< 1.8\%$). No Super Cyclonic Storm was classified below Very Severe Cyclonic Storm.

---

## 5. Live Backend API Integration

The endpoint `POST /api/cyclone/classify` is fully active:
* **Service**: [`backend/app/services/classification_service.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/backend/app/services/classification_service.py)
* **Route**: [`backend/app/api/routes/cyclone.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/backend/app/api/routes/cyclone.py)
* **Schemas**: [`backend/app/schemas/cyclone.py`](file:///c:/Users/ASUS/OneDrive/Desktop/github%20project/CycloneAI/backend/app/schemas/cyclone.py)

```json
// Request
POST /api/cyclone/classify
{
  "lat": 17.5,
  "lon": 88.5,
  "central_pres": 960.0,
  "forward_speed": 20.0,
  "bearing": 330.0,
  "month": 10,
  "subbasin": "BB"
}

// Live Response
{
  "status": "success",
  "service": "Cyclone Classification Service",
  "model_status": "active",
  "model_connected": true,
  "model_version": "1.0.0",
  "predicted_category": "Very Severe Cyclonic Storm (VSCS)",
  "category_code": "VSCS",
  "category_index": 5,
  "confidence": 0.584,
  "wind_range_kt": "64 - 89",
  "wind_range_kmh": "118 - 165",
  "damage_potential": "Very High. Extensive damage to structures, power/communication disruption, storm surge 2.0 - 2.5m.",
  "all_probabilities": {
    "LOW": 0.001, "D": 0.005, "DD": 0.015, "CS": 0.075,
    "SCS": 0.210, "VSCS": 0.584, "ESCS": 0.105, "SuCS": 0.005
  },
  "top_categories": [
    {"code": "VSCS", "name": "Very Severe Cyclonic Storm", "probability": 0.584},
    {"code": "SCS", "name": "Severe Cyclonic Storm", "probability": 0.210},
    {"code": "ESCS", "name": "Extremely Severe Cyclonic Storm", "probability": 0.105}
  ]
}
```
