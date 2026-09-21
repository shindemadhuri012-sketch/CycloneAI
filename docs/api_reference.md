# CycloneAI: REST API Reference Documentation
**Smart India Hackathon 2026**  
**API Version:** 1.0.0  
**Base URL:** `http://localhost:8000/api`  
**Interactive Swagger UI:** `http://localhost:8000/docs`  
**ReDoc Specification:** `http://localhost:8000/redoc`  

---

## 1. Overview & Conventions

All endpoints accept and return `application/json` unless otherwise specified.
Standard response format includes status markers, execution timestamps, and typed fields.

### HTTP Status Codes:
- `200 OK`: Request succeeded.
- `400 Bad Request`: Parameter boundary violation or physical constraint mismatch.
- `422 Unprocessable Entity`: Input payload failed schema validation (e.g. out-of-range coordinates).
- `500 Internal Server Error`: Unhandled server exception (gracefully shielded).

---

## 2. Core System Endpoints

### 2.1 Health Probe
`GET /api/health`

Returns dynamic status of all five AI model subsystems.

**Sample Response**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "production",
  "model_status": "active",
  "active_models": [
    "cyclone_detector",
    "cyclone_classifier",
    "cyclone_intensity",
    "cyclone_track",
    "explainable_ai"
  ]
}
```

---

### 2.2 System Metrics & Benchmark Report
`GET /api/system/metrics`

Returns official verified evaluation metrics from the Phase 10 independent test suite (70 held-out storms).

**Sample Response**:
```json
{
  "status": "success",
  "timestamp": "2026-09-20T16:30:40Z",
  "evaluation_scope": "North Indian Ocean Held-Out Test Split (70 Storms, 2,223 Points)",
  "leakage_verification": {
    "test_storms": 70,
    "leakage_count": 0,
    "leakage_percentage": 0.0
  },
  "models": {
    "cyclone_detector": {
      "roc_auc": 0.9615,
      "accuracy_pct": 93.66
    },
    "cyclone_classifier": {
      "adjacent_accuracy_pct": 79.85,
      "mace": 1.14
    },
    "cyclone_intensity": {
      "+24h_wind_mae_kt": 9.59,
      "wind_skill_gain_pct": 8.1,
      "rapid_intensification_roc_auc": 0.7304
    },
    "cyclone_track": {
      "+24h_ate_km": 149.92,
      "+48h_ate_km": 332.75,
      "skill_gain_vs_persistence_48h_pct": 20.8
    }
  }
}
```

---

## 3. Storm Catalog & Historical Data Endpoints

### 3.1 1-Click Operational Presets
`GET /api/storms/presets`

Returns famous benchmark historical cyclones pre-formatted for 1-click dashboard testing.

**Sample Response**:
```json
[
  {
    "sid": "2019117N03088",
    "name": "FANI",
    "season": 2019,
    "subbasin": "BB",
    "peak_grade": "ESCS",
    "peak_wind_kt": 115.0,
    "observation": {
      "lat": 16.0,
      "lon": 84.5,
      "central_pres": 932.0,
      "current_wind_speed_knots": 115.0,
      "forward_speed": 17.0,
      "bearing": 325.0,
      "month": 5,
      "subbasin": "BB"
    }
  }
]
```

---

### 3.2 Storm Catalog Search
`GET /api/storms/catalog?subbasin=BB&min_grade=VSCS&limit=10`

Search and filter through all 1,858 historical cyclones in the North Indian Ocean archive.

---

## 4. Unified Analytical Pipeline Endpoint

### 4.1 Run Unified Analysis Pipeline
`POST /api/pipeline/run`

Executes all five analytical capabilities (Identification, Classification, Intensity Forecasting, Track Trajectory Prediction, and Explainable AI) in a single synchronized call with sub-100ms execution latency.

**Sample Request**:
```json
{
  "storm_id": "FANI",
  "lat": 16.0,
  "lon": 84.5,
  "central_pres": 932.0,
  "current_wind_speed_knots": 115.0,
  "forward_speed": 17.0,
  "bearing": 325.0,
  "month": 5,
  "subbasin": "BB",
  "past_track": [
    {"latitude": 14.8, "longitude": 85.2, "intensity_knots": 105.0}
  ]
}
```

**Sample Response**:
```json
{
  "status": "success",
  "storm_id": "FANI",
  "timestamp": "2026-09-21T18:40:00Z",
  "execution_time_ms": 68.4,
  "identification": {
    "is_cyclone": true,
    "confidence": 0.994,
    "probability_cs": 0.994,
    "risk_level": "Critical"
  },
  "classification": {
    "predicted_category": "Extremely Severe Cyclonic Storm (ESCS)",
    "category_code": "ESCS",
    "confidence": 0.912,
    "all_probabilities": {
      "ESCS": 0.912,
      "VSCS": 0.071,
      "SuCS": 0.017
    }
  },
  "intensity": {
    "forecast_horizons": {
      "+6h": {"wind_knots": 118.2, "pressure_hpa": 928.4},
      "+12h": {"wind_knots": 120.5, "pressure_hpa": 925.1},
      "+24h": {
        "wind_knots": 122.0,
        "pressure_hpa": 922.0,
        "wind_bounds_90": {"low_knots": 108.0, "high_knots": 135.0}
      }
    },
    "rapid_intensification": {
      "is_alert_active": true,
      "probability": 0.84,
      "risk_level": "Critical (Rapid Intensification Detected)"
    }
  },
  "track": {
    "predicted_track": [
      {"lead_time": "+6h", "latitude": 16.8, "longitude": 84.8, "forward_speed_kmh": 17.2, "bearing_deg": 328.0},
      {"lead_time": "+12h", "latitude": 17.6, "longitude": 85.1, "forward_speed_kmh": 17.5, "bearing_deg": 332.0},
      {"lead_time": "+24h", "latitude": 19.1, "longitude": 85.6, "forward_speed_kmh": 18.0, "bearing_deg": 338.0},
      {"lead_time": "+48h", "latitude": 21.5, "longitude": 86.8, "forward_speed_kmh": 19.2, "bearing_deg": 348.0}
    ],
    "track_error_cone": [
      {"lead_time": "+6h", "radius_km": 45.0, "center": [16.8, 84.8]},
      {"lead_time": "+12h", "radius_km": 80.0, "center": [17.6, 85.1]},
      {"lead_time": "+24h", "radius_km": 140.0, "center": [19.1, 85.6]},
      {"lead_time": "+48h", "radius_km": 240.0, "center": [21.5, 86.8]}
    ]
  },
  "explainability": {
    "top_drivers": [
      {"feature": "pressure_deficit", "contribution": 18.4, "description": "Barometric core deficit"},
      {"feature": "coriolis_f", "contribution": 6.2, "description": "Planetary vorticity"}
    ],
    "axisymmetry_score": 0.88,
    "synoptic_briefing": "High-intensity mature tropical vortex exhibiting well-developed eye structure and extreme pressure deficit."
  }
}
```

---

## 5. Dedicated Modular Service Endpoints

- `POST /api/satellite/analyze`: Simulates convective cloud structure analysis.
- `POST /api/cyclone/classify`: Returns 8-tier category probability vector.
- `POST /api/intensity/predict`: Multi-horizon wind/pressure regression with quantile ribbons.
- `POST /api/track/predict`: Trajectory waypoint projection and GeoJSON uncertainty polygons.
- `POST /api/xai/explain`: TreeSHAP waterfall and natural language synoptic briefing.
- `POST /api/xai/saliency`: 2D Dvorak BD convective cloud ring saliency matrix.
- `POST /api/xai/sensitivity`: 1D/2D partial dependence sensitivity curves across physical ranges.
