/**
 * CycloneAI Frontend API Service Layer
 * Centralized client for communicating with FastAPI Backend.
 * Strictly operates on real-world meteorological data and verified ML models.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

/**
 * Checks overall backend and ML model subsystem health.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (error) {
    return {
      status: 'offline',
      service: 'CycloneAI Backend',
      model_status: 'unavailable',
      active_models: {},
      error: error.message,
    };
  }
}

/**
 * Unified multi-model pipeline execution.
 * Orchestrates Identification, Classification, Intensity, Track, and XAI.
 */
export async function runUnifiedPipeline(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/pipeline/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Pipeline execution failed: ${error.message}`,
      model_connected: false,
    };
  }
}

/**
 * Retrieves list of real North Indian Ocean cyclones.
 */
export async function fetchStormCatalog(params = {}) {
  try {
    const query = new URLSearchParams();
    if (params.year && params.year !== 'All') query.append('year', params.year);
    if (params.basin && params.basin !== 'All') query.append('basin', params.basin);
    if (params.search) query.append('search', params.search);
    if (params.limit) query.append('limit', params.limit);

    const res = await fetch(`${API_BASE}/storms/catalog?${query.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      total_storms: 0,
      storms: [],
      error: error.message,
    };
  }
}

/**
 * Retrieves 1-click famous benchmark presets (Fani, Amphan, Biparjoy, Mocha, etc.)
 */
export async function fetchStormPresets() {
  try {
    const res = await fetch(`${API_BASE}/storms/presets`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    return [];
  }
}

/**
 * Retrieves all chronological observation points for a specific storm.
 */
export async function fetchStormPoints(sid) {
  try {
    const res = await fetch(`${API_BASE}/storms/${sid}/points`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    return { status: 'error', points: [], error: error.message };
  }
}

/**
 * Retrieves verified test evaluation reports across all phases from disk.
 */
export async function fetchSystemMetrics() {
  try {
    const res = await fetch(`${API_BASE}/system/metrics`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    return { status: 'error', models: {}, error: error.message };
  }
}

/**
 * Satellite Analysis & Cyclone Identification (Phase 4).
 */
export async function analyzeSatellite(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/satellite/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to connect to satellite analyzer: ${error.message}`,
      model_connected: false,
    };
  }
}

/**
 * Cyclone Pattern & Category Classification (Phase 5).
 */
export async function classifyCyclone(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/cyclone/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to classify cyclone: ${error.message}`,
      model_connected: false,
    };
  }
}

/**
 * Cyclone Intensity Prediction (Phase 6).
 */
export async function predictIntensity(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/intensity/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to predict intensity: ${error.message}`,
      model_connected: false,
    };
  }
}

/**
 * Cyclone Track Trajectory Prediction (Phase 7).
 */
export async function predictTrack(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/track/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to predict track: ${error.message}`,
      model_connected: false,
    };
  }
}

/**
 * Explainable AI - Local Tree-Path Feature Attribution (Phase 8).
 */
export async function explainPrediction(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/xai/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to explain prediction: ${error.message}`,
      model_connected: false,
    };
  }
}

/**
 * Explainable AI - Dvorak BD Satellite Thermal Saliency (Phase 8).
 */
export async function fetchSaliency(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/xai/saliency`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to compute saliency: ${error.message}`,
    };
  }
}

/**
 * Explainable AI - 1D/2D Sensitivity and Counterfactuals (Phase 8).
 */
export async function fetchSensitivity(payload = {}) {
  try {
    const res = await fetch(`${API_BASE}/xai/sensitivity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      message: `Failed to compute sensitivity: ${error.message}`,
    };
  }
}
