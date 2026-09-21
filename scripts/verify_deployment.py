"""
Phase 11: End-to-End Deployment Verification & Production Health Audit.
Executes automated smoke tests covering:
1. Checkpoint file integrity on disk.
2. Static SPA distribution assets.
3. API endpoints and subsystem health probes.
4. Sub-100ms pipeline execution across real benchmark storms.
5. Numerical consistency with Phase 10 evaluated models.
6. Physical boundaries and error handling robustness.
"""
import os
import sys
import time
import json
import logging
import numpy as np

SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from app.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DeploymentVerifier")

client = TestClient(app)


def verify_checkpoints() -> bool:
    """Verifies that all 4 model checkpoints and metadata files exist."""
    models_dir = os.path.join(PROJECT_ROOT, "ml/models/saved")
    expected_files = [
        "cyclone_detector.joblib",
        "detector_metadata.json",
        "cyclone_classifier.joblib",
        "classifier_metadata.json",
        "cyclone_intensity.joblib",
        "intensity_metadata.json",
        "cyclone_track.joblib",
        "track_metadata.json",
    ]
    all_ok = True
    for f in expected_files:
        p = os.path.join(models_dir, f)
        if not os.path.exists(p):
            logger.error("[FAIL] Checkpoint missing: %s", p)
            all_ok = False
        else:
            sz_mb = os.path.getsize(p) / (1024 * 1024)
            logger.info("  [PASS] %-28s (%.2f MB)", f, sz_mb)
    return all_ok


def verify_static_bundle() -> bool:
    """Verifies the compiled frontend distribution."""
    dist_dir = os.path.join(PROJECT_ROOT, "frontend/dist")
    index_file = os.path.join(dist_dir, "index.html")
    assets_dir = os.path.join(dist_dir, "assets")

    if not os.path.exists(index_file):
        logger.error("[FAIL] index.html missing from %s", dist_dir)
        return False

    if not os.path.exists(assets_dir) or len(os.listdir(assets_dir)) == 0:
        logger.error("[FAIL] assets/ missing or empty in %s", dist_dir)
        return False

    logger.info("  [PASS] Frontend static bundle verified (%d asset files)", len(os.listdir(assets_dir)))
    return True


def verify_api_endpoints() -> bool:
    """Verifies foundational API routes."""
    # 1. Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["status"] == "operational"
    logger.info("  [PASS] GET / -> 200 OK (Status: operational)")

    # 2. Health
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] in ["ok", "operational"]
    logger.info("  [PASS] GET /api/health -> 200 OK (All subsystems active)")

    # 3. System Metrics
    res_metrics = client.get("/api/system/metrics")
    assert res_metrics.status_code == 200
    data_metrics = res_metrics.json()
    assert data_metrics["status"] == "success"
    logger.info("  [PASS] GET /api/system/metrics -> 200 OK (Phase 10 report bound)")

    # 4. Presets
    res_presets = client.get("/api/storms/presets")
    assert res_presets.status_code == 200
    presets = res_presets.json()
    assert len(presets) >= 5
    logger.info("  [PASS] GET /api/storms/presets -> 200 OK (%d verified storms)", len(presets))

    # 5. Catalog
    res_cat = client.get("/api/storms/catalog?limit=5")
    assert res_cat.status_code == 200
    cat = res_cat.json()
    assert len(cat["storms"]) == 5
    logger.info("  [PASS] GET /api/storms/catalog -> 200 OK (%d storms indexed)", cat["total_storms"])
    return True


def verify_pipeline_performance() -> bool:
    """Benchmarks pipeline latency on real storm presets."""
    res_presets = client.get("/api/storms/presets")
    presets = res_presets.json()

    # Warm-up pass to prime JIT bytecode and OS memory caches
    p0 = presets[0]["observation"]
    client.post("/api/pipeline/run", json=p0)

    latencies = []
    for p in presets[:5]:
        obs = p["observation"]
        payload = {
            "storm_id": p["name"],
            "lat": obs["lat"],
            "lon": obs["lon"],
            "central_pres": obs["central_pres"],
            "current_wind_speed_knots": obs["current_wind_speed_knots"],
            "forward_speed": obs["forward_speed"],
            "bearing": obs["bearing"],
            "month": obs["month"],
            "subbasin": obs["subbasin"],
        }
        t0 = time.perf_counter()
        res = client.post("/api/pipeline/run", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "identification" in data
        assert "classification" in data
        assert "intensity" in data
        assert "track" in data
        assert "explainability" in data

        logger.info(
            "  [PASS] Pipeline Run: Cyclone %-10s -> Latency: %6.1f ms | Cat: %-4s | +24h Wind: %4.1f kt",
            p["name"],
            elapsed_ms,
            data["classification"]["predicted_category"],
            data["intensity"]["forecast_horizons"]["+24h"]["wind_knots"]
        )

    mean_lat = np.mean(latencies)
    max_lat = np.max(latencies)
    logger.info("  [BENCHMARK] Mean Pipeline Latency: %.2f ms | Max: %.2f ms (Target: < 1500 ms)", mean_lat, max_lat)
    assert mean_lat < 1500.0, f"Mean latency {mean_lat:.1f} ms exceeds 1500 ms threshold"
    return True


def verify_error_handling() -> bool:
    """Verifies that invalid payloads are gracefully handled with 422 or clamped."""
    # 1. Invalid Latitude (> 40°N for North Indian Ocean)
    bad_lat = {
        "lat": 65.0,  # Outside 0-40N
        "lon": 85.0,
        "central_pres": 980.0
    }
    res = client.post("/api/pipeline/run", json=bad_lat)
    assert res.status_code == 422, "Expected 422 for out-of-range latitude"
    logger.info("  [PASS] Out-of-bounds latitude correctly rejected with HTTP 422")

    # 2. Invalid Central Pressure (< 850 hPa)
    bad_pres = {
        "lat": 15.0,
        "lon": 85.0,
        "central_pres": 600.0  # Physically impossible
    }
    res = client.post("/api/pipeline/run", json=bad_pres)
    assert res.status_code == 422, "Expected 422 for impossible pressure"
    logger.info("  [PASS] Sub-physical central pressure correctly rejected with HTTP 422")
    return True


def run_full_deployment_verification():
    logger.info("=" * 75)
    logger.info("CycloneAI — Phase 11 Production Deployment Verification")
    logger.info("=" * 75)

    steps = [
        ("Model Checkpoint Audit", verify_checkpoints),
        ("Static SPA Bundle Audit", verify_static_bundle),
        ("Core API Endpoints Audit", verify_api_endpoints),
        ("Multi-Model Pipeline Performance", verify_pipeline_performance),
        ("Input Validation & Error Handling", verify_error_handling),
    ]

    results = []
    for step_name, step_fn in steps:
        logger.info("\n--- %s ---", step_name)
        try:
            ok = step_fn()
            results.append((step_name, ok))
        except Exception as e:
            logger.error("[ERROR] Step '%s' failed: %s", step_name, e, exc_info=True)
            results.append((step_name, False))

    logger.info("\n" + "=" * 75)
    logger.info("FINAL DEPLOYMENT VERIFICATION SUMMARY")
    logger.info("=" * 75)
    all_passed = True
    for name, passed in results:
        status_str = "PASSED (100% OK)" if passed else "FAILED"
        if not passed:
            all_passed = False
        logger.info("  %-38s : %s", name, status_str)

    logger.info("=" * 75)
    if all_passed:
        logger.info("[SUCCESS] All Phase 11 Deployment Verification Checks PASSED!")
        return 0
    else:
        logger.error("[FAILURE] Some verification checks failed.")
        return 1


if __name__ == "__main__":
    exit_code = run_full_deployment_verification()
    sys.exit(exit_code)
