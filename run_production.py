"""
CycloneAI — Single-Command Production Runner.
Launches the unified FastAPI application serving the interactive frontend SPA
and REST API on port 8000.
"""
import os
import sys
import subprocess
import logging

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
DIST_DIR = os.path.join(FRONTEND_DIR, "dist")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml/models/saved")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CycloneAI.Production")


def verify_prerequisites():
    """Verifies all required model checkpoints and datasets are present."""
    required_models = [
        "cyclone_detector.joblib",
        "cyclone_classifier.joblib",
        "cyclone_intensity.joblib",
        "cyclone_track.joblib",
    ]
    missing = [m for m in required_models if not os.path.exists(os.path.join(MODELS_DIR, m))]
    if missing:
        logger.error("Missing model checkpoints in %s: %s", MODELS_DIR, missing)
        sys.exit(1)

    logger.info("All 4 trained model checkpoints verified in %s", MODELS_DIR)

    # Check frontend bundle
    index_html = os.path.join(DIST_DIR, "index.html")
    if not os.path.exists(index_html):
        logger.warning("Frontend production bundle not found in %s.", DIST_DIR)
        logger.info("Attempting to build frontend with npm run build...")
        try:
            subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True, shell=True)
            logger.info("Frontend successfully built.")
        except Exception as e:
            logger.error("Failed to build frontend bundle: %s. Please run 'npm run build' in frontend/.", e)
    else:
        logger.info("Frontend production distribution verified in %s", DIST_DIR)


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Starts the Uvicorn ASGI production server."""
    import uvicorn
    from app.core.config import settings

    logger.info("=" * 70)
    logger.info("CycloneAI — Intelligent Tropical Cyclone Analysis & Prediction System")
    logger.info("Smart India Hackathon 2026 Production Server")
    logger.info("=" * 70)
    logger.info("Access Application Dashboard: http://localhost:%d", port)
    logger.info("Access Swagger API Docs:      http://localhost:%d/docs", port)
    logger.info("Access System Health Probe:    http://localhost:%d/api/health", port)
    logger.info("=" * 70)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        app_dir=BACKEND_DIR
    )


if __name__ == "__main__":
    verify_prerequisites()
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    start_server(host=host, port=port)
