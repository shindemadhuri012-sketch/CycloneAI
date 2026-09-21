"""
Metrics and evaluation audit route.
Exposes machine-readable verified SIH benchmark reports for ModelPerformancePage.
"""
import os
import json
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)


def find_project_root() -> str:
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr and os.path.dirname(curr) != curr:
        if os.path.exists(os.path.join(curr, "PROJECT_STATUS.md")):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))


PROJECT_ROOT = find_project_root()
REPORTS_DIR = os.path.join(PROJECT_ROOT, "docs/models/evaluation_reports")

router = APIRouter(prefix="/system", tags=["System & Metrics"])


@router.get("/metrics", summary="Get Evaluated Scientific Benchmarks Across All Phases")
async def get_system_metrics():
    """
    Returns the real, verified scientific evaluation test results across all 4 AI models and XAI engine.
    Read directly from official evaluation reports. Zero synthetic metrics.
    """
    metrics = {
        "status": "success",
        "phase": "Phase 9 - Full System Integration",
        "models": {
            "cyclone_detector": None,
            "cyclone_classifier": None,
            "cyclone_intensity": None,
            "cyclone_track": None,
            "explainable_ai": None,
        }
    }

    report_map = {
        "cyclone_detector": "detector_test_report.json",
        "cyclone_classifier": "classifier_test_report.json",
        "cyclone_intensity": "intensity_test_report.json",
        "cyclone_track": "track_test_report.json",
        "explainable_ai": "xai_test_report.json",
    }

    for key, filename in report_map.items():
        filepath = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    metrics["models"][key] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read report {filepath}: {e}")
                metrics["models"][key] = {"error": f"Failed to parse report: {str(e)}"}
        else:
            metrics["models"][key] = {"status": f"Report file not found at {filepath}"}

    return metrics
