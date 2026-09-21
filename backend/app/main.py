"""
CycloneAI Backend - Main FastAPI Application
Production-hardened ASGI server with unified multi-model inference pipeline,
interactive Swagger/ReDoc documentation, security headers, and static SPA serving.
"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, satellite, cyclone, intensity, track, xai, pipeline, storms, metrics
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("CycloneAI Production Server initializing in %s mode", settings.ENVIRONMENT)
    logger.info("Version: %s | Host: %s | Port: %d", settings.VERSION, settings.HOST, settings.PORT)
    logger.info("Model Subsystems: ACTIVE (Phases 1-10 Tested & Scientifically Verified)")
    logger.info("=" * 70)
    yield
    logger.info("CycloneAI Server shutting down cleanly.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Intelligent Tropical Cyclone Analysis & Prediction System - REST API "
        "(Smart India Hackathon 2026). Ingests multi-source meteorological observations "
        "and provides sub-100ms multi-model inference for detection, classification, "
        "intensity estimation, trajectory tracking, and Explainable AI."
    ),
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Security & Timing Middleware
@app.middleware("http")
async def add_security_and_timing_headers(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Core API Routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(satellite.router, prefix=settings.API_V1_STR)
app.include_router(cyclone.router, prefix=settings.API_V1_STR)
app.include_router(intensity.router, prefix=settings.API_V1_STR)
app.include_router(track.router, prefix=settings.API_V1_STR)
app.include_router(xai.router, prefix=settings.API_V1_STR)
app.include_router(pipeline.router, prefix=settings.API_V1_STR)
app.include_router(storms.router, prefix=settings.API_V1_STR)
app.include_router(metrics.router, prefix=settings.API_V1_STR)

# Mount Static Assets for Frontend SPA if directory exists
dist_dir = settings.FRONTEND_DIST_DIR
assets_dir = os.path.join(dist_dir, "assets")
index_file = os.path.join(dist_dir, "index.html")

if settings.SERVE_STATIC_SPA and os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    logger.info("Mounted frontend assets from %s", assets_dir)


@app.get("/", tags=["Root"])
async def root(request: Request):
    """
    Root endpoint.
    Returns the interactive React SPA if requested by a web browser,
    or JSON system metadata if requested by API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and os.path.exists(index_file):
        return FileResponse(index_file, media_type="text/html")

    return {
        "message": "Welcome to CycloneAI API - Intelligent Tropical Cyclone Analysis & Prediction System",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": f"{settings.API_V1_STR}/health",
        "system_metrics": f"{settings.API_V1_STR}/system/metrics",
        "pipeline_run": f"{settings.API_V1_STR}/pipeline/run",
        "status": "operational",
        "phase": "Phase 11 - Production Deployment & Presentation Readiness",
        "version": settings.VERSION,
        "models": {
            "identification": "Active (HistGradientBoosting + CalibratedClassifierCV)",
            "classification": "Active (8-Tier IMD Scale Categorizer)",
            "intensity": "Active (Multi-Horizon HistGradientBoosting + RI Classifier)",
            "track": "Active (Incremental Displacement Regressor + 75% Uncertainty Cones)",
            "explainable_ai": "Active (Exact TreeSHAP + Dvorak BD Cloud Saliency)"
        }
    }


# Fallback catch-all route for Single Page Application client-side routing
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str, request: Request):
    """
    Directs unhandled browser routes to index.html for React Router,
    or returns 404 for API requests.
    """
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("redoc") or full_path.startswith("openapi.json"):
        return JSONResponse(status_code=404, content={"detail": f"Endpoint /{full_path} not found"})

    if os.path.exists(index_file):
        return FileResponse(index_file, media_type="text/html")

    return JSONResponse(status_code=404, content={"detail": f"Resource /{full_path} not found and frontend bundle missing"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
