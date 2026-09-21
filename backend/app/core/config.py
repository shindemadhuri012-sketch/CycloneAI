"""
Core configuration settings for the CycloneAI Backend.
Supports both decoupled development and unified single-container production deployment.
"""
import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))


class Settings(BaseSettings):
    PROJECT_NAME: str = "CycloneAI Backend"
    PROJECT_TITLE: str = "Intelligent Tropical Cyclone Analysis & Prediction System"
    API_V1_STR: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "production"
    VERSION: str = "1.0.0"

    # Frontend single-page app static distribution path
    SERVE_STATIC_SPA: bool = True
    FRONTEND_DIST_DIR: str = os.path.join(PROJECT_ROOT, "frontend/dist")

    # Checkpoint and data paths
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "ml/models/saved")
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data/processed")

    # CORS Origins allowed to communicate with this backend
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]

    model_config = SettingsConfigDict(
        env_file=os.path.join(BACKEND_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
