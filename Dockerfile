# ====================================================================
# CycloneAI - Production Multi-Stage Dockerfile
# Stage 1: Build React 19 Frontend SPA
# Stage 2: Production Python FastAPI ASGI Server
# ====================================================================

# Stage 1: Frontend Build
FROM node:22-alpine AS frontend-builder
WORKDIR /build

# Install dependencies with lockfile caching
COPY frontend/package*.json ./
RUN npm ci --silent

# Copy source code and build production distribution
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Server
FROM python:3.11-slim AS production-runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    ENVIRONMENT=production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python backend dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend application code
COPY backend/ ./backend/
COPY PROJECT_STATUS.md ./PROJECT_STATUS.md
COPY run_production.py ./run_production.py

# Copy ML model artifacts and verified processed datasets
COPY ml/ ./ml/
COPY data/processed/ ./data/processed/
COPY docs/models/evaluation_reports/ ./docs/models/evaluation_reports/

# Copy compiled frontend SPA bundle from Stage 1
COPY --from=frontend-builder /build/dist ./frontend/dist

# Expose default production port (Render/Railway dynamically inject $PORT)
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Launch production server via Uvicorn with dynamic platform port binding
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --no-access-log"]

