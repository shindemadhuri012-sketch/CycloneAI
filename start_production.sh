#!/usr/bin/env bash
# ====================================================================
# CycloneAI — Linux / macOS Production Server Launcher
# Smart India Hackathon 2026
# Launches unified FastAPI application on http://localhost:8000
# ====================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "===================================================================="
echo "Starting CycloneAI Production Server (SIH 2026 Prototype)"
echo "===================================================================="

if [ -f "backend/.venv/bin/python" ]; then
    PYTHON_EXEC="backend/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
else
    PYTHON_EXEC="python"
fi

echo "[INFO] Using Python interpreter: $PYTHON_EXEC"
$PYTHON_EXEC run_production.py
