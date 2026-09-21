# ML Inference Module (`ml/inference/`)

## Purpose
This module provides optimized, production-grade inference pipelines invoked by the FastAPI backend services.

## Architecture
- `predictor.py`: Unified inference pipeline that accepts raw or preprocessed image buffers, runs forward passes, and formats typed prediction structures.
- Decoupled from training code to minimize runtime dependencies and maximize latency efficiency.
- Supports model export to ONNX runtime / TensorRT for accelerated deployment.

*Current Phase 1 Status: Stubbed. Backend services return explicit 'model not connected' responses.*
