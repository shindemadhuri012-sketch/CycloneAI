# System Architecture Documentation (`docs/architecture/`)

## Overview
CycloneAI is built with a decoupled 4-tier architecture designed for operational resilience, scalability, and scientific auditability:

```
+-----------------------------------------------------------------+
|                   Frontend Presentation Tier                    |
|             (React 19 + Vite + Tailwind CSS + Leaflet)          |
+-----------------------------------------------------------------+
                                │  REST / JSON
                                ▼
+-----------------------------------------------------------------+
|                    FastAPI Application Tier                     |
|           (Pydantic Schemas, CORS, Modular Services)            |
+-----------------------------------------------------------------+
                                │  Inference Hooks
                                ▼
+-----------------------------------------------------------------+
|                   ML Inference & Analytics Tier                 |
|             (PyTorch, Vision Backbones, Grad-CAM XAI)           |
+-----------------------------------------------------------------+
                                │  Data Pipeline
                                ▼
+-----------------------------------------------------------------+
|                     Data & Storage Tier                         |
|      (Raw Satellite Archives, Processed Tensors, IBTrACS)       |
+-----------------------------------------------------------------+
```

## Key Architectural Principles
1. **Decoupled Business Logic**: API route handlers do not contain ML model definitions or raw calculations. All inference calls route through injectable service abstractions in `backend/app/services/`.
2. **Scientific Transparency**: The system explicitly flags model status and data requirements. No simulated confidence values or hallucinated predictions are permitted.
3. **Extensibility**: The ingestion and inference pipelines support plugging in new geostationary sensors (INSAT-3D, INSAT-3DR, Himawari, GOES) without refactoring the frontend or UI contracts.
