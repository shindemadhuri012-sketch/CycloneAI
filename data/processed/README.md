# Processed Satellite Data (`data/processed/`)

## Purpose
Contains calibrated, normalized, spatialized, and quality-controlled datasets ready for ingestion into deep learning training and validation pipelines.

## Derived Products
- Geometrically aligned and centered storm patches (e.g., 256x256 pixel numpy `.npy` or `.h5` arrays).
- Calibrated Brightness Temperature arrays (Kelvin) normalized to `[0, 1]` or standardized via z-score.
- Multi-channel composite tensors (e.g., TIR + WV + VIS).
- Split manifests (`train.csv`, `val.csv`, `test.csv`).

## Rules
- Excluded from git tracking via `.gitignore`.
- Deterministically reproducible using scripts from `ml/preprocessing/`.
