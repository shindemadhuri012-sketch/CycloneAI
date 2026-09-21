# Raw Satellite Data (`data/raw/`)

## Purpose
Stores original, unmodified satellite imagery and telemetry files as obtained directly from earth observation portals (e.g., MOSDAC, NOAA, JMA).

## Data Types
- Full-disk or sector HDF5 / NetCDF4 geostationary files.
- Raw multi-spectral bands (TIR-1, TIR-2, WV, VIS).

## Storage Rules
- Files in this directory are strictly ignored by version control via `.gitignore`.
- Raw data is immutable. Never overwrite original raw files; always write derived products to `data/processed/`.
- No fake or synthetic cyclone images are permitted in this directory.
