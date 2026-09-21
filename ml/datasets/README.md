# CycloneAI Dataset Engineering Pipeline (`ml/datasets/`)

This package provides verified, reproducible ingestion, parsing, loading, and validation routines for tropical cyclone data.

## Subpackage Layout

- **`download/`**: Verified acquisition scripts fetching public datasets from authoritative endpoints (e.g. NOAA NCEI IBTrACS).
- **`metadata/`**: Parsers converting raw tabular/hierarchical formats into clean, typed DataFrames adhering to the `data_dictionary.md` schema.
- **`loaders/`**: PyTorch `Dataset` implementations for multi-channel satellite imagery and sequential track coordinates.
- **`validation/`**: Data integrity audit scripts and storm-level train/validation/test split strategies preventing data leakage.

## Usage Quickstart

### 1. Download North Indian Ocean Best-Track Data
```powershell
python ml/datasets/download/download_ibtracs.py --basin NI
```

### 2. Validate Data Integrity & Storm Splits
```powershell
python ml/datasets/validation/validate_tracks.py
```
