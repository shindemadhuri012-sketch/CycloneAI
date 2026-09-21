# External Meteorological Datasets (`data/external/`)

## Purpose
Houses ground-truth meteorological archives and auxiliary environmental datasets used for label alignment, validation, and multi-modal feature fusion.

## Planned Datasets
- **IBTrACS (International Best Track Archive for Climate Stewardship)**: 3-hourly or 6-hourly storm center positions, maximum sustained wind speeds, central pressure, and storm classifications.
- **IMD Best Track Reports**: Official post-season cyclone reports published by the India Meteorological Department for North Indian Ocean events.
- **NWP Reanalysis / Environmental Fields**: Sea Surface Temperature (SST), 850-200 hPa vertical wind shear, and mid-tropospheric relative humidity from ERA5 or GFS.

## Rules
- All records must originate from verified scientific meteorological agencies.
- Excluded from version control via `.gitignore` except documentation and schemas.
