# Comprehensive Tropical Cyclone Dataset Comparison

This document provides a comparative analysis of candidate datasets researched for **CycloneAI** (Smart India Hackathon 2026). All fields, parameters, and limitations reflect verified scientific documentation.

---

## 1. Candidate Dataset Matrix

| Dataset | Source / Provider | Period | Geographic Region | Imagery Available | Track Coordinates | Intensity (Wind / Pressure) | Meteorological Labels | Multi-Source Suitability | Access Method | License / Restrictions | Recommended Use in CycloneAI | Key Limitations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **IBTrACS (v04r01)** | NOAA NCEI / WMO | 1848–Present (NIO: 1884–Present) | Global (Basin subsets: NI, NA, EP, WP, SI, SP, SA) | No (Tabular track & meteorology only) | **Yes** (3-hr & 6-hr lat/lon) | **Yes** (WMO, IMD `NEWDELHI_*`, JTWC `USA_*` 1-min & 3-min winds, central MSLP in hPa) | **Yes** (Nature of storm, IMD grade, WMO category) | **High** (Serves as global ground-truth anchor for spatial imagery) | Direct HTTPS / FTP download (CSV, NetCDF, Shapefile) | Public Domain (Open Data, NOAA) | **Primary Track & Intensity Ground Truth**. Essential for trajectory forecasting and validation. | No native satellite pixel arrays. |
| **TCIR (Tropical Cyclone Image-to-intensity Regression)** | National Taiwan University (Chen et al., ACM KDD 2018) | 2003–2017 | Global (all tropical ocean basins including North Indian Ocean) | **Yes** (4 channels: IR, WV, VIS, Passive Microwave PMW) | **Yes** (Integrated within metadata) | **Yes** (Vmax in knots, interpolated to image frames) | **Yes** (Matched with IBTrACS storm records) | **High** (Pre-aligned image-intensity-track benchmark) | Direct academic HTTP download (HDF5 format, ~70,000 samples) | Academic Research / Open (Attribution required) | **Primary Satellite Imagery Benchmark**. Ideal for detection, classification, intensity regression, and Grad-CAM pre-training. | Bounded to 2003–2017; images cropped to fixed 201×201 grid at 4 km/pixel. |
| **INSAT-3D / 3DR Imager (L1B/L1C)** | ISRO / MOSDAC (Space Applications Centre) | 2013–Present (3D), 2016–Present (3DR) | South Asia & Indian Ocean (74°E & 82°E orbital slots) | **Yes** (6 channels: VIS, SWIR, MIR, TIR-1, TIR-2, WV) | Auxiliary (Inferred via IMD track bulletins) | Auxiliary (Cross-referenced via IMD bulletins) | Unlabeled raw pixels (requires IBTrACS alignment) | **High** (Highest native resolution over Bay of Bengal & Arabian Sea) | Semi-automated (Requires free registration on mosdac.gov.in & API order tokens) | Restricted to Registered Users (Free for educational/research in India) | **Secondary Satellite Source & Demonstration**. Local Indian regional calibration and real-world operational testing. | Bulk download requires authenticated API orders; full-disk granules are large (~100 MB–1 GB per slot). |
| **NOAA GridSat-B1 (v02)** | NOAA NCEI | 1980–Present | Global (70°S to 70°N, 0.07° equal-angle grid) | **Yes** (IR 11 µm, VIS 0.6 µm, WV 6.7 µm CDR) | No (Requires external IBTrACS join) | No (Requires external IBTrACS join) | Unlabeled raster grids | **High** (Longest uniform geostationary CDR, 3-hourly cadence) | Direct HTTPS / THREDDS / AWS S3 Open Data | Public Domain (NOAA Open Access) | **Fallback Large-Scale Archive**. Useful for extending temporal span backwards to 1980. | Requires geospatial storm cropping and coordinate re-projection pipeline from global NetCDF-4 grids. |
| **IMD RSMC Best Track Reports** | India Meteorological Department (RSMC New Delhi) | 1990–Present | North Indian Ocean (Bay of Bengal & Arabian Sea) | No (Official scientific PDFs & text tables) | **Yes** (3-hourly official IMD coordinates) | **Yes** (Official IMD 3-min sustained wind speed & central pressure) | **Yes** (Official IMD scale: D, DD, CS, SCS, VSCS, ESCS, SuCS) | **High** (Direct source for North Indian Ocean operational truth) | Web portal / PDF reports (`rsmcnewdelhi.imd.gov.in`) | Government of India (Public Information) | **Validation Standard**. (Note: Already incorporated into IBTrACS under `NEWDELHI_*` columns). | Dispersed PDF reports if obtained directly; parsing raw tables is prone to formatting inconsistencies. |
| **NASA IMPACT DeepCyclone** | NASA Marshall Space Flight Center / IMPACT | 2016–2020 | Global | **Yes** (Geostationary infrared chips, GOES & Himawari) | Inferred | **Yes** (Wind speed regression labels) | Category bins | **Moderate** (Focused on intensity estimation) | Academic download / Zenodo / GitHub | Open Access | **Comparative Reference**. Benchmark comparison for intensity regression architectures. | Primary focus on Atlantic and Pacific; limited coverage of North Indian Ocean. |
| **EUMETSAT IODC (Meteosat-8/9)** | EUMETSAT | 2016–Present (Meteosat-8 IODC until 2022, Meteosat-9 thereafter) | Indian Ocean Data Coverage (41.5°E / 45.5°E) | **Yes** (12 SEVIRI channels) | No | No | Unlabeled | **Moderate** (Excellent radiometric quality over Arabian Sea) | EUMETSAT Data Store / API (Registration required) | Open for Research with EUMETSAT Earth Observation Portal account | **Optional Auxiliary**. High-resolution multi-spectral validation. | Complex API token management, multi-gigabyte data volumes per storm. |

---

## 2. Evidence-Based Dataset Synthesis & Selection

A single dataset **cannot satisfy all requirements** of the Smart India Hackathon problem statement:
- Raw satellite repositories (MOSDAC, GridSat, EUMETSAT) lack automated storm-centered segmentation, quality-controlled bounding boxes, and ground-truth intensity labels.
- Historical track databases (IBTrACS) possess world-class numerical trajectories and IMD-verified classifications, but lack satellite imagery pixel arrays.

### Recommended Dual-Source Architecture:
1. **Satellite Feature Backbone**: **TCIR (Tropical Cyclone Image-to-intensity Regression)**
   - Pre-centered 201×201 pixel arrays across 4 physically meaningful channels (Infrared, Water Vapor, Visible, Passive Microwave).
   - Pre-aligned with storm identifiers and ground-truth intensities.
   - Enables training deep convolutional backbones and Vision Transformers on a standard laptop/GPU without downloading tens of terabytes of full-disk satellite imagery.
2. **Track, Intensity & IMD Verification Core**: **NOAA NCEI IBTrACS v04r01 (North Indian Ocean Subset: `ibtracs.NI.list.v04r01.csv`)**
   - Contains all recorded cyclones in the Bay of Bengal and Arabian Sea.
   - Embeds official India Meteorological Department RSMC New Delhi measurements: `NEWDELHI_WIND`, `NEWDELHI_PRES`, and `NEWDELHI_GRADE`.
   - Directly downloadable via HTTPS without API keys or registrations.
3. **Demonstration & Regional Evaluation**: **ISRO INSAT-3D/3DR via MOSDAC**
   - Sample granules for major Indian cyclones (e.g., Cyclone Amphan 2020, Cyclone Biparjoy 2023, Cyclone Michaung 2023) to demonstrate compatibility with native Indian space assets during the SIH presentation.
