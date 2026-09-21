# Selected Dataset Strategy & System Mapping

## 1. Architectural Strategy Overview

To fulfill the Smart India Hackathon 2026 problem statement within practical computational limits for a student engineering team, **CycloneAI** implements a **hybrid multi-source data pipeline**:

```
+-------------------------------------------------------------+
|                      SATELLITE IMAGERY                      |
|                Primary: TCIR Benchmark HDF5                 |
|            Secondary: ISRO MOSDAC INSAT-3D Samples          |
+-------------------------------------------------------------+
                               │
               Link via (SID, ISO_TIME, Coordinates)
                               │
                               ▼
+-------------------------------------------------------------+
|                 TRACK & INTENSITY GROUND TRUTH              |
|        Primary: NOAA NCEI IBTrACS v04r01 (NI Subset)        |
|           Official IMD RSMC New Delhi Measurements          |
+-------------------------------------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
|                  FIVE CORE AI/ML CAPABILITIES               |
|  1. Detection  2. Classification  3. Intensity  4. Track    |
|                      5. Explainable AI                      |
+-------------------------------------------------------------+
```

---

## 2. Selected Core Datasets

### A. Primary Satellite Imagery: **TCIR Benchmark Dataset**
- **Citation**: Chen, B., Chen, B.-F., & Lin, H.-T. (2018). *Rotation-blended CNNs on a new open dataset for tropical cyclone image-to-intensity regression*. ACM KDD 2018.
- **Why Selected**:
  - Contains **70,000+ pre-processed and centered storm chips** (201×201 pixels, ~4 km/pixel resolution).
  - Provides **4 distinct physical channels**:
    1. **Infrared (IR)** (10.5–11.5 µm): Captures cloud-top brightness temperature, convective eye walls, and spiral feeder bands.
    2. **Water Vapor (WV)** (6.5–7.0 µm): Captures mid-to-upper tropospheric moisture patterns and environmental steering flows.
    3. **Visible (VIS)** (0.55–0.75 µm): Captures fine cloud textures, overshooting tops, and low-level center circulation (daytime).
    4. **Passive Microwave (PMW)**: Penetrates upper cloud decks to reveal precipitation cores and eyewall replacement cycles.
  - Formatted cleanly as HDF5 files (`matrix` array and `info` table), making it immediately ingestible by PyTorch `DataLoader` without requiring complex GIS re-projection pipelines.

### B. Primary Track & Intensity Ground Truth: **NOAA NCEI IBTrACS v04r01**
- **Subset**: North Indian Ocean (`ibtracs.NI.list.v04r01.csv`).
- **Citation**: Knapp, K. R., et al. (2018/2024). *International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4.01*. NOAA NCEI. DOI: 10.25921/82ty-9e16.
- **Why Selected**:
  - World standard for tropical cyclone best tracks.
  - **Embedded IMD Measurements**: Directly includes official observations from the India Meteorological Department (RSMC New Delhi) under columns `NEWDELHI_LAT`, `NEWDELHI_LON`, `NEWDELHI_WIND`, `NEWDELHI_PRES`, and `NEWDELHI_GRADE`.
  - Open data, publicly hosted, zero registration required, downloadable via direct HTTPS.
  - Compact file size (~5 MB for NI basin, ~250 MB for global) enables fast processing and validation.

### C. Secondary Regional Satellite Source: **ISRO MOSDAC INSAT-3D/3DR (Demonstration Granules)**
- **Why Selected**:
  - Directly represents Indian space infrastructure (ISRO geostationary satellites at 74°E and 82°E).
  - Selected historical storm granules (e.g., Cyclone Amphan 2020, Cyclone Biparjoy 2023) will be processed as sample demonstration fixtures in `data/sample/` to prove operational integration with native Indian Earth Observation satellites.

---

## 3. Global Pretraining vs. North Indian Ocean Focus

### The Data Scarcity Dilemma
The North Indian Ocean (Bay of Bengal and Arabian Sea) is an active tropical cyclone basin, but accounts for only **~5 to 7% of global cyclones** (averaging ~4 to 6 named cyclonic disturbances per year). 
- In 30 years of satellite observations, the North Indian Ocean has yielded approximately 150 to 180 notable cyclonic events.
- Attempting to train a deep vision model (e.g., ConvNeXt, Vision Transformer) from scratch solely on ~150 storms inevitably causes severe overfitting and failure to generalize.

### The CycloneAI Strategy:
1. **Pretraining on Global Satellite Data (TCIR)**: The universal physics of cyclone spiral banding, central dense overcast (CDO), and eyewall dynamics are consistent across all tropical oceans. We leverage global samples to learn robust spatial feature representations.
2. **Fine-Tuning & Evaluation on North Indian Ocean (IBTrACS NI)**: The models are evaluated and calibrated against the IMD 3-minute sustained wind standards and regional bathymetric/atmospheric characteristics of the Bay of Bengal and Arabian Sea.

---

## 4. Multi-Modal Linking Schema

Satellite image chips and numerical track records are bridged through a composite key:

$$\text{Composite Key} = (\text{Storm Identifier [SID]}, \text{Observation Timestamp [ISO\_TIME]})$$

- **Spatial Tolerance**: Within $\pm 0.5^\circ$ latitude/longitude of the best-track storm center.
- **Temporal Tolerance**: Satellite observations within $\pm 30$ minutes of the nearest 3-hourly or 6-hourly best-track report.

---

## 5. Dataset-to-Feature / Task Mapping

| System Feature / Task | Assigned Dataset | Input Data Features | Output / Target Label |
| :--- | :--- | :--- | :--- |
| **1. Cyclone Presence Identification** | TCIR (Storm chips vs. Disturbance/Negative chips) | Multi-channel satellite tensor: `[B, C=4, H=201, W=201]` | Binary presence flag: `0` (Cloud cluster / non-cyclone) vs. `1` (Organized vortex) |
| **2. Cyclone Pattern & Scale Classification** | TCIR + IBTrACS IMD Grade | Normalized IR + WV channels + temporal sequence | Multi-class category: IMD 7-class scale (`D`, `DD`, `CS`, `SCS`, `VSCS`, `ESCS`, `SuCS`) |
| **3. Intensity Estimation & Trend** | TCIR + IBTrACS Wind / Pressure | Infrared Brightness Temp + Past 6h/12h wind history | Continuous regression: $V_{\text{max}}$ (knots) and $P_{\text{min}}$ (hPa) with +6h, +12h, +24h trends |
| **4. Track Trajectory Prediction** | IBTrACS NI (`LAT`, `LON`, `ISO_TIME`) + Image Latent Embedding | Sequence of past coordinates $[(lat_t, lon_t)_{t=-12h}^0]$ + environmental vectors | Future coordinates: $[(lat_{t+\Delta}, lon_{t+\Delta})]$ for $\Delta \in \{+6h, +12h, +24h, +48h\}$ |
| **5. Explainable AI (Grad-CAM)** | TCIR IR & VIS Channels + Classification Model | Backpropagated gradients into final convolutional / attention layer | 2D heatmap overlaid on satellite imagery highlighting eyewall and curved convective feeder bands |

---

## 6. Minimal Realistic Development Subset

For rapid local iteration, development, and unit testing on student machines:
- **Track Subset**: 1990–2024 North Indian Ocean cyclones from `ibtracs.NI.list.v04r01.csv` (~1,500 track rows, ~5 MB).
- **Satellite Subset**: A curated partition of 500 to 1,000 multi-channel storm frames (~100 MB to 200 MB) covering distinct lifecycle stages (Genesis, Deepening, Severe Cyclone, Landfall/Decay) representing benchmark Indian Ocean storms (Amphan, Fani, Biparjoy, Tauktae).
