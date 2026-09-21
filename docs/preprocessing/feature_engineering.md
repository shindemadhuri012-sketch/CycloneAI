# CycloneAI Feature Engineering Taxonomy

This document catalogs all raw, engineered, and derived features generated across the CycloneAI data preprocessing pipeline.

---

## 1. Kinematic & Trajectory Features

| Feature Name | Data Type | Physical Unit | Derivation / Formula | Meteorological Interpretation | Used in Tasks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `LAT` | Float32 | Decimal Degrees | Direct best-track coordinate | Current cyclone center latitude ($0^\circ\text{N} - 35^\circ\text{N}$) | Track, Intensity |
| `LON` | Float32 | Decimal Degrees | Direct best-track coordinate | Current cyclone center longitude ($45^\circ\text{E} - 105^\circ\text{E}$) | Track, Intensity |
| `NORM_LAT` | Float32 | $[0.0, 1.0]$ | $(\text{LAT} - 0.0) / 35.0$ | Normalized latitudinal position | Neural Track Inputs |
| `NORM_LON` | Float32 | $[0.0, 1.0]$ | $(\text{LON} - 45.0) / 60.0$ | Normalized longitudinal position | Neural Track Inputs |
| `DELTA_LAT` | Float32 | Degrees / step | $\text{LAT}_t - \text{LAT}_{t-1}$ | Latitudinal velocity component | Track Forecasting |
| `DELTA_LON` | Float32 | Degrees / step | $\text{LON}_t - \text{LON}_{t-1}$ | Zonal (longitudinal) velocity component | Track Forecasting |
| `DIST_KM` | Float32 | Kilometers | Haversine great-circle distance | Translation displacement between observations | Track Forecasting |
| `FORWARD_SPEED_KMH` | Float32 | km/h | $\text{DIST\_KM} / \Delta t_{\text{hours}}$ | Forward translation velocity of vortex | Track, Recurvature |
| `BEARING_DEG` | Float32 | Degrees ($0^\circ - 360^\circ$) | Initial great-circle heading | Direction of storm motion (e.g. Northward, West-Northwest) | Track Forecasting |

---

## 2. Atmospheric & Intensity Features

| Feature Name | Data Type | Physical Unit | Derivation / Formula | Meteorological Interpretation | Used in Tasks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EFFECTIVE_WIND_KT` | Float32 | Knots (kt) | $\max(\text{NEWDELHI\_WIND}, \text{WMO\_WIND})$ | Maximum sustained surface wind speed ($1\text{ kt} = 1.852\text{ km/h}$) | Intensity, Classification |
| `CENTRAL_PRES_HPA` | Float32 | hPa / mbar | $\text{NEWDELHI\_PRES}$ or IMD formula imputation | Central minimum barometric pressure | Intensity, Track |
| `PRESSURE_DEFICIT` | Float32 | hPa | $\max(0.0, 1010.0 - \text{CENTRAL\_PRES\_HPA})$ | Pressure drop from ambient periphery; proxy for cyclone energy | Intensity, RI Detection |
| `NORM_WIND` | Float32 | $[0.0, 1.0]$ | $\text{EFFECTIVE\_WIND\_KT} / 150.0$ | Normalized intensity level | Neural Network Inputs |
| `NORM_PRES_DEFICIT` | Float32 | $[0.0, 1.0]$ | $\text{PRESSURE\_DEFICIT} / 100.0$ | Normalized barometric gradient | Neural Network Inputs |
| `IMD_GRADE` | String | Code | Official IMD category string (`D` through `SuCS`) | Official operational Indian categorization | Classification Target |
| `IMD_GRADE_INDEX` | Int64 | $[0, 6]$ | Ordinal mapping of IMD grade | Target class index for focal cross-entropy loss | Classification Target |

---

## 3. Multi-Spectral Spatial Tensor Features

| Tensor Layer | Physical Band | Wavelength | Normalization Formula | Visual Representation | Target Capabilities |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Channel 0** | Thermal Infrared (IR) | $10.8\ \mu\text{m}$ | $1.0 - \frac{T_{\text{kelvin}} - 170.0}{150.0}$ | Cloud-top convective temperatures; cold eyewall convection has highest value | Detection, Intensity, XAI |
| **Channel 1** | Water Vapor (WV) | $6.7\ \mu\text{m}$ | $\frac{T_{\text{kelvin}} - 190.0}{90.0}$ | Upper tropospheric humidity and dry air intrusion channels | Intensity, Classification |
| **Channel 2** | Visible (VIS) | $0.65\ \mu\text{m}$ | $\text{clip}(R_{\text{albedo}}, 0.0, 1.0)$ | Low-level vortex circulation center and cloud texture (daytime) | Detection, Center Fix |
| **Channel 3** | Passive Microwave (PMW) | $85 - 92\ \text{GHz}$ | $\frac{T_{\text{kelvin}} - 150.0}{150.0}$ | Inner-core rainbands and double eyewall structure | Intensity, RI Detection |

---

## 4. Multi-Modal Alignment Bridge

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `image_id` | String | Reference filename or HDF5 index of the preprocessed $201 \times 201 \times 4$ satellite tensor |
| `sid` | String | 13-character IBTrACS Storm Identifier (e.g. `2020136N10090`) |
| `satellite_time` | ISO 8601 | Exact acquisition timestamp of the satellite scan |
| `track_time` | ISO 8601 | Nearest synoptic observation timestamp (within $\le 45\text{ minutes}$) |
| `dt_minutes` | Float32 | Temporal delta between satellite scan and best-track observation |
