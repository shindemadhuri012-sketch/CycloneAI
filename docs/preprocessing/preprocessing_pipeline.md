# Data Preprocessing Pipeline Specification

**Smart India Hackathon 2026**  
**Project:** CycloneAI — Intelligent Tropical Cyclone Analysis & Prediction System  
**Module:** `ml/preprocessing/`

---

## 1. Pipeline Overview

The CycloneAI Data Preprocessing Pipeline converts heterogeneous multi-source raw inputs into synchronized, calibrated, and physically consistent tensors for downstream neural networks.

```
RAW DATA INGESTION
├── Satellite Imagery (HDF5 Multi-Spectral)
└── Best-Track Archive (NOAA IBTrACS North Indian Ocean CSV)
                   │
                   ▼
METEOROLOGICAL CALIBRATION & FEATURE ENGINEERING
├── Satellite: Radiometric Normalization [0.0, 1.0] + Eyewall Contrast
└── Track: Kinematic Vectors (Δlat, Δlon, speed, bearing) + IMD Pressure Deficits
                   │
                   ▼
TEMPORAL SLIDING WINDOW GENERATION
├── Past 24h Trajectory Sequence (t-18h, t-12h, t-6h, t0)
└── Future Forecast Horizons (+6h, +12h, +18h, +24h)
                   │
                   ▼
STORM-LEVEL LEAKAGE-FREE SPLITTING
├── Train Manifest (70% unique storms)
├── Validation Manifest (15% unique storms)
└── Test Manifest (15% unique storms)
```

---

## 2. Satellite Imagery Preprocessing

### A. Spectral Channel Radiometric Normalization
Satellite image arrays are scaled according to verified meteorological bounds:

| Channel | Physical Variable | Calibration Range | Normalization Formula | Meteorological Inversion |
| :---: | :---: | :---: | :---: | :---: |
| **IR (10.8 µm)** | Cloud-Top Brightness Temp | $170.0\text{ K} - 320.0\text{ K}$ | $T_{\text{norm}} = 1.0 - \frac{\text{clip}(T, 170, 320) - 170}{320 - 170}$ | **Yes**: Cold eyewall tops ($\le -65^\circ\text{C}$) map to $1.0$; warm sea maps to $0.0$. |
| **WV (6.7 µm)** | Upper-Tropospheric Moisture | $190.0\text{ K} - 280.0\text{ K}$ | $T_{\text{norm}} = \frac{\text{clip}(T, 190, 280) - 190}{280 - 190}$ | No: High moisture values retain higher activation. |
| **VIS (0.65 µm)** | Surface / Cloud Albedo | $0.0 - 1.0$ ($0\% - 100\%$) | $T_{\text{norm}} = \text{clip}(R, 0.0, 1.0)$ | No: Daytime reflectance directly represents optical thickness. |
| **PMW (85-92 GHz)** | Polarized Microwave Temp | $150.0\text{ K} - 300.0\text{ K}$ | $T_{\text{norm}} = \frac{\text{clip}(T, 150, 300) - 150}{300 - 150}$ | No: Highlights heavy hydrometeor rainbands. |

### B. Eyewall Contrast & Dvorak Temperature Slicing
To emulate operational Dvorak BD (Broad-Dvorak) curve visual analysis, infrared arrays undergo discrete temperature slicing:
- Step 1: Eye temperature $\le 0^\circ\text{C}$ ($273.15\text{ K}$)
- Step 2: Outer spiral bands $\le -31^\circ\text{C}$ ($242.15\text{ K}$)
- Step 3: Medium gray convection $\le -42^\circ\text{C}$ ($231.15\text{ K}$)
- Step 4: White core $\le -54^\circ\text{C}$ ($219.15\text{ K}$)
- Step 5: Black eyewall ring $\le -64^\circ\text{C}$ ($209.15\text{ K}$)
- Step 6: Cold ring $\le -75^\circ\text{C}$ ($198.15\text{ K}$)
- Step 7: Extreme overshooting tops $\le -80^\circ\text{C}$ ($193.15\text{ K}$)

### C. Physics-Preserving Coriolis Augmentation
- **Atmospheric Rule**: Tropical cyclones in the Northern Hemisphere (including Bay of Bengal and Arabian Sea) rotate **strictly counter-clockwise** due to Coriolis acceleration.
- **Augmentation Standard**: Single-axis horizontal or vertical flips invert vortex chirality and produce an unphysical clockwise anticyclone. Therefore, CycloneAI exclusively permits:
  1. Discrete 2D rotations: $90^\circ, 180^\circ, 270^\circ$ (strictly preserves counter-clockwise curl).
  2. Dual-axis simultaneous reflection (equivalent to a $180^\circ$ rotation).

---

## 3. Best-Track & Kinematic Preprocessing

### A. Geodesic Distance & Translation Speed
For storm center coordinates $(lat_1, lon_1)$ at $t-1$ and $(lat_2, lon_2)$ at $t$, the great-circle geodesic distance $d$ in kilometers is computed using the Haversine equation:

$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
$$d = 2 \cdot R_{\text{earth}} \cdot \arctan2\left(\sqrt{a}, \sqrt{1 - a}\right)$$

where $R_{\text{earth}} = 6371.0\text{ km}$, $\Delta\phi = \text{radians}(lat_2 - lat_1)$, $\Delta\lambda = \text{radians}(lon_2 - lon_1)$.

Forward translation speed:
$$v_{\text{forward}} = \min\left(\frac{d}{\Delta t_{\text{hours}}}, 120.0\text{ km/h}\right)$$

### B. Compass Heading Bearing
The initial forward bearing angle $\theta \in [0^\circ, 360^\circ]$ is calculated as:

$$\theta = \text{atan2}\Big(\sin(\Delta\lambda)\cos(\phi_2), \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta\lambda)\Big)$$

### C. Pressure Deficit & IMD Empirical Imputation
Ambient sea-level pressure in the tropical Indian Ocean boundary layer is standardized at $P_{\text{ambient}} = 1010.0\text{ hPa}$.

Pressure Deficit:
$$\Delta P = \max(0.0, P_{\text{ambient}} - P_{\text{central}})$$

When central barometric pressure is unrecorded in historical track data, it is estimated using the official **India Meteorological Department (IMD) Mishra & Gupta empirical formula**:

$$V_{\text{max}} = 14.2 \cdot \sqrt{\Delta P} \implies P_{\text{central}} = P_{\text{ambient}} - \left(\frac{V_{\text{max}}}{14.2}\right)^2$$

This enforces hydrostatic consistency rather than injecting unphysical mean imputation.

---

## 4. Sliding Window Sequence Formulation

For trajectory and intensity sequence models (e.g. LSTM / Transformer / GNN):
- **Input Window ($L_{\text{in}} = 4$ steps, $24\text{ hours}$ past at 6-hr cadence)**:
  $$\mathbf{X}_t = \big[(\text{lat}_\tau, \text{lon}_\tau, \Delta\text{lat}_\tau, \Delta\text{lon}_\tau, v_\tau, \theta_\tau, V_{\tau}, \Delta P_\tau)\big]_{\tau=t-18\text{h}}^{t}$$
- **Target Horizons ($L_{\text{out}} = 4$ steps, $+6\text{h}, +12\text{h}, +18\text{h}, +24\text{h}$)**:
  $$\mathbf{Y}_t = \big[(\text{lat}_\tau, \text{lon}_\tau, V_{\tau})\big]_{\tau=t+6\text{h}}^{t+24\text{h}}$$

**Isolation Rule**: All sliding sequences are constructed strictly inside each unique `SID`. Sequences never bridge between two different storms.

---

## 5. Storm-Level Leakage Prevention

- Partitions are divided strictly by unique `SID`:
  - **Train Manifest**: 70% of historical cyclones.
  - **Validation Manifest**: 15% of historical cyclones.
  - **Test Manifest**: 15% of historical cyclones.
- Inter-partition storm ID intersection:
  $$\text{Train}_{\text{SID}} \cap \text{Val}_{\text{SID}} = \emptyset, \quad \text{Train}_{\text{SID}} \cap \text{Test}_{\text{SID}} = \emptyset, \quad \text{Val}_{\text{SID}} \cap \text{Test}_{\text{SID}} = \emptyset$$
