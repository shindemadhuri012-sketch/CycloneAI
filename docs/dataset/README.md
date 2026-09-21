# Dataset Research & Specifications (`docs/dataset/`)

## Target Satellite Sensors
1. **INSAT-3D & INSAT-3DR (ISRO)**
   - Orbit: Geostationary (74°E & 82°E)
   - Coverage: Indian Ocean, Bay of Bengal, Arabian Sea, South Asia
   - Channels:
     - Visible (VIS): 0.55 - 0.75 µm (1 km resolution)
     - Shortwave Infrared (SWIR): 1.55 - 1.70 µm (1 km resolution)
     - Middle Infrared (MIR): 3.80 - 4.00 µm (4 km resolution)
     - Thermal Infrared 1 (TIR-1): 10.2 - 11.3 µm (4 km resolution)
     - Thermal Infrared 2 (TIR-2): 11.5 - 12.5 µm (4 km resolution)
     - Water Vapor (WV): 6.5 - 7.1 µm (8 km resolution)
   - Cadence: 30 minutes (staggered to 15 minutes between 3D and 3DR)

2. **Himawari-8 & 9 (JMA)**
   - Advanced Himawari Imager (AHI) with 16 spectral channels, providing cross-basin validation and auxiliary observations.

3. **Ground Truth / Best-Track Sources**
   - **IBTrACS**: Global baseline offering 3-hourly consensus storm centers, wind speeds, and pressures.
   - **IMD RSMC New Delhi Reports**: Ground-truth records for North Indian Ocean cyclones.
