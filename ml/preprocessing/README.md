# ML Preprocessing Module (`ml/preprocessing/`)

## Purpose
This module houses deterministic data transformation pipelines that convert raw satellite observations into clean, normalized tensors ready for neural network training and inference.

## Key Pipelines
1. **Calibration & Radiometry**:
   - Conversion of raw digital counts (DN) to Brightness Temperature (Kelvin / Celsius) for Infrared bands.
   - Top-of-Atmosphere (TOA) reflectance calculation for Visible bands.
2. **Geospatial Processing**:
   - Center crop extraction focused on the cyclone eye / central dense overcast (CDO).
   - Coordinate re-projection and spatial resampling to standard resolutions (e.g., 224x224, 256x256, 512x512).
3. **Quality Control & Cleansing**:
   - Bad pixel masking and cloud-edge artifact suppression.
   - Temporal missing-frame interpolation.
4. **Data Augmentation**:
   - Physics-informed augmentations (random rotational flips preserving cyclone vortex physics).

*Status: Architecture defined. Implementation commences in Phase 3.*
