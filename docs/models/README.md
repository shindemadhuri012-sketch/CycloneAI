# Model Architecture Specifications (`docs/models/`)

## Candidate AI/ML Architecture Pipeline

### 1. Cyclone Identification (Presence Detection)
- **Task**: Binary image classification (Cyclone vs. Non-Cyclone / Disturbance).
- **Architectures**: EfficientNet-B2/B3, ConvNeXt-Tiny.
- **Inputs**: Infrared Brightness Temperature patch (256x256x1 or 256x256x3 multi-channel).
- **Target Metric**: High Recall to prevent missing nascent cyclonic disturbances.

### 2. Cyclone Category Classification
- **Task**: Multi-class classification conforming to IMD categories:
  - Low Pressure Area (< 17 knots)
  - Depression (17–27 knots)
  - Deep Depression (28–33 knots)
  - Cyclonic Storm (34–47 knots)
  - Severe Cyclonic Storm (48–63 knots)
  - Very Severe Cyclonic Storm (64–89 knots)
  - Extremely Severe Cyclonic Storm (90–119 knots)
  - Super Cyclonic Storm (>= 120 knots)
- **Loss Function**: Weighted Focal Loss to mitigate class imbalance in extreme events.

### 3. Intensity Estimation
- **Task**: Continuous regression predicting Maximum Sustained Wind Speed (knots) and Central Minimum Sea-Level Pressure (hPa).
- **Architectures**: Multi-task CNN/ViT regression head.

### 4. Track Prediction
- **Task**: Multi-step spatio-temporal coordinate forecasting (+6h, +12h, +24h, +48h).
- **Architectures**: Spatio-temporal Transformer / ConvLSTM fusing past trajectory sequences with extracted satellite feature vectors.
