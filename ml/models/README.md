# ML Models Module (`ml/models/`)

## Purpose
Contains neural network architecture definitions in PyTorch, parameter specifications, and model factory interfaces.

## Architectures Under Consideration
1. **Detection & Classification**:
   - `convnext.py` / `efficientnet.py`: Modern convolutional backbones well-suited for fine-grained cloud spiral pattern analysis.
   - `vit_cyclone.py`: Vision Transformer for capturing global spatial relationships across convective cloud bands.
2. **Intensity Regression**:
   - Dual-head CNN/ViT outputting wind speed (knots) and central pressure (hPa) with uncertainty estimation.
3. **Track Forecasting**:
   - `spatiotemporal_transformer.py` or `lstm_tracker.py`: Recurrent and attention-based trajectory predictors fusing past coordinate sequences with satellite feature embeddings.

*Note: Model weight files (*.pt, *.pth, *.onnx) are strictly excluded from version control via `.gitignore`.*
