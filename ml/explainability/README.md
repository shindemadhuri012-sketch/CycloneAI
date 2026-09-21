# ML Explainability Module (`ml/explainability/`)

## Purpose
Provides scientific interpretability tools that generate visual and feature-attribution explanations for AI predictions.

## Methodology
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Generates 2D heatmaps overlaying satellite imagery to visualize whether the neural network is focusing on genuine physical cloud patterns (eyewall, primary feeder band, central dense overcast) rather than spurious background artifacts.
- **Attention Map Visualization**: For Vision Transformer architectures, extracts self-attention weights across spatial patches.
- **Physical Validation**: Comparing AI focus areas against meteorological Dvorak rules (curved band patterns, shear patterns, eye characteristics).

*Status: Ready for implementation in Phase 8.*
