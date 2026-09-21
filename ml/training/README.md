# ML Training Module (`ml/training/`)

## Purpose
This module contains training orchestrators, loss functions, learning rate schedules, and experiment tracking scripts for all CycloneAI model components.

## Planned Training Routines
- `train_detector.py`: Binary cross-entropy training for cyclone presence detection.
- `train_classifier.py`: Categorical cross-entropy with focal loss for structural category classification.
- `train_intensity.py`: Smooth L1 / Huber loss regression for maximum sustained wind speed and central pressure.
- `train_tracker.py`: Mean squared error / geodesic distance loss for trajectory prediction.

## Experimentation Standards
- Deterministic seed initialization for scientific reproducibility.
- K-fold cross-validation grouped by storm season to avoid temporal leakage.
- Checkpointing best model weights based on validation loss and operational metrics.

*Status: Ready for training routines in Phases 4–7.*
