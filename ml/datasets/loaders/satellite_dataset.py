"""
Satellite image dataset loader for multi-spectral cyclone chips.
Loads 4-channel tensors [IR, WV, VIS, PMW] and performs channel-wise radiometric scaling.
"""
from typing import Dict, Optional, Tuple, Callable
import numpy as np

class CycloneSatelliteDataset:
    """
    Dataset loader for satellite image arrays.
    Compatible with HDF5 benchmark arrays (such as TCIR) and preprocessed numpy granules.
    """
    def __init__(
        self,
        images: np.ndarray,      # Shape: [N, 201, 201, C]
        labels: np.ndarray,      # Intensity (wind speed) or category class indices
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.images)

    def _normalize_channels(self, chip: np.ndarray) -> np.ndarray:
        """
        Applies verified physical domain normalization to spectral channels.
        Chip shape: [H, W, C]
        """
        chip_norm = chip.copy().astype(np.float32)

        # Channel 0: Infrared (Kelvin, typical range 170K - 320K)
        if chip_norm.shape[-1] > 0:
            chip_norm[:, :, 0] = np.clip((chip_norm[:, :, 0] - 170.0) / (320.0 - 170.0), 0.0, 1.0)

        # Channel 1: Water Vapor (Kelvin, typical range 190K - 280K)
        if chip_norm.shape[-1] > 1:
            chip_norm[:, :, 1] = np.clip((chip_norm[:, :, 1] - 190.0) / (280.0 - 190.0), 0.0, 1.0)

        # Channel 2: Visible (Reflectance, typical range 0.0 - 1.0)
        if chip_norm.shape[-1] > 2:
            chip_norm[:, :, 2] = np.clip(chip_norm[:, :, 2], 0.0, 1.0)

        # Channel 3: Passive Microwave (Kelvin, typical range 150K - 300K)
        if chip_norm.shape[-1] > 3:
            chip_norm[:, :, 3] = np.clip((chip_norm[:, :, 3] - 150.0) / (300.0 - 150.0), 0.0, 1.0)

        return chip_norm

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        chip = self.images[idx]
        if self.normalize:
            chip = self._normalize_channels(chip)

        # Transpose from [H, W, C] to PyTorch standard [C, H, W]
        chip_tensor = np.transpose(chip, (2, 0, 1))

        if self.transform:
            chip_tensor = self.transform(chip_tensor)

        return {
            "image": chip_tensor,
            "label": self.labels[idx]
        }
