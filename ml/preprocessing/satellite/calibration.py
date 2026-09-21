"""
Radiometric Calibration & Multi-Spectral Standardization
Transforms raw sensor radiances / Kelvin temperatures into standardized [0.0, 1.0] physical feature tensors.
"""
from typing import Dict, Tuple
import numpy as np

# Verified physical domain ranges across meteorological spectral channels
SPECTRAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "IR": (170.0, 320.0),   # Thermal Infrared 10.8 µm (Cloud top temps down to -103°C up to warm seas 47°C)
    "WV": (190.0, 280.0),   # Water Vapor 6.7 µm (Upper-to-mid tropospheric moisture)
    "VIS": (0.0, 1.0),      # Visible Albedo 0.65 µm (0% to 100% reflectance)
    "PMW": (150.0, 300.0)   # Passive Microwave 85-92 GHz (Internal precipitation structure)
}

def calibrate_channel(
    data: np.ndarray,
    min_val: float,
    max_val: float,
    invert_ir: bool = True
) -> np.ndarray:
    """
    Calibrates a single spectral band to [0.0, 1.0].
    
    Args:
        data: 2D or 3D numpy array of raw physical measurements.
        min_val: Lower physical boundary.
        max_val: Upper physical boundary.
        invert_ir: If True (standard in meteorology for IR), colder cloud tops (lower Kelvin)
                  map to HIGHER activation values (1.0 = deep convection / high cloud tops).
    """
    # Replace any non-finite entries with domain median
    valid_mask = np.isfinite(data)
    if not np.all(valid_mask):
        median_val = np.nanmedian(data) if np.any(valid_mask) else min_val
        data = np.where(valid_mask, data, median_val)

    # Physical clamping to prevent sensor glitch artifacts
    clamped = np.clip(data, min_val, max_val)

    # Min-max normalization
    normalized = (clamped - min_val) / (max_val - min_val)

    # For thermal infrared, invert so that colder eyewalls and deep convection have highest values
    if invert_ir:
        normalized = 1.0 - normalized

    return normalized.astype(np.float32)

def calibrate_satellite_tensor(
    tensor: np.ndarray,
    channels: Tuple[str, ...] = ("IR", "WV", "VIS", "PMW")
) -> np.ndarray:
    """
    Standardizes a multi-channel satellite image tensor [H, W, C] or [N, H, W, C].
    """
    is_batch = tensor.ndim == 4
    if not is_batch:
        tensor = np.expand_dims(tensor, axis=0)

    calibrated = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[-1]

    for c in range(num_channels):
        channel_name = channels[c] if c < len(channels) else "IR"
        min_val, max_val = SPECTRAL_BOUNDS.get(channel_name, (170.0, 320.0))
        invert = (channel_name == "IR")

        for b in range(tensor.shape[0]):
            calibrated[b, :, :, c] = calibrate_channel(
                tensor[b, :, :, c],
                min_val=min_val,
                max_val=max_val,
                invert_ir=invert
            )

    return calibrated if is_batch else calibrated[0]
