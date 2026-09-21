"""
Meteorological Eyewall Contrast & Convective Feature Enhancement
Implements Dvorak-aligned temperature slicing and gradient edge filtering to highlight eyewall geometry.
"""
import numpy as np

# Operational Dvorak BD (Broad-Dvorak) Temperature Step Boundaries (in Kelvin)
DVORAK_BD_BOUNDARIES = [
    273.15,  #  0°C (Warm eye threshold)
    242.15,  # -31°C (Outer spiral bands)
    231.15,  # -42°C (Medium gray convective boundary)
    219.15,  # -54°C (White core threshold)
    209.15,  # -64°C (Black eyewall ring threshold)
    198.15,  # -75°C (Light gray cold ring)
    193.15,  # -80°C (Medium gray extreme overshooting tops)
]

def apply_dvorak_bd_slicing(ir_kelvin: np.ndarray) -> np.ndarray:
    """
    Transforms raw Infrared Kelvin array into Dvorak BD discrete enhancement steps [0 to 7].
    Preserves the standard meteorological visual scale used by operational cyclone forecasters.
    """
    sliced = np.zeros_like(ir_kelvin, dtype=np.uint8)
    for i, threshold in enumerate(DVORAK_BD_BOUNDARIES):
        # Colder temperatures yield higher step indices
        sliced[ir_kelvin <= threshold] = i + 1
    return sliced

def enhance_eyewall_features(
    ir_normalized: np.ndarray,
    eyewall_threshold: float = 0.65,
    boost_factor: float = 1.25
) -> np.ndarray:
    """
    Enhances the contrast of cold central dense overcast (CDO) and eyewall pixels.
    Assumes normalized IR input where 1.0 represents coldest cloud tops.
    """
    enhanced = ir_normalized.copy()
    # High activations corresponding to deep convective eyewall are emphasized non-linearly
    mask = enhanced >= eyewall_threshold
    enhanced[mask] = np.clip(
        eyewall_threshold + (enhanced[mask] - eyewall_threshold) * boost_factor,
        0.0,
        1.0
    )
    return enhanced.astype(np.float32)

def compute_gradient_magnitude(image_2d: np.ndarray) -> np.ndarray:
    """
    Computes spatial gradient magnitude using central differences to highlight eyewall curvature.
    """
    gy, gx = np.gradient(image_2d)
    magnitude = np.sqrt(gx**2 + gy**2)
    max_mag = np.max(magnitude)
    if max_mag > 0:
        magnitude /= max_mag
    return magnitude.astype(np.float32)
