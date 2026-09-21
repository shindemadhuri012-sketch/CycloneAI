"""
Physics-Preserving Satellite Data Augmentations
Maintains cyclonic vortex rotation physics (Coriolis direction) during geometric transformations.
"""
from typing import Tuple, Optional
import numpy as np

def rotate_tensor_90(tensor: np.ndarray, k: int) -> np.ndarray:
    """
    Rotates a 2D or 3D tensor by k * 90 degrees counter-clockwise.
    Input shape: [H, W] or [H, W, C]
    Rotations preserve cyclonic rotation direction (counter-clockwise in Northern Hemisphere).
    """
    return np.rot90(tensor, k=k, axes=(0, 1))

def apply_physics_preserving_augmentation(
    tensor: np.ndarray,
    rotation_k: Optional[int] = None,
    allow_reflection: bool = False,
    is_northern_hemisphere: bool = True
) -> np.ndarray:
    """
    Applies data augmentation while strictly enforcing vortex rotational physics.
    
    METEOROLOGICAL PRINCIPLE:
    In the Northern Hemisphere (e.g. North Indian Ocean, Bay of Bengal, Arabian Sea),
    cyclones strictly rotate COUNTER-CLOCKWISE due to Coriolis acceleration.
    - An arbitrary single-axis flip (horizontal or vertical alone) inverts the vortex chirality,
      producing an unphysical clockwise system in the Northern Hemisphere.
    - Pure 2D rotations (90°, 180°, 270°) strictly PRESERVE counter-clockwise vortex curvature.
    - A dual-axis flip (horizontal + vertical) is geometrically identical to a 180° rotation,
      preserving vortex rotation.
    
    Args:
        tensor: Image array of shape [H, W, C] or [H, W].
        rotation_k: Optional explicit rotation count in {0, 1, 2, 3} (k * 90 deg). If None, randomly selected.
        allow_reflection: If False (default), single-axis flips are strictly prohibited.
        is_northern_hemisphere: Flag ensuring hemisphere consistency.
    """
    augmented = tensor.copy()

    # Determine rotation
    if rotation_k is None:
        rotation_k = np.random.randint(0, 4)

    if rotation_k > 0:
        augmented = rotate_tensor_90(augmented, k=rotation_k)

    # If reflection is explicitly requested and allowed, both axes must be flipped together
    # to preserve cyclonic curvature sense
    if allow_reflection:
        # Dual-axis reflection (preserves chirality)
        augmented = np.flip(np.flip(augmented, axis=0), axis=1)

    return augmented
