"""
Satellite Convective Pattern Analyzer for Cyclone Identification.
Extracts physical morphological convective features from calibrated satellite brightness
temperature arrays (e.g. INSAT-3D TIR-1 / GridSat-B1) and identifies cyclonic vortex signatures.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Meteorological thresholds for tropical convective analysis
DEEP_CONVECTION_TEMP_K = 210.0  # -63.15 deg C (WMO benchmark for severe deep tropical convection)
MODERATE_CONVECTION_TEMP_K = 230.0  # -43.15 deg C
WARM_OCEAN_BACKGROUND_K = 295.0


class SatelliteConvectiveAnalyzer:
    """
    Morphological convective pattern analyzer for satellite infrared imagery.
    Computes physical cloud-top temperature statistics, axisymmetric organization,
    and convective banding indices.
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size

    def extract_convective_features(
        self, brightness_temp_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Extracts meteorological convective features from a 2D brightness temperature array (Kelvin).
        If normalized [0, 1] data is passed, it scales back to [170 K, 320 K].

        Returns:
            Dictionary of physical convective descriptors.
        """
        arr = np.asarray(brightness_temp_matrix, dtype=np.float32)

        # Handle normalized [0, 1] inputs if detected
        if arr.max() <= 1.5 and arr.min() >= 0.0:
            # Scaled: 0 is coldest (170 K) or 0 is warmest (320 K) depending on inversion
            # In Phase 3, cold tops are inverted so 1.0 = cold (170 K) and 0.0 = warm (320 K)
            # We convert to physical Kelvin: T = 320 - arr * 150
            temp_k = 320.0 - arr * 150.0
        else:
            temp_k = np.clip(arr, 160.0, 330.0)

        h, w = temp_k.shape
        cy, cx = h // 2, w // 2

        # 1. Cloud-top temperature extrema
        t_min = float(np.min(temp_k))
        t_mean = float(np.mean(temp_k))
        t_center = float(temp_k[cy, cx])

        # 2. Deep convection fractions
        total_pixels = float(h * w)
        deep_convective_pixels = float(np.sum(temp_k <= DEEP_CONVECTION_TEMP_K))
        deep_convective_fraction = deep_convective_pixels / total_pixels

        moderate_convective_pixels = float(np.sum(temp_k <= MODERATE_CONVECTION_TEMP_K))
        moderate_convective_fraction = moderate_convective_pixels / total_pixels

        # 3. Radial distance grid from vortex center
        y_indices, x_indices = np.ogrid[:h, :w]
        r_grid = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

        # Inner core region (r <= 25% of image radius)
        max_r = min(cy, cx)
        inner_mask = r_grid <= (0.25 * max_r)
        outer_mask = (r_grid > (0.25 * max_r)) & (r_grid <= (0.75 * max_r))

        inner_mean_t = float(np.mean(temp_k[inner_mask])) if np.any(inner_mask) else t_mean
        outer_mean_t = float(np.mean(temp_k[outer_mask])) if np.any(outer_mask) else t_mean

        # Radial gradient: positive if inner core is colder than outer periphery (deep central convection)
        # or if an eye is present (warm center surrounded by cold eyewall ring)
        radial_gradient = outer_mean_t - inner_mean_t

        # 4. Azimuthal symmetry index (standard deviation across angular octants)
        angles = np.arctan2(y_indices - cy, x_indices - cx)
        octant_means = []
        for i in range(8):
            ang_start = -np.pi + i * (np.pi / 4)
            ang_end = ang_start + (np.pi / 4)
            oct_mask = (angles >= ang_start) & (angles < ang_end) & outer_mask
            if np.any(oct_mask):
                octant_means.append(float(np.mean(temp_k[oct_mask])))

        if len(octant_means) >= 4:
            azimuthal_std = float(np.std(octant_means))
            # Normalized symmetry: 1.0 is perfectly symmetric, 0.0 is highly asymmetric
            symmetry_score = float(np.clip(1.0 - (azimuthal_std / 30.0), 0.0, 1.0))
        else:
            azimuthal_std = 15.0
            symmetry_score = 0.50

        # 5. Eye / Warm Core detection
        # An eye is identified if the central 5x5 region is significantly warmer than the surrounding ring
        r_ring_mask = (r_grid >= 0.10 * max_r) & (r_grid <= 0.25 * max_r)
        ring_min_t = float(np.min(temp_k[r_ring_mask])) if np.any(r_ring_mask) else 260.0
        eye_temp_contrast = float(t_center - ring_min_t)
        eye_detected = bool(eye_temp_contrast >= 5.0 and ring_min_t <= DEEP_CONVECTION_TEMP_K)

        # 6. Composite Convective Organization Score (0.0 to 1.0)
        # Deep convection contributes 40%, symmetry 30%, minimum temperature 30%
        cold_score = np.clip((240.0 - t_min) / 60.0, 0.0, 1.0)
        convection_score = np.clip(deep_convective_fraction / 0.15, 0.0, 1.0)
        org_score = float(0.40 * convection_score + 0.30 * symmetry_score + 0.30 * cold_score)
        if eye_detected:
            org_score = min(1.0, org_score + 0.15)

        return {
            "t_min_kelvin": round(t_min, 2),
            "t_mean_kelvin": round(t_mean, 2),
            "t_center_kelvin": round(t_center, 2),
            "deep_convective_fraction": round(deep_convective_fraction, 4),
            "moderate_convective_fraction": round(moderate_convective_fraction, 4),
            "radial_temp_gradient_k": round(radial_gradient, 2),
            "azimuthal_symmetry_score": round(symmetry_score, 4),
            "eye_detected": eye_detected,
            "eye_contrast_k": round(eye_temp_contrast, 2),
            "organization_score": round(org_score, 4),
        }

    def analyze_image(
        self,
        image_matrix: np.ndarray,
        sensor: str = "INSAT-3D",
        channel: str = "TIR-1",
    ) -> Dict[str, Any]:
        """
        Analyzes a satellite image and returns identification diagnosis.
        """
        features = self.extract_convective_features(image_matrix)
        org_score = features["organization_score"]

        # Classification rules based on Dvorak and satellite meteorological criteria
        if features["eye_detected"] or org_score >= 0.65:
            diagnosis = "Organized Tropical Cyclone Present"
            is_cyclone = True
            confidence = float(max(0.70, org_score))
            risk = "Severe" if features["eye_detected"] else "High"
        elif org_score >= 0.45:
            diagnosis = "Developing Deep Depression / Formative Cyclonic Signature"
            is_cyclone = True
            confidence = float(org_score)
            risk = "Moderate"
        elif org_score >= 0.25:
            diagnosis = "Tropical Disturbance / Unorganized Convection"
            is_cyclone = False
            confidence = float(1.0 - org_score)
            risk = "Low"
        else:
            diagnosis = "Non-Cyclonic / Quiescent Cloud Field"
            is_cyclone = False
            confidence = float(1.0 - org_score)
            risk = "Minimal"

        return {
            "sensor": sensor,
            "channel": channel,
            "is_cyclone_identified": is_cyclone,
            "diagnosis": diagnosis,
            "confidence": round(confidence, 4),
            "risk_level": risk,
            "convective_features": features,
        }
