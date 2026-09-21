"""
Dvorak-Aligned Satellite Convective Saliency Engine.
Translates satellite infrared cloud-top brightness temperature fields into
meteorologically grounded spatial saliency heatmaps, Dvorak BD enhancement zones,
eyewall axisymmetry gradients, and spiral convective banding masks.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Official Dvorak BD-Curve Enhancement Temperature Slices (Celsius)
DVORAK_BD_SLICES = [
    {"label": "Warm Eye / Clear Pocket", "t_min": -30.0, "t_max": 20.0, "color": "#FFCC00", "code": "EYE"},
    {"label": "Outer Convection / Cirrus Canopy", "t_min": -41.0, "t_max": -30.0, "color": "#70A0D0", "code": "OUTER"},
    {"label": "Curved Convective Banding (Medium Gray)", "t_min": -54.0, "t_max": -41.0, "color": "#5080B0", "code": "BAND_MED"},
    {"label": "Deep Convective Banding (Light Gray)", "t_min": -64.0, "t_max": -54.0, "color": "#90C0E0", "code": "BAND_LT"},
    {"label": "Central Dense Overcast / Eyewall (Dark Gray)", "t_min": -70.0, "t_max": -64.0, "color": "#304060", "code": "CDO_DG"},
    {"label": "Intense Convective Ring (Black)", "t_min": -76.0, "t_max": -70.0, "color": "#101828", "code": "RING_BLK"},
    {"label": "Violent Overshooting Tops (White)", "t_min": -86.0, "t_max": -76.0, "color": "#FFFFFF", "code": "TOP_WHT"},
    {"label": "Extreme Convective Core (Light Gray Cold)", "t_min": -100.0, "t_max": -86.0, "color": "#E0E0E0", "code": "CORE_CD"},
]


class SatelliteSaliencyEngine:
    """
    Computes spatial convective attribution masks and morphological indicators
    from satellite infrared brightness temperature (TB) matrices.
    """

    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size
        self.bd_slices = DVORAK_BD_SLICES

    def generate_synthetic_cyclone_field(
        self,
        center_row: Optional[int] = None,
        center_col: Optional[int] = None,
        intensity_factor: float = 0.7,
    ) -> np.ndarray:
        """
        Synthesizes a realistic physical cloud-top brightness temperature field (Celsius)
        matching an axisymmetric tropical cyclone with warm eye, cold ring, and spiral bands.
        """
        N = self.grid_size
        cr = center_row if center_row is not None else N // 2
        cc = center_col if center_col is not None else N // 2

        Y, X = np.ogrid[:N, :N]
        dist = np.sqrt((X - cc) ** 2 + (Y - cr) ** 2)
        theta = np.arctan2(Y - cr, X - cc)

        # Baseline tropical background temperature (-10C to +15C)
        tb = -15.0 + 30.0 * np.clip(dist / (N * 0.45), 0.0, 1.0)

        # Eyewall cold ring (radius ~ 25 to 45 pixels)
        eyewall_mask = np.exp(-((dist - 35.0) ** 2) / (2.0 * (15.0 ** 2)))
        cold_ring_t = -78.0 * intensity_factor
        tb = tb * (1.0 - eyewall_mask) + cold_ring_t * eyewall_mask

        # Warm eye pocket (radius < 18 pixels)
        eye_mask = np.exp(-(dist ** 2) / (2.0 * (12.0 ** 2)))
        tb = tb * (1.0 - eye_mask) + (-15.0) * eye_mask

        # Logarithmic spiral convective bands
        spiral_phase = theta - 1.8 * np.log(np.maximum(10.0, dist))
        spiral_band = np.cos(spiral_phase * 2.0)
        spiral_convection = np.clip(spiral_band, 0.0, 1.0) * np.exp(-dist / 80.0)
        tb -= spiral_convection * 30.0 * intensity_factor

        return np.clip(tb, -90.0, 30.0).astype(np.float32)

    def compute_convective_saliency(
        self, tb_matrix: np.ndarray, center: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Computes 2D convective saliency heatmap and Dvorak BD morphological metrics.
        """
        N = tb_matrix.shape[0]
        if center is None:
            # Estimate center as centroid of deepest cold clouds (TB < -65C)
            cold_mask = tb_matrix < -65.0
            if np.sum(cold_mask) > 50:
                y_idx, x_idx = np.nonzero(cold_mask)
                cy, cx = int(np.mean(y_idx)), int(np.mean(x_idx))
            else:
                cy, cx = N // 2, N // 2
        else:
            cy, cx = center

        # 1. Thermal convective activation: deeper cold tops = higher saliency
        # Linear scale: -40C (0.0) to -80C (1.0)
        saliency_raw = np.clip((-tb_matrix - 40.0) / 40.0, 0.0, 1.0)

        # 2. Radial proximity weighting to focus on the central vortex
        Y, X = np.ogrid[:N, :N]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        vortex_weight = np.exp(-(dist ** 2) / (2.0 * (70.0 ** 2)))

        saliency = saliency_raw * vortex_weight
        # Normalize to [0.0, 1.0]
        s_max = float(np.max(saliency))
        if s_max > 1e-4:
            saliency = saliency / s_max

        # 3. Morphological Indicators
        # Eyewall ring area (dist between 15 and 50 pixels)
        eyewall_zone = (dist >= 15.0) & (dist <= 50.0)
        eyewall_tb = tb_matrix[eyewall_zone]
        eyewall_min_tb = float(np.min(eyewall_tb)) if len(eyewall_tb) > 0 else -60.0

        # Eye pocket (dist < 15 pixels)
        eye_zone = dist < 15.0
        eye_tb = tb_matrix[eye_zone]
        eye_max_tb = float(np.max(eye_tb)) if len(eye_tb) > 0 else -40.0

        eye_contrast = max(0.0, eye_max_tb - eyewall_min_tb)

        # Axisymmetry score: circular variance of eyewall brightness temperature
        azimuthal_slices = 16
        slice_means = []
        theta_grid = np.arctan2(Y - cy, X - cx)
        for i in range(azimuthal_slices):
            t_low = -np.pi + i * (2.0 * np.pi / azimuthal_slices)
            t_high = t_low + (2.0 * np.pi / azimuthal_slices)
            mask_slice = eyewall_zone & (theta_grid >= t_low) & (theta_grid < t_high)
            if np.sum(mask_slice) > 10:
                slice_means.append(float(np.mean(tb_matrix[mask_slice])))

        if len(slice_means) >= 8:
            std_azimuth = float(np.std(slice_means))
            axisymmetry = max(0.0, min(1.0, 1.0 - (std_azimuth / 15.0)))
        else:
            axisymmetry = 0.5

        # Convective area coverage (TB < -65C) in km2 (assuming ~4km/pixel at NIO resolution)
        pixel_area_km2 = 16.0
        cold_core_pixels = int(np.sum(tb_matrix < -65.0))
        convective_area_km2 = cold_core_pixels * pixel_area_km2

        # Dvorak BD Slices Area Distribution
        bd_distribution = []
        total_pixels = float(N * N)
        for s in self.bd_slices:
            mask_s = (tb_matrix >= s["t_min"]) & (tb_matrix < s["t_max"])
            pct = round(float(np.sum(mask_s) / total_pixels) * 100.0, 2)
            bd_distribution.append({
                "code": s["code"],
                "label": s["label"],
                "color": s["color"],
                "coverage_pct": pct,
            })

        # Downsample saliency to 64x64 grid for low-bandwidth JSON transfer
        downsample_factor = max(1, N // 64)
        saliency_64 = saliency[::downsample_factor, ::downsample_factor]

        return {
            "vortex_center": [cy, cx],
            "axisymmetry_score": round(axisymmetry, 3),
            "eye_contrast_celsius": round(eye_contrast, 1),
            "eyewall_min_temp_celsius": round(eyewall_min_tb, 1),
            "eye_max_temp_celsius": round(eye_max_tb, 1),
            "deep_convective_area_km2": convective_area_km2,
            "dvorak_bd_distribution": bd_distribution,
            "saliency_grid_shape": list(saliency_64.shape),
            "saliency_matrix": np.round(saliency_64, 3).tolist(),
        }
