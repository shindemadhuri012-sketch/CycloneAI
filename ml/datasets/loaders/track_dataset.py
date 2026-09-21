"""
PyTorch-compatible dataset loader for sequential cyclone track trajectory prediction.
Constructs sliding temporal windows of past coordinates and intensities to forecast future steps.
"""
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

class CycloneTrackDataset:
    """
    Sequence dataset for cyclone track trajectory forecasting.
    Input: Past k time steps of (latitude, longitude, wind_speed, pressure).
    Target: Future m time steps of (latitude, longitude).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        history_steps: int = 4,   # e.g., 4 steps * 6h = 24h past trajectory
        forecast_steps: int = 4,  # e.g., +6h, +12h, +18h, +24h future trajectory
        feature_cols: List[str] = None,
        target_cols: List[str] = None
    ):
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.feature_cols = feature_cols or ["LAT", "LON", "WMO_WIND", "WMO_PRES"]
        self.target_cols = target_cols or ["LAT", "LON"]

        self.samples = []
        self._build_sequences(df)

    def _build_sequences(self, df: pd.DataFrame):
        """
        Groups data by storm SID to ensure sequence windows NEVER cross different cyclones.
        """
        for sid, storm_df in df.groupby("SID"):
            # Ensure chronological order
            storm_df = storm_df.sort_values("ISO_TIME")
            features = storm_df[self.feature_cols].values
            targets = storm_df[self.target_cols].values

            total_len = len(features)
            required_len = self.history_steps + self.forecast_steps

            if total_len >= required_len:
                for i in range(total_len - required_len + 1):
                    x_seq = features[i : i + self.history_steps]
                    y_seq = targets[i + self.history_steps : i + required_len]

                    # Filter out sequences containing NaN in lat/lon
                    if not np.isnan(x_seq[:, :2]).any() and not np.isnan(y_seq).any():
                        self.samples.append({
                            "sid": sid,
                            "input_seq": x_seq.astype(np.float32),
                            "target_seq": y_seq.astype(np.float32)
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]
        return {
            "sid": sample["sid"],
            "features": sample["input_seq"],
            "targets": sample["target_seq"]
        }
