"""
Comprehensive Preprocessing Pipeline Validation & Quality Assurance Suite
Verifies feature completeness, physical bounds, absence of NaNs, and 0% data leakage.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from ml.preprocessing.track.sequence_builder import generate_storm_trajectory_sequences
from ml.preprocessing.satellite.calibration import calibrate_satellite_tensor
from ml.preprocessing.satellite.augmentations import apply_physics_preserving_augmentation

def run_validation():
    print("=" * 75)
    print("CYCLONEAI PREPROCESSING VALIDATION & INTEGRITY SUITE")
    print("=" * 75)

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"

    # 1. Check artifact existence
    files_to_check = [
        "tracks_processed.csv",
        "train_manifest.csv",
        "val_manifest.csv",
        "test_manifest.csv"
    ]
    print("[CHECK 1/5] Checking generated dataset artifacts...")
    for f in files_to_check:
        p = processed_dir / f
        assert p.exists(), f"Missing expected artifact: {p}"
        print(f"  [OK] Found artifact: {f} ({p.stat().st_size / 1024:.1f} KB)")

    # 2. Check engineered columns
    print("\n[CHECK 2/5] Checking feature schema & absence of corrupted NaNs...")
    df = pd.read_csv(processed_dir / "tracks_processed.csv")
    required_cols = [
        "SID", "ISO_TIME", "LAT", "LON", "DELTA_LAT", "DELTA_LON",
        "DIST_KM", "FORWARD_SPEED_KMH", "BEARING_DEG", "EFFECTIVE_WIND_KT",
        "PRESSURE_DEFICIT", "CENTRAL_PRES_HPA", "IMD_GRADE", "IMD_GRADE_INDEX"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        nan_count = df[col].isna().sum()
        if col in ["LAT", "LON", "DELTA_LAT", "DELTA_LON", "FORWARD_SPEED_KMH", "BEARING_DEG", "CENTRAL_PRES_HPA"]:
            assert nan_count == 0, f"Unexpected NaN values in core feature {col}: {nan_count}"
        print(f"  [OK] Feature '{col}': Present (NaN count: {nan_count})")

    # 3. Check physical value bounds
    print("\n[CHECK 3/5] Verifying physical bounds & meteorological limits...")
    assert (df["FORWARD_SPEED_KMH"] >= 0.0).all() and (df["FORWARD_SPEED_KMH"] <= 120.0).all(), "Forward speed out of physical bounds!"
    assert (df["BEARING_DEG"] >= 0.0).all() and (df["BEARING_DEG"] <= 360.0).all(), "Bearing angle out of bounds!"
    assert (df["CENTRAL_PRES_HPA"] >= 870.0).all() and (df["CENTRAL_PRES_HPA"] <= 1015.0).all(), "Central pressure out of physical bounds!"
    assert (df["PRESSURE_DEFICIT"] >= 0.0).all(), "Pressure deficit cannot be negative!"
    print("  [OK] Forward translation speed bounds: [0, 120] km/h (PASS)")
    print("  [OK] Heading compass bearing bounds:   [0, 360] degrees (PASS)")
    print("  [OK] Central barometric pressure:      [870, 1015] hPa (PASS)")
    print("  [OK] Pressure deficit bounds:          >= 0.0 hPa (PASS)")

    # 4. Strict Data Leakage Audit
    print("\n[CHECK 4/5] Auditing storm-level dataset manifests for data leakage...")
    train_df = pd.read_csv(processed_dir / "train_manifest.csv")
    val_df = pd.read_csv(processed_dir / "val_manifest.csv")
    test_df = pd.read_csv(processed_dir / "test_manifest.csv")

    train_storms = set(train_df["SID"].unique())
    val_storms = set(val_df["SID"].unique())
    test_storms = set(test_df["SID"].unique())

    leakage_train_val = train_storms.intersection(val_storms)
    leakage_train_test = train_storms.intersection(test_storms)
    leakage_val_test = val_storms.intersection(test_storms)

    assert len(leakage_train_val) == 0, f"DATA LEAKAGE DETECTED between Train and Val: {leakage_train_val}"
    assert len(leakage_train_test) == 0, f"DATA LEAKAGE DETECTED between Train and Test: {leakage_train_test}"
    assert len(leakage_val_test) == 0, f"DATA LEAKAGE DETECTED between Val and Test: {leakage_val_test}"

    print(f"  [OK] Unique storms - Train: {len(train_storms)}, Val: {len(val_storms)}, Test: {len(test_storms)}")
    print("  [OK] Inter-split storm overlap: EXACTLY ZERO STORMS (0% Data Leakage)")

    # 5. Test sequence generation & satellite transformation modules
    print("\n[CHECK 5/5] Testing trajectory sliding windows and satellite calibrations...")
    seq_samples = generate_storm_trajectory_sequences(train_df, history_steps=4, forecast_steps=4)
    print(f"  [OK] Trajectory sequence windows generated: {len(seq_samples):,} pairs")
    if len(seq_samples) > 0:
        s0 = seq_samples[0]
        assert s0["input_coords"].shape == (4, 2)
        assert s0["input_kinematics"].shape == (4, 4)
        assert s0["target_coords"].shape == (4, 2)
        print(f"  [OK] Input coords shape: {s0['input_coords'].shape}, Target coords: {s0['target_coords'].shape}")

    # Test satellite calibration on sample physical array
    dummy_chip = np.random.uniform(180.0, 310.0, size=(201, 201, 4)).astype(np.float32)
    calibrated = calibrate_satellite_tensor(dummy_chip)
    assert calibrated.shape == (201, 201, 4)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    augmented = apply_physics_preserving_augmentation(calibrated, rotation_k=1)
    assert augmented.shape == (201, 201, 4)
    print("  [OK] Satellite radiometric calibration & physics-preserving rotation verified.")

    print("\n" + "=" * 75)
    print("[FINAL RESULT] ALL 5 PREPROCESSING VERIFICATION CHECKS PASSED (100% SUCCESS)")
    print("=" * 75)

if __name__ == "__main__":
    run_validation()
