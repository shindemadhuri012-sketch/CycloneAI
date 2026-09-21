"""
Storm-Level & Season-Level Data Splitting Strategy
Prevents data leakage by ensuring observations from the same cyclone NEVER appear in both train and test sets.
"""
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

def split_by_storm_id(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into Train, Validation, and Test partitions strictly by unique Storm Identifier (SID).
    Zero temporal or spatial leakage between partitions.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    unique_sids = df["SID"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled_sids = rng.permutation(unique_sids)

    n_total = len(shuffled_sids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_sids = set(shuffled_sids[:n_train])
    val_sids = set(shuffled_sids[n_train : n_train + n_val])
    test_sids = set(shuffled_sids[n_train + n_val:])

    train_df = df[df["SID"].isin(train_sids)].copy()
    val_df = df[df["SID"].isin(val_sids)].copy()
    test_df = df[df["SID"].isin(test_sids)].copy()

    # Integrity verification
    overlap_train_val = train_sids.intersection(val_sids)
    overlap_train_test = train_sids.intersection(test_sids)
    overlap_val_test = val_sids.intersection(test_sids)

    assert len(overlap_train_val) == 0, f"Data leakage between Train and Val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Data leakage between Train and Test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Data leakage between Val and Test: {overlap_val_test}"

    return train_df, val_df, test_df

def split_by_season(
    df: pd.DataFrame,
    train_until_season: int = 2018,
    val_seasons: List[int] = [2019, 2020, 2021],
    test_from_season: int = 2022
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data chronologically by cyclone season (Year).
    This strictly simulates real-world operational forecasting where models are trained
    on past historical seasons and evaluated exclusively on future unseen cyclone seasons.
    """
    train_df = df[df["SEASON"] <= train_until_season].copy()
    val_df = df[df["SEASON"].isin(val_seasons)].copy()
    test_df = df[df["SEASON"] >= test_from_season].copy()

    return train_df, val_df, test_df

def get_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    return {
        "train_records": len(train_df),
        "train_storms": train_df["SID"].nunique(),
        "val_records": len(val_df),
        "val_storms": val_df["SID"].nunique(),
        "test_records": len(test_df),
        "test_storms": test_df["SID"].nunique(),
    }
