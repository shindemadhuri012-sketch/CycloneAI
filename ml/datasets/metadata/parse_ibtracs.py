"""
Parser and metadata standardizer for NOAA NCEI IBTrACS v04r01.
Handles the special 2-line header format of IBTrACS CSVs and extracts IMD columns.
"""
import pandas as pd
from pathlib import Path
from typing import Optional

def parse_ibtracs_csv(csv_path: Path, basin_filter: Optional[str] = "NI") -> pd.DataFrame:
    """
    Parses an IBTrACS CSV file, removing the unit row and casting columns to proper types.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"IBTrACS file not found at: {csv_path}")

    # Row 0 contains column names; Row 1 contains physical units description
    df = pd.read_csv(
        csv_path,
        skiprows=[1],  # Skip the units row
        low_memory=False,
        na_values=[" ", "NOT_NAMED", "MM", ""]
    )

    # Core required columns
    columns_to_keep = [
        "SID", "SEASON", "NUMBER", "BASIN", "SUBBASIN", "NAME", "ISO_TIME",
        "NATURE", "LAT", "LON", "WMO_WIND", "WMO_PRES",
        "NEWDELHI_LAT", "NEWDELHI_LON", "NEWDELHI_WIND", "NEWDELHI_PRES", "NEWDELHI_GRADE",
        "USA_WIND", "USA_PRES"
    ]

    # Filter to columns that exist in the file
    available_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[available_cols].copy()

    # Convert timestamps
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    # Numeric casting
    numeric_cols = [
        "SEASON", "NUMBER", "LAT", "LON", "WMO_WIND", "WMO_PRES",
        "NEWDELHI_LAT", "NEWDELHI_LON", "NEWDELHI_WIND", "NEWDELHI_PRES",
        "USA_WIND", "USA_PRES"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter by basin if requested
    if basin_filter and "BASIN" in df.columns:
        df = df[df["BASIN"] == basin_filter].copy()

    # Sort deterministically by storm identifier and timestamp
    df.sort_values(by=["SID", "ISO_TIME"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    default_csv = project_root / "data" / "external" / "ibtracs.NI.list.v04r01.csv"
    if default_csv.exists():
        df_parsed = parse_ibtracs_csv(default_csv)
        print(f"[INFO] Successfully parsed {len(df_parsed)} track rows across {df_parsed['SID'].nunique()} unique storms.")
        print(df_parsed[["SID", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "NEWDELHI_WIND"]].head())
    else:
        print(f"[INFO] Download file first with download_ibtracs.py. Expected at: {default_csv}")
