"""
Wind & Pressure Relationship Modeling, Deficit Calculation, and Scale Mapping
Implements official IMD pressure-wind equations and Rapid Intensification (RI) indicators.
"""
from typing import Optional, Dict
import numpy as np
import pandas as pd

# Standard ambient peripheral sea-level pressure in North Indian Ocean tropical marine boundary layer (hPa)
AMBIENT_PRESSURE_HPA = 1010.0

# IMD 7-Class Intensity Scale Mapping
IMD_GRADE_TO_INDEX: Dict[str, int] = {
    "D": 0,     # Depression (17 - 27 kt)
    "DD": 1,    # Deep Depression (28 - 33 kt)
    "CS": 2,    # Cyclonic Storm (34 - 47 kt)
    "SCS": 3,   # Severe Cyclonic Storm (48 - 63 kt)
    "VSCS": 4,  # Very Severe Cyclonic Storm (64 - 89 kt)
    "ESCS": 5,  # Extremely Severe Cyclonic Storm (90 - 119 kt)
    "SuCS": 6   # Super Cyclonic Storm (>= 120 kt)
}

INDEX_TO_IMD_GRADE: Dict[int, str] = {v: k for k, v in IMD_GRADE_TO_INDEX.items()}

def compute_pressure_deficit(central_pressure_hpa: float, p_ambient: float = AMBIENT_PRESSURE_HPA) -> float:
    """
    Computes pressure drop: Delta P = P_ambient - P_central.
    Higher deficit indicates a more intense cyclone.
    """
    if np.isnan(central_pressure_hpa) or central_pressure_hpa <= 0:
        return np.nan
    return float(max(0.0, p_ambient - central_pressure_hpa))

def estimate_pressure_from_wind_imd(
    wind_knots: float,
    p_ambient: float = AMBIENT_PRESSURE_HPA,
    c_const: float = 14.2
) -> float:
    """
    Estimates central pressure using the official IMD Mishra & Gupta empirical pressure-wind formula:
    V_max = c * sqrt(Delta P)  ==>  Delta P = (V_max / c)^2  ==>  P_min = P_ambient - (V_max / c)^2
    
    Provides physically consistent imputation for missing pressure observations in historical track records.
    """
    if np.isnan(wind_knots) or wind_knots <= 0:
        return np.nan
    delta_p = (wind_knots / c_const) ** 2
    estimated_p = p_ambient - delta_p
    return float(np.clip(estimated_p, 870.0, 1015.0))

def estimate_wind_from_pressure_imd(
    central_pressure_hpa: float,
    p_ambient: float = AMBIENT_PRESSURE_HPA,
    c_const: float = 14.2
) -> float:
    """
    Inverts the IMD pressure-wind formula: V_max = c * sqrt(P_ambient - P_min)
    """
    if np.isnan(central_pressure_hpa) or central_pressure_hpa >= p_ambient:
        return np.nan
    delta_p = max(0.0, p_ambient - central_pressure_hpa)
    wind_knots = c_const * np.sqrt(delta_p)
    return float(np.clip(wind_knots, 0.0, 180.0))

def map_wind_to_imd_grade(wind_knots: float) -> str:
    """
    Determines official IMD intensity grade classification from sustained wind speed (knots).
    """
    if np.isnan(wind_knots) or wind_knots < 17.0:
        return "LOW"  # Low Pressure Area / disturbance
    elif wind_knots < 28.0:
        return "D"
    elif wind_knots < 34.0:
        return "DD"
    elif wind_knots < 48.0:
        return "CS"
    elif wind_knots < 64.0:
        return "SCS"
    elif wind_knots < 90.0:
        return "VSCS"
    elif wind_knots < 120.0:
        return "ESCS"
    else:
        return "SuCS"

def map_imd_grade_to_index(grade_str: Optional[str]) -> int:
    """
    Maps IMD categorical grade string to ordinal target index [0 to 6].
    Returns -1 for unclassified disturbances.
    """
    if not grade_str or not isinstance(grade_str, str):
        return -1
    return IMD_GRADE_TO_INDEX.get(grade_str.strip().upper(), -1)

def map_index_to_imd_grade(idx: int) -> str:
    return INDEX_TO_IMD_GRADE.get(idx, "UNKNOWN")

def compute_rapid_intensification_flag(wind_t0: float, wind_t24: float, threshold_knots: float = 30.0) -> int:
    """
    Binary Rapid Intensification (RI) classification:
    1 if wind speed increases by at least 30 knots in a 24-hour window, 0 otherwise.
    """
    if np.isnan(wind_t0) or np.isnan(wind_t24):
        return -1
    return 1 if (wind_t24 - wind_t0) >= threshold_knots else 0
