"""
Natural Language Meteorological Narrator Engine.
Synthesizes local feature attributions, physical thermodynamic balance,
kinematic trajectory vectors, and Rapid Intensification risk into standardized
IMD/RSMC-style synoptic diagnostic briefings for disaster management decision-support.
"""
from typing import Dict, List, Optional, Any


class MeteorologicalNarrator:
    """
    Translates model attribution vectors and physical features into
    clear, authoritative natural language operational diagnostic advisories.
    """

    @staticmethod
    def generate_diagnostic_narrative(
        category_name: str,
        category_code: str,
        confidence_pct: float,
        top_drivers: List[Dict[str, Any]],
        current_wind_kt: float,
        central_pres_hpa: float,
        predicted_wind_24h_kt: float,
        ri_alert: bool,
        subbasin: str = "BB",
    ) -> str:
        """
        Synthesizes a structured operational briefing from attribution results.
        """
        basin_full = "Bay of Bengal" if "BB" in subbasin.upper() else "Arabian Sea"

        # 1. Primary intensifier and inhibitor from attributions
        top_pos = [d for d in top_drivers if d.get("direction") == "positive" or d.get("contribution", 0) > 0]
        top_neg = [d for d in top_drivers if d.get("direction") == "negative" or d.get("contribution", 0) < 0]

        primary_pos = top_pos[0]["feature"].replace("_", " ") if top_pos else "core pressure deficit"
        primary_neg = top_neg[0]["feature"].replace("_", " ") if top_neg else "environmental forward shear"

        # 2. Delta intensity over 24 hours
        d_v24 = predicted_wind_24h_kt - current_wind_kt
        if d_v24 >= 25.0:
            trend_str = f"rapidly intensifying with projected +{d_v24:.1f} kt surge"
        elif d_v24 >= 10.0:
            trend_str = f"steadily intensifying by +{d_v24:.1f} kt"
        elif d_v24 >= -5.0:
            trend_str = "maintaining a near steady-state intensity"
        else:
            trend_str = f"weakening by {abs(d_v24):.1f} kt due to atmospheric/land interactions"

        # 3. RI status statement
        if ri_alert:
            ri_statement = (
                "CRITICAL WARNING: Atmospheric conditions favor Rapid Intensification (RI) "
                "within the next 24 hours (threshold >= 30 kt/24h). Accelerated coastal preparedness advised."
            )
        else:
            ri_statement = (
                "Normal intensification progression expected; explosive Rapid Intensification threshold is not triggered."
            )

        # 4. Construct coherent diagnostic paragraph
        narrative = (
            f"Vortex over the {basin_full} is assessed as a {category_name} ({category_code}) "
            f"with {confidence_pct:.1f}% confidence. "
            f"Observed central pressure is {central_pres_hpa:.1f} hPa with peak sustained winds of {current_wind_kt:.1f} knots. "
            f"Machine learning attribution demonstrates that the primary physical driver elevating system severity is {primary_pos}, "
            f"while {primary_neg} acts as the primary stabilizing factor. "
            f"Over the next 24 hours, the vortex is projected to be {trend_str}, reaching {predicted_wind_24h_kt:.1f} knots. "
            f"{ri_statement}"
        )

        return narrative

    @staticmethod
    def generate_track_narrative(
        current_lat: float,
        current_lon: float,
        pred_lat_24h: float,
        pred_lon_24h: float,
        forward_speed_kmh: float,
        bearing_deg: float,
        subbasin: str = "BB",
    ) -> str:
        """
        Formulates track trajectory steering narrative.
        """
        # Compass quadrant
        quadrants = ["North", "North-Northeast", "Northeast", "East-Northeast",
                     "East", "East-Southeast", "Southeast", "South-Southeast",
                     "South", "South-Southwest", "Southwest", "West-Southwest",
                     "West", "West-Northwest", "Northwest", "North-Northwest"]
        idx = int((bearing_deg + 11.25) / 22.5) % 16
        dir_str = quadrants[idx]

        dlat = pred_lat_24h - current_lat
        dlon = pred_lon_24h - current_lon

        recurvature = "exhibits poleward recurvature" if dlat > 2.0 and dlon > -1.0 else "follows a stable steering trajectory"

        return (
            f"System is translating {dir_str} (bearing {bearing_deg:.0f}°) at {forward_speed_kmh:.1f} km/h. "
            f"Trajectory forecast {recurvature} under regional mid-tropospheric ridge steering, "
            f"reaching ({pred_lat_24h:.2f}°N, {pred_lon_24h:.2f}°E) at +24h."
        )
