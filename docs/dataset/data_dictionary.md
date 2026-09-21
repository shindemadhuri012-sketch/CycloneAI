# CycloneAI Data Dictionary & Field Specifications

This document formally specifies data schemas, data types, physical units, coordinate reference systems, and missing-value handling protocols for all datasets incorporated into CycloneAI.

---

## 1. NOAA NCEI IBTrACS (v04r01) Best-Track Schema

Source File: `ibtracs.NI.list.v04r01.csv` (North Indian Ocean Basin Subset)

| Field Name | Data Type | Physical Unit | Description / Constraints | Missing Value Representation |
| :--- | :--- | :--- | :--- | :--- |
| `SID` | String | Identifier | Unique 13-character storm identifier (e.g. `2020136N10090`). Formed by Year, Day of Year, Initial Latitude/Longitude. | None (Primary key) |
| `SEASON` | Integer | Year | Year storm was active (e.g. `2020`). | None |
| `NUMBER` | Integer | Count | Consecutive storm number of the season in the basin. | None |
| `BASIN` | String | Code | Ocean basin code (`NI` for North Indian Ocean). | None |
| `SUBBASIN` | String | Code | Sub-basin code: `BB` (Bay of Bengal) or `AS` (Arabian Sea). | Empty / `MM` |
| `NAME` | String | Text | Official assigned storm name (e.g., `AMPHAN`, `FANI`, `BIPARJOY`, `NOT_NAMED`). | `NOT_NAMED` / Empty |
| `ISO_TIME` | String | UTC Timestamp | Observation timestamp in ISO 8601 format: `YYYY-MM-DD hh:mm:ss` (typically 3-hourly: 00, 03, 06, 09, 12, 15, 18, 21 UTC). | None |
| `NATURE` | String | Code | System classification: `TS` (Tropical Storm), `NR` (Not Reported), `ET` (Extratropical), `DS` (Disturbance). | `NR` |
| `LAT` | Float | Decimal Degrees | Latitude of storm center ($-90.00$ to $+90.00$; North is positive). | Empty / NaN |
| `LON` | Float | Decimal Degrees | Longitude of storm center ($-180.00$ to $+180.00$; East is positive). | Empty / NaN |
| `WMO_WIND` | Float | Knots (kt) | World Meteorological Organization consensus maximum sustained wind speed. 1 knot = 1.852 km/h. | Empty / NaN |
| `WMO_PRES` | Float | Millibars / hPa | WMO consensus minimum central sea-level pressure. | Empty / NaN |
| `NEWDELHI_LAT` | Float | Decimal Degrees | Official storm center latitude determined by IMD RSMC New Delhi. | Empty / NaN |
| `NEWDELHI_LON` | Float | Decimal Degrees | Official storm center longitude determined by IMD RSMC New Delhi. | Empty / NaN |
| `NEWDELHI_WIND`| Float | Knots (kt) | Official IMD 3-minute average maximum sustained surface wind speed. | Empty / NaN |
| `NEWDELHI_PRES`| Float | Millibars / hPa | Official IMD central barometric pressure. | Empty / NaN |
| `NEWDELHI_GRADE`| String | Category Code | Official IMD intensity grade: `D`, `DD`, `CS`, `SCS`, `VSCS`, `ESCS`, `SuCS`. | Empty / NaN |
| `USA_WIND` | Float | Knots (kt) | Official JTWC (Joint Typhoon Warning Center) 1-minute sustained wind speed. | Empty / NaN |
| `USA_PRES` | Float | Millibars / hPa | Official JTWC estimated central pressure. | Empty / NaN |

---

## 2. TCIR Multi-Spectral Satellite Image Schema

Source File: `TCIR_*.h5` (HDF5 Benchmark Container)

### A. Matrix Tensor (`/matrix`)
- **Dimensions**: $[N, 201, 201, 4]$
  - $N$: Number of observation samples.
  - Height / Width: $201 \times 201$ pixels centered on the cyclone eye.
  - Spatial Resolution: Approximately $4\text{ km} \times 4\text{ km}$ per pixel ($\approx 800\text{ km} \times 800\text{ km}$ synoptic footprint).
  - Channels ($C=4$):

| Channel Index | Channel Name | Spectral Wavelength | Physical Variable | Typical Physical Range | Preprocessing Normalization |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | Infrared (IR) | $10.5 - 11.5\ \mu\text{m}$ | Brightness Temperature | $170\text{ K} - 320\text{ K}$ | $T_{\text{norm}} = \frac{T - 170.0}{320.0 - 170.0}$ |
| `1` | Water Vapor (WV) | $6.5 - 7.0\ \mu\text{m}$ | Upper-Tropospheric Moisture | $190\text{ K} - 280\text{ K}$ | $T_{\text{norm}} = \frac{T - 190.0}{280.0 - 190.0}$ |
| `2` | Visible (VIS) | $0.55 - 0.75\ \mu\text{m}$ | Surface / Cloud Albedo | $0.0 - 1.0\ (0\% - 100\%)$ | Clamped to $[0.0, 1.0]$; 0 at night |
| `3` | Passive Microwave (PMW) | $85 - 92\ \text{GHz}$ | Polarized Brightness Temp | $150\text{ K} - 300\text{ K}$ | $T_{\text{norm}} = \frac{T - 150.0}{300.0 - 150.0}$ |

### B. Info Metadata (`/info`)
Stored as a Pandas DataFrame inside the HDF5 group:
- `id`: Storm tracking identifier string corresponding to IBTrACS `SID`.
- `time`: UTC timestamp string (`YYYYMMDDhhmm`).
- `lat`: Centered storm center latitude.
- `lon`: Centered storm center longitude.
- `vmax`: Best-track maximum sustained wind speed (knots).
- `mslp`: Minimum sea-level pressure (hPa).

---

## 3. Physical Units & Coordinate Standards

1. **Wind Speed**: Standardized in **Knots (kt)**.
   $$\text{Speed in km/h} = \text{Speed in knots} \times 1.852$$
   $$\text{Speed in m/s} = \text{Speed in knots} \times 0.5144$$
   *Note on Averaging Times*: IMD uses a **3-minute** sustained wind average; JTWC uses a **1-minute** average; WMO consensus uses a **10-minute** average. CycloneAI models explicitly target the **IMD 3-minute standard** for Indian operations, with JTWC conversion formulas documented in Phase 3.
2. **Atmospheric Pressure**: Standardized in **Hectopascals (hPa)** (equivalent to millibars, mbar). Standard sea-level pressure is $1013.25\text{ hPa}$. Intense cyclones drop below $950\text{ hPa}$.
3. **Geospatial Coordinates**:
   - Coordinate Reference System (CRS): **WGS 84 (EPSG:4326)**.
   - Latitude: Positive values denote North ($^\circ\text{N}$); negative denote South ($^\circ\text{S}$).
   - Longitude: Positive values denote East ($^\circ\text{E}$); negative denote West ($^\circ\text{W}$).
4. **Timestamps**: All timestamps strictly adhere to **Universal Coordinated Time (UTC)** in ISO 8601 format (`YYYY-MM-DDTHH:MM:SSZ`). Indian Standard Time (IST) is derived as $\text{IST} = \text{UTC} + 5\text{ hours } 30\text{ minutes}$.
