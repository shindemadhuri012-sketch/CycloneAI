/**
 * Meteorological Constants & Scale Definitions
 */

export const BASINS = [
  { id: 'BoB', name: 'Bay of Bengal', center: [15.0, 88.0] },
  { id: 'AS', name: 'Arabian Sea', center: [16.0, 68.0] },
  { id: 'NIO', name: 'North Indian Ocean (All)', center: [15.0, 78.0] },
];

export const IMD_SCALE = [
  { code: 'D', name: 'Depression', minKnots: 17, maxKnots: 27 },
  { code: 'DD', name: 'Deep Depression', minKnots: 28, maxKnots: 33 },
  { code: 'CS', name: 'Cyclonic Storm', minKnots: 34, maxKnots: 47 },
  { code: 'SCS', name: 'Severe Cyclonic Storm', minKnots: 48, maxKnots: 63 },
  { code: 'VSCS', name: 'Very Severe Cyclonic Storm', minKnots: 64, maxKnots: 89 },
  { code: 'ESCS', name: 'Extremely Severe Cyclonic Storm', minKnots: 90, maxKnots: 119 },
  { code: 'SuCS', name: 'Super Cyclonic Storm', minKnots: 120, maxKnots: 250 },
];

export const SATELLITE_SENSORS = [
  { id: 'INSAT-3D', name: 'INSAT-3D (ISRO)', agency: 'ISRO', orbitalSlot: '74°E' },
  { id: 'INSAT-3DR', name: 'INSAT-3DR (ISRO)', agency: 'ISRO', orbitalSlot: '82°E' },
  { id: 'HIMAWARI', name: 'Himawari-8/9', agency: 'JMA', orbitalSlot: '140.7°E' },
  { id: 'GOES-16', name: 'GOES-East', agency: 'NOAA', orbitalSlot: '75.2°W' },
];
