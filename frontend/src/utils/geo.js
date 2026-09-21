/**
 * Geospatial utility helpers for tropical cyclone coordinate manipulation.
 */

export function formatCoordinate(lat, lng) {
  if (lat === null || lat === undefined || lng === null || lng === undefined) {
    return 'Coordinates N/A';
  }
  const latDir = lat >= 0 ? 'N' : 'S';
  const lngDir = lng >= 0 ? 'E' : 'W';
  return `${Math.abs(lat).toFixed(2)}°${latDir}, ${Math.abs(lng).toFixed(2)}°${lngDir}`;
}

export function knotsToKmph(knots) {
  if (!knots && knots !== 0) return null;
  return Math.round(knots * 1.852);
}
