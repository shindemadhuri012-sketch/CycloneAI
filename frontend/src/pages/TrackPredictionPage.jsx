import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Polygon, useMap } from 'react-leaflet';
import L from 'leaflet';
import { 
  Navigation, 
  MapPin, 
  Clock, 
  Compass, 
  Layers, 
  AlertTriangle, 
  RefreshCw, 
  Cpu, 
  Info,
  CheckCircle2,
  ShieldCheck
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import { predictTrack, fetchStormPresets } from '../services/api';

// Fix Leaflet default marker icon path in bundled setups
try {
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  });
} catch (e) {
  console.warn('Leaflet icon initialization notice:', e);
}

// Helper component to smoothly re-center map when storm coordinates change
function ChangeMapView({ coords }) {
  const map = useMap();
  useEffect(() => {
    if (coords && typeof coords[0] === 'number' && typeof coords[1] === 'number') {
      map.setView(coords, map.getZoom());
    }
  }, [coords, map]);
  return null;
}

export default function TrackPredictionPage() {
  const [presets, setPresets] = useState([]);
  const [selectedStorm, setSelectedStorm] = useState('FANI');
  const [isQuerying, setIsQuerying] = useState(false);
  const [apiResponse, setApiResponse] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Active observation parameters
  const [lat, setLat] = useState(16.0);
  const [lon, setLon] = useState(86.5);
  const [centralPres, setCentralPres] = useState(970.0);
  const [currentWind, setCurrentWind] = useState(85.0);
  const [forwardSpeed, setForwardSpeed] = useState(16.0);
  const [bearing, setBearing] = useState(320.0);

  const defaultCenter = [16.0, 85.0];
  const defaultZoom = 5;

  useEffect(() => {
    let isMounted = true;
    async function load() {
      try {
        const p = await fetchStormPresets();
        if (!isMounted) return;
        if (p && Array.isArray(p) && p.length > 0) {
          setPresets(p);
          applyPreset(p[0]);
        } else {
          handlePredict();
        }
      } catch (err) {
        console.error('Error fetching presets:', err);
        if (isMounted) handlePredict();
      }
    }
    load();
    return () => { isMounted = false; };
  }, []);

  const applyPreset = (preset) => {
    if (!preset || !preset.observation) return;
    setSelectedStorm(preset.name || 'CYCLONE');
    const obs = preset.observation;
    setLat(typeof obs.lat === 'number' ? obs.lat : 16.0);
    setLon(typeof obs.lon === 'number' ? obs.lon : 86.5);
    setCentralPres(typeof obs.central_pres === 'number' ? obs.central_pres : 970.0);
    setCurrentWind(typeof obs.current_wind_speed_knots === 'number' ? obs.current_wind_speed_knots : 85.0);
    setForwardSpeed(typeof obs.forward_speed === 'number' ? obs.forward_speed : 16.0);
    setBearing(typeof obs.bearing === 'number' ? obs.bearing : 320.0);
    runTrackForecast(obs, preset.name);
  };

  const handlePredict = () => {
    runTrackForecast({
      lat,
      lon,
      central_pres: centralPres,
      current_wind_speed_knots: currentWind,
      forward_speed: forwardSpeed,
      bearing
    }, selectedStorm);
  };

  const runTrackForecast = async (params, stormName = selectedStorm) => {
    setIsQuerying(true);
    setErrorMessage(null);
    try {
      const res = await predictTrack({
        storm_id: stormName || selectedStorm,
        lat: typeof params.lat === 'number' ? params.lat : 16.0,
        lon: typeof params.lon === 'number' ? params.lon : 86.5,
        central_pres: typeof params.central_pres === 'number' ? params.central_pres : 970.0,
        current_wind_speed_knots: typeof params.current_wind_speed_knots === 'number' ? params.current_wind_speed_knots : 85.0,
        forward_speed: typeof params.forward_speed === 'number' ? params.forward_speed : 16.0,
        bearing: typeof params.bearing === 'number' ? params.bearing : 320.0,
        month: 5,
        subbasin: 'BB'
      });
      if (res && res.status === 'error') {
        setErrorMessage(res.message || 'Track prediction error returned from server.');
      }
      setApiResponse(res);
    } catch (e) {
      console.error('Track forecast failed:', e);
      setErrorMessage(e.message || 'Failed to connect to track forecast service.');
    } finally {
      setIsQuerying(false);
    }
  };

  const waypoints = (apiResponse && Array.isArray(apiResponse.predicted_track)) ? apiResponse.predicted_track : [];
  const cones = (apiResponse && Array.isArray(apiResponse.track_error_cone)) ? apiResponse.track_error_cone : [];

  // Polyline positions: starting with current position, followed by predicted waypoints
  const validCurrentCoord = (typeof lat === 'number' && typeof lon === 'number' && !isNaN(lat) && !isNaN(lon))
    ? [lat, lon]
    : defaultCenter;

  const validWaypointsCoords = waypoints
    .filter(w => typeof w.latitude === 'number' && typeof w.longitude === 'number' && !isNaN(w.latitude) && !isNaN(w.longitude))
    .map(w => [w.latitude, w.longitude]);

  const polylinePositions = [validCurrentCoord, ...validWaypointsCoords];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
            <Navigation className="w-4 h-4" />
            <span>SPATIO-TEMPORAL TRAJECTORY FORECASTING</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Cyclone Track & Uncertainty Cones</h1>
          <p className="text-sm text-slate-400">
            Multi-horizon trajectory forecasting (+6h, +12h, +24h, +48h) via incremental displacement vectors with 75% empirical uncertainty cones.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <StatusBadge 
            status={apiResponse?.model_connected !== false ? 'operational' : 'offline'} 
            label={apiResponse?.model_connected !== false ? 'Track Model: Active (+17.3% Skill)' : 'Model Disconnected'} 
          />
        </div>
      </div>

      {/* Preset Selector Banner */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-cyan-400 font-bold">REAL STORM PRESETS:</span>
          <div className="flex flex-wrap gap-1.5">
            {presets.map((p) => (
              <button
                key={p.sid || p.name}
                onClick={() => applyPreset(p)}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all cursor-pointer ${
                  selectedStorm === p.name
                    ? 'bg-cyan-600 text-white font-bold shadow-md shadow-cyan-600/30'
                    : 'bg-slate-900/90 text-slate-300 hover:bg-slate-800 border border-slate-700/60'
                }`}
              >
                {p.name} {p.season ? `(${p.season})` : ''}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handlePredict}
          disabled={isQuerying}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-cyan-600/20 cursor-pointer transition-all"
        >
          {isQuerying ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Cpu className="w-3.5 h-3.5" />}
          <span>Re-Forecast Trajectory</span>
        </button>
      </div>

      {/* Error / Alert notice if any */}
      {errorMessage && (
        <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Main Grid: Leaflet Map (8 cols) & Waypoint Table (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Map Viewport (8 cols) */}
        <div className="lg:col-span-8 space-y-3">
          <div className="glass-panel p-2 rounded-2xl border border-slate-800 relative h-[560px] overflow-hidden flex flex-col">
            <div className="flex-1 w-full rounded-xl overflow-hidden relative">
              <MapContainer
                center={validCurrentCoord}
                zoom={defaultZoom}
                scrollWheelZoom={true}
                className="w-full h-full"
              >
                <ChangeMapView coords={validCurrentCoord} />
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* Uncertainty Cone Polygon Rings */}
                {cones.map((c, idx) => {
                  const polygonCoords = c.polygon_coordinates || c.polygon_coords;
                  if (!polygonCoords || !Array.isArray(polygonCoords) || polygonCoords.length === 0) {
                    return null;
                  }
                  return (
                    <Polygon
                      key={`cone-${idx}`}
                      positions={polygonCoords}
                      pathOptions={{
                        color: '#06b6d4',
                        fillColor: '#0891b2',
                        fillOpacity: 0.12,
                        weight: 1.5,
                        dashArray: '4, 4'
                      }}
                    />
                  );
                })}

                {/* Trajectory Polyline */}
                {polylinePositions.length > 1 && (
                  <Polyline
                    positions={polylinePositions}
                    pathOptions={{ color: '#06b6d4', weight: 3, opacity: 0.9 }}
                  />
                )}

                {/* Current Position Marker */}
                <Marker position={validCurrentCoord}>
                  <Popup>
                    <div className="text-xs space-y-1">
                      <strong className="text-cyan-700 block">CURRENT EYE FIX (0h)</strong>
                      <div>{selectedStorm} Center</div>
                      <div>Lat: {lat}°N, Lon: {lon}°E</div>
                      <div>Central Pres: {centralPres} hPa</div>
                      <div>Translation: {forwardSpeed} km/h @ {bearing}°</div>
                    </div>
                  </Popup>
                </Marker>

                {/* Predicted Waypoint Markers */}
                {waypoints.map((w, idx) => {
                  if (typeof w.latitude !== 'number' || typeof w.longitude !== 'number' || isNaN(w.latitude) || isNaN(w.longitude)) {
                    return null;
                  }
                  return (
                    <Marker key={`wp-${idx}`} position={[w.latitude, w.longitude]}>
                      <Popup>
                        <div className="text-xs space-y-1">
                          <strong className="text-cyan-700 block">WAYPOINT {w.horizon}</strong>
                          <div>Lat: {w.latitude}°N, Lon: {w.longitude}°E</div>
                          <div>Forward Speed: {w.forward_speed_kmh} km/h</div>
                          <div>Bearing: {w.bearing_deg}°</div>
                          <div className="text-slate-500 font-mono">Displacement Vector Output</div>
                        </div>
                      </Popup>
                    </Marker>
                  );
                })}
              </MapContainer>
            </div>
          </div>
        </div>

        {/* Right Column: Waypoints Table & Kinematic Controls (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Waypoint Chronology Table */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <Clock className="w-4 h-4 text-cyan-400" />
                <span>Forecast Waypoints</span>
              </h2>
              <span className="text-xs font-mono text-cyan-400">+48h Lead Time</span>
            </div>

            <div className="space-y-2.5">
              {/* 0h Current Fix */}
              <div className="p-3 rounded-xl bg-slate-900/90 border border-cyan-800/40 space-y-1">
                <div className="flex items-center justify-between text-xs font-bold text-cyan-400">
                  <span>0h (Current Fix)</span>
                  <span className="font-mono text-white">{lat}°N, {lon}°E</span>
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                  <span>Translation:</span>
                  <span>{forwardSpeed} km/h @ {bearing}°</span>
                </div>
              </div>

              {/* Waypoints List */}
              {waypoints.map((w, i) => (
                <div key={i} className="p-3 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-xs font-bold text-slate-200">
                    <span className="text-cyan-400 font-mono">{w.horizon}</span>
                    <span className="font-mono">{w.latitude}°N, {w.longitude}°E</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span>Forward Speed:</span>
                    <span>{w.forward_speed_kmh} km/h</span>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                    <span>Heading:</span>
                    <span>{w.bearing_deg}°</span>
                  </div>
                </div>
              ))}

              {waypoints.length === 0 && (
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-xs text-slate-400 text-center">
                  {isQuerying ? 'Computing trajectory waypoints...' : 'No trajectory waypoints available. Select a storm preset above.'}
                </div>
              )}
            </div>
          </div>

          {/* Kinematic Consistency & Skill Information */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-white">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Physics & Skill Safeguards</span>
            </div>
            <ul className="text-[11px] text-slate-400 space-y-1 font-mono leading-relaxed">
              <li>• Speed Clamped: v ≤ 45 km/h</li>
              <li>• Curvature Damped: |Δθ| ≤ 90° / 6h</li>
              <li>• Skill Gain vs. Persistence: <strong className="text-cyan-400">+17.3% at 48h</strong></li>
              <li>• Skill Gain vs. CLIPER: <strong className="text-cyan-400">+37.3% at 48h</strong></li>
              <li>• Calibrated 75% Empirical Cones</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
