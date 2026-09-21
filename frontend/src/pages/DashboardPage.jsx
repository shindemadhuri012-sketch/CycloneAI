import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Radar, 
  Layers, 
  Wind, 
  TrendingUp, 
  Navigation, 
  AlertTriangle, 
  Satellite, 
  Cpu, 
  ArrowRight,
  ShieldCheck,
  Activity,
  Info,
  Sparkles,
  RefreshCw,
  Play,
  Clock,
  CheckCircle2
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import { checkBackendHealth, runUnifiedPipeline, fetchStormPresets } from '../services/api';

export default function DashboardPage() {
  const [health, setHealth] = useState({ status: 'checking', model_status: 'loading' });
  const [presets, setPresets] = useState([]);
  const [selectedStorm, setSelectedStorm] = useState('FANI');
  const [isRunning, setIsRunning] = useState(false);
  const [pipelineData, setPipelineData] = useState(null);

  // Active observation parameters
  const [lat, setLat] = useState(16.0);
  const [lon, setLon] = useState(86.5);
  const [centralPres, setCentralPres] = useState(970.0);
  const [currentWind, setCurrentWind] = useState(85.0);
  const [forwardSpeed, setForwardSpeed] = useState(16.0);
  const [bearing, setBearing] = useState(320.0);

  useEffect(() => {
    async function init() {
      const h = await checkBackendHealth();
      setHealth(h);

      const p = await fetchStormPresets();
      if (p && p.length > 0) {
        setPresets(p);
        applyPreset(p[0]);
      } else {
        runAnalysis();
      }
    }
    init();
  }, []);

  const applyPreset = (preset) => {
    setSelectedStorm(preset.name);
    setLat(preset.observation.lat);
    setLon(preset.observation.lon);
    setCentralPres(preset.observation.central_pres);
    setCurrentWind(preset.observation.current_wind_speed_knots);
    setForwardSpeed(preset.observation.forward_speed);
    setBearing(preset.observation.bearing);
    executePipeline(preset.observation, preset.name);
  };

  const runAnalysis = () => {
    executePipeline({
      lat,
      lon,
      central_pres: centralPres,
      current_wind_speed_knots: currentWind,
      forward_speed: forwardSpeed,
      bearing
    }, selectedStorm);
  };

  const executePipeline = async (params, stormName) => {
    setIsRunning(true);
    try {
      const res = await runUnifiedPipeline({
        storm_id: stormName || selectedStorm,
        lat: params.lat,
        lon: params.lon,
        central_pres: params.central_pres,
        current_wind_speed_knots: params.current_wind_speed_knots,
        forward_speed: params.forward_speed,
        bearing: params.bearing,
        month: 5,
        subbasin: 'BB'
      });
      setPipelineData(res);
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunning(false);
    }
  };

  const OVERVIEW_CARDS = [
    {
      title: 'Cyclone Detection',
      status: 'operational',
      statusLabel: 'Active (0.9319 AUC)',
      icon: Radar,
      description: 'Binary identification of cyclonic vortex formation from multi-spectral satellite granules.',
      details: pipelineData?.identification?.is_cyclone ? 'Vortex Detected (High Confidence)' : 'Model Active',
      link: '/satellite-analysis',
      accent: 'border-cyan-500/20 hover:border-cyan-500/50'
    },
    {
      title: 'Cyclone Classification',
      status: 'operational',
      statusLabel: 'Active (91.8% Match)',
      icon: Layers,
      description: 'Multi-class categorization aligned with official IMD 8-tier intensity scale.',
      details: pipelineData?.classification?.predicted_category || 'Depression to Super Cyclone',
      link: '/classification',
      accent: 'border-blue-500/20 hover:border-blue-500/50'
    },
    {
      title: 'Intensity Forecasting',
      status: 'operational',
      statusLabel: 'Active (+8.1% Skill)',
      icon: Wind,
      description: 'Multi-horizon (+6h, +12h, +24h) Vmax and Pmin prediction with 90% confidence bands.',
      details: pipelineData?.intensity?.forecast_horizons?.['+24h']
        ? `+24h: ${pipelineData.intensity.forecast_horizons['+24h'].wind_knots} kt (${pipelineData.intensity.forecast_horizons['+24h'].wind_bounds_90.low_knots}-${pipelineData.intensity.forecast_horizons['+24h'].wind_bounds_90.high_knots})`
        : 'Continuous Vmax & Pmin',
      link: '/intensity',
      accent: 'border-teal-500/20 hover:border-teal-500/50'
    },
    {
      title: 'Track Prediction',
      status: 'operational',
      statusLabel: 'Active (+17.3% Skill)',
      icon: Navigation,
      description: 'Incremental displacement vector forecasting with 75% empirical uncertainty cones.',
      details: pipelineData?.track?.predicted_track?.length
        ? `${pipelineData.track.predicted_track.length} Waypoints (+6h to +48h)`
        : 'Coordinate Progression',
      link: '/track',
      accent: 'border-purple-500/20 hover:border-purple-500/50'
    },
    {
      title: 'Explainable AI (XAI)',
      status: 'operational',
      statusLabel: 'Active (348.9× Ratio)',
      icon: Sparkles,
      description: 'Tree-Path Saabas feature attribution satisfying the exact Efficiency Axiom.',
      details: pipelineData?.explainability?.axisymmetry_score !== undefined
        ? `Axisymmetry: ${pipelineData.explainability.axisymmetry_score.toFixed(2)} | ΔT: ${pipelineData.explainability.eye_contrast_celsius}°C`
        : 'Attribution & Saliency',
      link: '/explainability',
      accent: 'border-amber-500/20 hover:border-amber-500/50'
    },
    {
      title: 'Historical Benchmarks',
      status: 'operational',
      statusLabel: 'Verified IBTrACS',
      icon: ShieldCheck,
      description: 'North Indian Ocean archive (1,858 cyclones) with 0.00% leakage test split.',
      details: '70 Held-out Test Storms',
      link: '/historical',
      accent: 'border-emerald-500/20 hover:border-emerald-500/50'
    }
  ];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
            <Cpu className="w-4 h-4" />
            <span>INTELLIGENT TROPICAL CYCLONE ANALYSIS & PREDICTION SYSTEM</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Meteorological Command Center</h1>
          <p className="text-sm text-slate-400">
            Real-time multi-model intelligence: Detection, Classification, Intensity, Track, and Explainable AI.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <StatusBadge 
            status={health.model_status === 'active' ? 'operational' : 'degraded'} 
            label={`System Status: ${health.model_status === 'active' ? 'All 4 AI Models + XAI Active' : 'Checking...'}`} 
          />
        </div>
      </div>

      {/* Quick Run & Preset Bar */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold">1-CLICK REAL STORM PRESETS:</span>
            <div className="flex flex-wrap gap-1.5">
              {presets.map((p) => (
                <button
                  key={p.sid}
                  onClick={() => applyPreset(p)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-mono transition-all cursor-pointer ${
                    selectedStorm === p.name
                      ? 'bg-cyan-600 text-white font-bold shadow-md shadow-cyan-600/30'
                      : 'bg-slate-900/90 text-slate-300 hover:bg-slate-800 border border-slate-700/60'
                  }`}
                >
                  {p.name} ({p.season})
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={runAnalysis}
            disabled={isRunning}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white text-xs font-bold shadow-lg shadow-cyan-600/20 cursor-pointer"
          >
            {isRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
            <span>Run Complete Pipeline</span>
          </button>
        </div>

        {/* Holistic Analysis Dossier Card */}
        {pipelineData && (
          <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-700/80 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-white uppercase tracking-wider">
                  Analyzed System: {pipelineData.storm_id || 'CYCLONE'}
                </span>
                <span className="text-xs font-mono text-cyan-400 px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-800/60">
                  Lat {lat}°N, Lon {lon}°E
                </span>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Inference Latency: {pipelineData.execution_time_ms} ms
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Category */}
              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">IMD CLASSIFICATION</div>
                <div className="text-base font-bold text-cyan-300 truncate">
                  {pipelineData.classification?.predicted_category || 'Classified'}
                </div>
                <div className="text-[11px] text-slate-400">
                  Confidence: {((pipelineData.classification?.confidence || 0) * 100).toFixed(1)}%
                </div>
              </div>

              {/* Current Wind & Pressure */}
              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">CURRENT INTENSITY</div>
                <div className="text-base font-bold text-teal-300">
                  {currentWind} knots | {centralPres} hPa
                </div>
                <div className="text-[11px] text-slate-400">
                  Deficit: {(1010 - centralPres).toFixed(1)} hPa
                </div>
              </div>

              {/* +24h Forecast & RI */}
              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">+24H INTENSITY FORECAST</div>
                <div className="text-base font-bold text-amber-300">
                  {pipelineData.intensity?.forecast_horizons?.['+24h']?.wind_knots || currentWind} kt
                </div>
                <div className="text-[11px] font-mono text-rose-400">
                  {pipelineData.intensity?.rapid_intensification?.is_ri_expected ? '⚠️ RAPID INTENSIFICATION' : 'Steady Evolution'}
                </div>
              </div>

              {/* Trajectory */}
              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-[10px] font-mono text-slate-400">TRACK PROGRESSION</div>
                <div className="text-base font-bold text-purple-300">
                  {pipelineData.track?.predicted_track?.length || 4} Horizons (+48h)
                </div>
                <div className="text-[11px] text-slate-400">
                  Heading: {bearing}° at {forwardSpeed} km/h
                </div>
              </div>
            </div>

            {/* Synoptic Briefing */}
            {pipelineData.explainability?.synoptic_briefing && (
              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 text-xs font-mono text-slate-300 leading-relaxed">
                <strong className="text-cyan-400">IMD Synoptic Diagnostic:</strong> {pipelineData.explainability.synoptic_briefing}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 6 Subsystem Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {OVERVIEW_CARDS.map((card, idx) => {
          const Icon = card.icon;
          return (
            <Link
              key={idx}
              to={card.link}
              className={`glass-panel p-6 rounded-2xl border transition-all duration-300 group flex flex-col justify-between ${card.accent}`}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="w-10 h-10 rounded-xl bg-slate-800/80 border border-slate-700/60 flex items-center justify-center text-cyan-400 group-hover:scale-110 transition-transform">
                    <Icon className="w-5 h-5" />
                  </div>
                  <StatusBadge status={card.status} label={card.statusLabel} />
                </div>

                <div>
                  <h3 className="text-base font-semibold text-white group-hover:text-cyan-400 transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    {card.description}
                  </p>
                </div>
              </div>

              <div className="pt-4 mt-4 border-t border-slate-800/80 flex items-center justify-between text-xs">
                <span className="font-mono text-slate-400">{card.details}</span>
                <span className="text-cyan-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform font-medium">
                  Launch <ArrowRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
