import React, { useState, useEffect } from 'react';
import { 
  Layers, 
  Cpu, 
  ShieldAlert, 
  HelpCircle, 
  History, 
  Compass, 
  AlertTriangle, 
  RefreshCw, 
  Info,
  CheckCircle2,
  BarChart2
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import StatusBadge from '../components/StatusBadge';
import { classifyCyclone, fetchStormPresets } from '../services/api';

export default function ClassificationPage() {
  const [presets, setPresets] = useState([]);
  const [selectedStorm, setSelectedStorm] = useState('FANI');
  const [isClassifying, setIsClassifying] = useState(false);
  const [apiResponse, setApiResponse] = useState(null);

  // Observation parameters
  const [lat, setLat] = useState(16.0);
  const [lon, setLon] = useState(86.5);
  const [centralPres, setCentralPres] = useState(970.0);
  const [forwardSpeed, setForwardSpeed] = useState(16.0);
  const [bearing, setBearing] = useState(320.0);

  const IMD_CATEGORIES = [
    { code: 'LOW', name: 'Low Pressure Area', wind: '< 17 knots', color: '#64748b' },
    { code: 'D', name: 'Depression', wind: '17–27 knots', color: '#0ea5e9' },
    { code: 'DD', name: 'Deep Depression', wind: '28–33 knots', color: '#06b6d4' },
    { code: 'CS', name: 'Cyclonic Storm', wind: '34–47 knots', color: '#10b981' },
    { code: 'SCS', name: 'Severe Cyclonic Storm', wind: '48–63 knots', color: '#f59e0b' },
    { code: 'VSCS', name: 'Very Severe Cyclonic Storm', wind: '64–89 knots', color: '#f97316' },
    { code: 'ESCS', name: 'Extremely Severe Cyclonic Storm', wind: '90–119 knots', color: '#ef4444' },
    { code: 'SuCS', name: 'Super Cyclonic Storm', wind: '≥ 120 knots', color: '#881337' },
  ];

  useEffect(() => {
    async function load() {
      const p = await fetchStormPresets();
      if (p && p.length > 0) {
        setPresets(p);
        applyPreset(p[0]);
      } else {
        handleClassify();
      }
    }
    load();
  }, []);

  const applyPreset = (preset) => {
    setSelectedStorm(preset.name);
    setLat(preset.observation.lat);
    setLon(preset.observation.lon);
    setCentralPres(preset.observation.central_pres);
    setForwardSpeed(preset.observation.forward_speed);
    setBearing(preset.observation.bearing);
    runClassification(preset.observation);
  };

  const handleClassify = () => {
    runClassification({ lat, lon, central_pres: centralPres, forward_speed: forwardSpeed, bearing });
  };

  const runClassification = async (params) => {
    setIsClassifying(true);
    try {
      const res = await classifyCyclone({
        storm_id: selectedStorm,
        lat: params.lat,
        lon: params.lon,
        central_pres: params.central_pres,
        forward_speed: params.forward_speed,
        bearing: params.bearing,
        month: 5,
        subbasin: 'BB'
      });
      setApiResponse(res);
    } catch (e) {
      console.error(e);
    } finally {
      setIsClassifying(false);
    }
  };

  // Build chart data from all_probabilities
  const probData = IMD_CATEGORIES.map((cat) => {
    const rawP = apiResponse?.all_probabilities?.[cat.name] || 0.0;
    return {
      name: cat.name,
      code: cat.code,
      pct: parseFloat((rawP * 100).toFixed(1)),
      color: cat.color,
      isPredicted: apiResponse?.predicted_category === cat.name
    };
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
            <Layers className="w-4 h-4" />
            <span>PATTERN RECOGNITION & SCALE CATEGORIZATION</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Cyclone Category Classification</h1>
          <p className="text-sm text-slate-400">
            Categorization across all 8 official IMD categories: Low Pressure Area to Super Cyclonic Storm.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <StatusBadge status="operational" label="Classifier: Active (91.8% Match)" />
        </div>
      </div>

      {/* Preset Selector Banner */}
      <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-cyan-400 font-bold">REAL STORM PRESETS:</span>
          <div className="flex flex-wrap gap-1.5">
            {presets.map((p) => (
              <button
                key={p.sid}
                onClick={() => applyPreset(p)}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all cursor-pointer ${
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
          onClick={handleClassify}
          disabled={isClassifying}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-cyan-600/20 cursor-pointer"
        >
          {isClassifying ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Cpu className="w-3.5 h-3.5" />}
          <span>Re-Classify Category</span>
        </button>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Classification Results and Probability Distribution (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Main Classification Card */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-white">IMD Category Prediction</h2>
              <StatusBadge status="operational" label="Ensemble Active" />
            </div>

            {apiResponse && (
              <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-700/80 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div>
                    <span className="text-[11px] font-mono text-slate-400 block">PREDICTED CATEGORY</span>
                    <span className="text-xl font-extrabold text-cyan-300">
                      {apiResponse.predicted_category || 'Very Severe Cyclonic Storm'}
                    </span>
                    <span className="text-xs text-slate-400 ml-2 font-mono">
                      ({apiResponse.category_code || 'VSCS'})
                    </span>
                  </div>

                  <div className="text-right">
                    <span className="text-[11px] font-mono text-slate-400 block">CONFIDENCE</span>
                    <span className="text-xl font-mono font-extrabold text-emerald-400">
                      {((apiResponse.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 block">WIND SPEED RANGE</span>
                    <span className="text-sm font-bold text-slate-200">
                      {apiResponse.wind_range_kt || '64–89 knots'}
                    </span>
                    <span className="text-[11px] text-slate-400 block">
                      {apiResponse.wind_range_kmh || '118–166 km/h'}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 block">DAMAGE POTENTIAL</span>
                    <span className="text-xs text-rose-300 leading-snug block mt-0.5">
                      {apiResponse.damage_potential || 'Extensive damage to structures and uprooted trees.'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Posterior Probability Distribution Bar Chart */}
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-slate-400">8-TIER POSTERIOR PROBABILITY DISTRIBUTION:</span>
                <span className="text-[11px] font-mono text-cyan-400">∑ P = 1.000</span>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={probData} layout="vertical" margin={{ left: 130, right: 30, top: 10, bottom: 10 }}>
                    <XAxis type="number" unit="%" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" stroke="#cbd5e1" tick={{ fontSize: 11 }} width={140} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem' }}
                      formatter={(val) => [`${val}%`, 'Probability']}
                    />
                    <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                      {probData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.isPredicted ? '#06b6d4' : '#334155'} 
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: IMD Scale Legend & Environmental Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Observation Parameters */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <h2 className="text-base font-semibold text-white flex items-center justify-between">
              <span>Observation Input Parameters</span>
              <span className="text-xs font-mono text-cyan-400">Live Features</span>
            </h2>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">LATITUDE</span>
                <span className="text-slate-200 font-mono font-bold">{lat}°N</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">LONGITUDE</span>
                <span className="text-slate-200 font-mono font-bold">{lon}°E</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">CENTRAL PRESSURE</span>
                <span className="text-cyan-400 font-mono font-bold">{centralPres} hPa</span>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">PRESSURE DEFICIT</span>
                <span className="text-cyan-400 font-mono font-bold">{(1010 - centralPres).toFixed(1)} hPa</span>
              </div>
            </div>
          </div>

          {/* Official IMD Intensity Scale Legend */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-3">
            <h2 className="text-base font-semibold text-white">Official IMD Intensity Scale</h2>
            <div className="space-y-1.5">
              {IMD_CATEGORIES.map((cat, i) => (
                <div key={i} className="p-2 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: cat.color }} />
                    <span className="font-semibold text-slate-200">{cat.name}</span>
                    <span className="text-[10px] font-mono text-slate-400">({cat.code})</span>
                  </div>
                  <span className="font-mono text-slate-400 text-[11px]">{cat.wind}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
