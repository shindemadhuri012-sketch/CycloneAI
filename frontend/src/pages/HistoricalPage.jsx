import React, { useState, useEffect } from 'react';
import { 
  History, 
  Search, 
  Filter, 
  Database, 
  Calendar, 
  Globe, 
  Layers, 
  AlertCircle,
  FileQuestion,
  RefreshCw,
  Wind,
  Navigation,
  CheckCircle2,
  Cpu
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';
import { fetchStormCatalog, fetchStormPoints } from '../services/api';

export default function HistoricalPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedYear, setSelectedYear] = useState('All');
  const [selectedRegion, setSelectedRegion] = useState('All');
  const [storms, setStorms] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // Selected storm inspection modal / panel
  const [inspectedStorm, setInspectedStorm] = useState(null);
  const [stormPoints, setStormPoints] = useState([]);
  const [isLoadingPoints, setIsLoadingPoints] = useState(false);

  const YEARS = ['All', '2024', '2023', '2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013'];

  useEffect(() => {
    loadCatalog();
  }, [selectedYear, selectedRegion]);

  const loadCatalog = async () => {
    setIsLoading(true);
    try {
      const res = await fetchStormCatalog({
        year: selectedYear,
        basin: selectedRegion,
        search: searchQuery,
        limit: 50
      });
      if (res && res.storms) {
        setStorms(res.storms);
        setTotalCount(res.total_storms);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    loadCatalog();
  };

  const handleInspectStorm = async (storm) => {
    setInspectedStorm(storm);
    setIsLoadingPoints(true);
    try {
      const details = await fetchStormPoints(storm.sid);
      if (details && details.points) {
        setStormPoints(details.points);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoadingPoints(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
            <History className="w-4 h-4" />
            <span>ARCHIVAL RESEARCH & CLIMATOLOGY EXPLORER</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Historical Cyclone Archive</h1>
          <p className="text-sm text-slate-400">
            Searchable meteorological archive of real North Indian Ocean cyclones (IBTrACS NIO best-track dataset, 1842-2024).
          </p>
        </div>

        <div className="flex items-center gap-2">
          <StatusBadge status="operational" label={`Archive Active: ${totalCount} Storms Indexed`} />
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
        <form onSubmit={handleSearchSubmit} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Search Input */}
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search storm by name or SID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 bg-slate-900/90 border border-slate-700/80 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          {/* Year Filter */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-slate-400 shrink-0" />
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
            >
              {YEARS.map((y) => (
                <option key={y} value={y}>{y === 'All' ? 'All Seasons' : `Season ${y}`}</option>
              ))}
            </select>
          </div>

          {/* Subbasin Filter */}
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-slate-400 shrink-0" />
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="All">All NIO Subbasins</option>
              <option value="BB">Bay of Bengal (BB)</option>
              <option value="AS">Arabian Sea (AS)</option>
            </select>
          </div>

          {/* Search Action */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-cyan-600/20 cursor-pointer"
          >
            {isLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
            <span>Filter Archive</span>
          </button>
        </form>
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Storms List Table (7 cols or 12 cols if no inspect) */}
        <div className={inspectedStorm ? 'lg:col-span-7 space-y-4' : 'lg:col-span-12 space-y-4'}>
          <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-mono text-cyan-400 font-bold">
                HISTORICAL BEST-TRACK RECORDS (DISPLAYING {storms.length} OF {totalCount})
              </span>
              <span className="text-xs font-mono text-slate-400">Zero Synthetic Records</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/70 border-b border-slate-800 text-[11px] font-mono text-slate-400">
                  <tr>
                    <th className="py-3 px-4">Storm Name</th>
                    <th className="py-3 px-3">Season</th>
                    <th className="py-3 px-3">Basin</th>
                    <th className="py-3 px-3">Peak Grade</th>
                    <th className="py-3 px-3">Vmax (kt)</th>
                    <th className="py-3 px-3">Min Pmin</th>
                    <th className="py-3 px-3">Points</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 font-mono">
                  {storms.map((s) => (
                    <tr
                      key={s.sid}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        inspectedStorm?.sid === s.sid ? 'bg-cyan-950/30' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-bold text-white">{s.name}</td>
                      <td className="py-3 px-3 text-slate-300">{s.season}</td>
                      <td className="py-3 px-3">
                        <span className="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-400 text-[10px]">
                          {s.subbasin === 'BB' ? 'Bay of Bengal' : s.subbasin === 'AS' ? 'Arabian Sea' : s.subbasin}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-bold text-amber-400">{s.max_grade}</td>
                      <td className="py-3 px-3 text-emerald-400 font-bold">{s.peak_wind_kt} kt</td>
                      <td className="py-3 px-3 text-slate-300">{s.min_pressure_hpa} hPa</td>
                      <td className="py-3 px-3 text-slate-400">{s.point_count}</td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => handleInspectStorm(s)}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-cyan-600 text-slate-200 hover:text-white text-[11px] transition-all cursor-pointer font-medium"
                        >
                          Inspect Track
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Selected Storm Chronological Inspection Drawer (5 cols) */}
        {inspectedStorm && (
          <div className="lg:col-span-5 space-y-4">
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-base font-bold text-white">
                    Cyclone {inspectedStorm.name} ({inspectedStorm.season})
                  </h3>
                  <p className="text-xs font-mono text-cyan-400">SID: {inspectedStorm.sid}</p>
                </div>
                <button
                  onClick={() => setInspectedStorm(null)}
                  className="text-xs text-slate-400 hover:text-white"
                >
                  ✕ Close
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">PEAK INTENSITY</span>
                  <span className="text-emerald-400 font-bold">{inspectedStorm.peak_wind_kt} knots</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-500 block">MIN PRESSURE</span>
                  <span className="text-cyan-400 font-bold">{inspectedStorm.min_pressure_hpa} hPa</span>
                </div>
              </div>

              <div className="space-y-2">
                <span className="text-xs font-mono text-slate-400">CHRONOLOGICAL OBSERVATIONS ({stormPoints.length}):</span>
                {isLoadingPoints ? (
                  <div className="p-6 text-center text-xs text-slate-400">Loading observations...</div>
                ) : (
                  <div className="max-h-80 overflow-y-auto space-y-1.5 pr-1">
                    {stormPoints.map((pt, idx) => (
                      <div key={idx} className="p-2 rounded-lg bg-slate-900/80 border border-slate-800/80 text-xs font-mono flex items-center justify-between">
                        <div>
                          <div className="text-slate-200 font-bold">{pt.latitude}°N, {pt.longitude}°E</div>
                          <div className="text-[10px] text-slate-500">{pt.timestamp.replace('T', ' ').slice(0, 16)}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-emerald-400 font-bold">{pt.wind_kt} kt</div>
                          <div className="text-[10px] text-slate-400">{pt.central_pres} hPa</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
