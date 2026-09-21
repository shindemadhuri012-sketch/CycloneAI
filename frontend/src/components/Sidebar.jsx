import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Satellite, 
  Layers, 
  Wind, 
  Navigation, 
  Sparkles,
  History, 
  BarChart3, 
  BookOpen,
  X,
  Radio
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Satellite Analysis', path: '/satellite-analysis', icon: Satellite },
  { name: 'Classification', path: '/classification', icon: Layers },
  { name: 'Intensity Prediction', path: '/intensity', icon: Wind },
  { name: 'Track Prediction', path: '/track', icon: Navigation },
  { name: 'Explainable AI', path: '/explainability', icon: Sparkles },
  { name: 'Historical Cyclones', path: '/historical', icon: History },
  { name: 'Model Performance', path: '/model-performance', icon: BarChart3 },
  { name: 'Methodology', path: '/methodology', icon: BookOpen },
];

export default function Sidebar({ isOpen, onClose }) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div 
          onClick={onClose} 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden transition-opacity"
        />
      )}

      {/* Sidebar Container */}
      <aside 
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-meteo-850/95 backdrop-blur-md border-r border-slate-800 flex flex-col transition-transform duration-300 ease-in-out lg:static lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header Branding */}
        <div className="flex items-center justify-between h-16 px-5 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/20">
              <span className="text-xl">🌪️</span>
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-lg tracking-tight text-white">CycloneAI</span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                  SIH'26
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-normal truncate">Meteorological Intelligence</p>
            </div>
          </div>
          <button 
            onClick={onClose} 
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Operational Status Pill */}
        <div className="px-4 pt-4 pb-2">
          <div className="px-3 py-2 rounded-lg bg-meteo-900 border border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5 font-medium">
              <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              System Status
            </span>
            <span className="font-mono text-cyan-400 font-semibold">STANDBY</span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-3 py-3 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => {
                  if (window.innerWidth < 1024) onClose();
                }}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                    <span className="truncate">{item.name}</span>
                    {isActive && (
                      <span className="ml-auto w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400"></span>
                    )}
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Footer Info */}
        <div className="p-4 border-t border-slate-800 text-xs text-slate-400">
          <div className="flex items-center justify-between text-[11px] font-mono mb-1">
            <span>PHASE 1</span>
            <span className="text-emerald-400 font-medium">FOUNDATION</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div className="bg-cyan-500 h-full w-[8.33%] rounded-full"></div>
          </div>
          <p className="mt-2 text-[10px] text-slate-500 leading-tight">
            Research & decision support prototype.
          </p>
        </div>
      </aside>
    </>
  );
}
