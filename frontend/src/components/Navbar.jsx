import React, { useState, useEffect } from 'react';
import { Menu, ShieldAlert, Wifi, Clock, Server } from 'lucide-react';
import StatusBadge from './StatusBadge';
import { checkBackendHealth } from '../services/api';

export default function Navbar({ onMenuClick }) {
  const [time, setTime] = useState(new Date().toUTCString());
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    // Clock tick
    const timer = setInterval(() => {
      setTime(new Date().toUTCString());
    }, 1000);

    // Initial and periodic backend status check
    const checkStatus = async () => {
      const res = await checkBackendHealth();
      setBackendStatus(res.status === 'ok' ? 'online' : 'offline');
    };
    checkStatus();
    const statusInterval = setInterval(checkStatus, 30000);

    return () => {
      clearInterval(timer);
      clearInterval(statusInterval);
    };
  }, []);

  return (
    <header className="h-16 bg-meteo-850/80 backdrop-blur-md border-b border-slate-800 px-4 lg:px-6 flex items-center justify-between z-20 sticky top-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 lg:hidden"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-slate-100 tracking-wide uppercase font-mono">
              Operational Command Console
            </h1>
            <span className="hidden sm:inline-block w-1 h-1 rounded-full bg-slate-600"></span>
            <span className="hidden sm:inline-block text-xs text-slate-400 font-mono">
              Basin: North Indian Ocean
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* UTC Clock */}
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-meteo-900 border border-slate-800 text-xs font-mono text-slate-300">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>{time.replace('GMT', 'UTC')}</span>
        </div>

        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span>FastAPI:</span>
          </div>
          <StatusBadge 
            status={backendStatus === 'online' ? 'online' : 'offline'} 
            label={backendStatus === 'online' ? 'Backend Live' : 'Backend Offline'} 
          />
        </div>

        {/* Advisory Warning Pill */}
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-950/40 border border-rose-800/40 text-[11px] text-rose-300 font-mono">
          <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
          <span>SIH Prototype — Non-Operational</span>
        </div>
      </div>
    </header>
  );
}
