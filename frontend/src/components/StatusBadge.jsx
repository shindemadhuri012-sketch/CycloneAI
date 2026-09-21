import React from 'react';

export default function StatusBadge({ status, label, className = '' }) {
  const getStyles = () => {
    switch (status?.toLowerCase()) {
      case 'ok':
      case 'online':
      case 'active':
        return 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60 ring-1 ring-emerald-500/20';
      case 'warning':
      case 'standby':
      case 'awaiting model':
        return 'bg-amber-950/60 text-amber-400 border-amber-800/60 ring-1 ring-amber-500/20';
      case 'not_connected':
      case 'model not connected':
      case 'offline':
      case 'data required':
      case 'not available':
      default:
        return 'bg-slate-900/80 text-slate-400 border-slate-700/60 ring-1 ring-slate-600/20';
    }
  };

  const getDotStyles = () => {
    switch (status?.toLowerCase()) {
      case 'ok':
      case 'online':
      case 'active':
        return 'bg-emerald-400 animate-pulse';
      case 'warning':
      case 'standby':
      case 'awaiting model':
        return 'bg-amber-400';
      default:
        return 'bg-slate-500';
    }
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono rounded-full border ${getStyles()} ${className}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${getDotStyles()}`}></span>
      {label || status}
    </span>
  );
}
