import React from 'react';
import { Cpu, AlertCircle, Database, Layers } from 'lucide-react';
import StatusBadge from './StatusBadge';

export default function EmptyState({
  title = "Module Awaiting Model",
  description = "This capability will be active once the underlying AI/ML model is trained on curated satellite archives.",
  badgeStatus = "awaiting model",
  badgeLabel = "Awaiting Model",
  icon: Icon = Cpu,
  phase = "Phase Planned"
}) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center glass-panel rounded-xl border border-slate-800/80 my-4">
      <div className="p-3.5 rounded-2xl bg-slate-800/60 border border-slate-700/50 text-cyan-400 mb-4 shadow-inner">
        <Icon className="w-8 h-8 stroke-[1.5]" />
      </div>
      <div className="mb-2">
        <StatusBadge status={badgeStatus} label={badgeLabel} />
      </div>
      <h3 className="text-base font-semibold text-slate-100 mt-1 mb-1">{title}</h3>
      <p className="text-sm text-slate-400 max-w-md leading-relaxed">{description}</p>
      {phase && (
        <span className="mt-4 text-xs font-mono text-cyan-400/80 bg-cyan-950/40 px-2.5 py-0.5 rounded border border-cyan-800/40">
          Target: {phase}
        </span>
      )}
    </div>
  );
}
