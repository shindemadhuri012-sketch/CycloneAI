import React from 'react';
import { 
  BookOpen, 
  ArrowDown, 
  Users, 
  ShieldAlert, 
  Satellite, 
  Layers, 
  Cpu, 
  Sparkles, 
  LayoutDashboard,
  CheckCircle2,
  Building,
  GraduationCap,
  Radio,
  LifeBuoy
} from 'lucide-react';

export default function MethodologyPage() {
  const PIPELINE_STEPS = [
    { title: '1. Multi-source Satellite Data', desc: 'Ingestion of multi-spectral observations (TIR-1, TIR-2, WV, VIS) from INSAT-3D/3DR, Himawari, and GOES.', tag: 'Data Ingestion' },
    { title: '2. Data Collection', desc: 'Automated retrieval of raw HDF5/NetCDF files from MOSDAC and Earth Observation archives.', tag: 'Collection' },
    { title: '3. Data Cleaning', desc: 'Artifact removal, missing granule interpolation, de-striping, and geospatial alignment.', tag: 'Quality Control' },
    { title: '4. Preprocessing', desc: 'Radiometric calibration (DN to Kelvin), storm centering, standard crop extraction, and normalization.', tag: 'Preprocessing' },
    { title: '5. Feature Extraction', desc: 'Extraction of spatial patterns, convective cloud spirals, and eye eyewall structures via deep vision encoders.', tag: 'Representation' },
    { title: '6. AI / ML Models', desc: 'Orchestration of specialized deep neural architectures trained on historical best-track benchmarks.', tag: 'Core AI' },
    { title: '7. Identification (Presence)', desc: 'Binary detection classifier distinguishing cyclonic vortices from unstructured cloud clusters.', tag: 'Task 1' },
    { title: '8. Classification (Pattern/Scale)', desc: 'Multi-class categorization aligned with IMD and Saffir-Simpson intensity scales.', tag: 'Task 2' },
    { title: '9. Intensity Prediction', desc: 'Regression forecasting of Maximum Sustained Wind Speed (knots) and Central Barometric Pressure (hPa).', tag: 'Task 3' },
    { title: '10. Track Prediction', desc: 'Spatio-temporal trajectory progression forecasting predicting future storm eye coordinates and error cones.', tag: 'Task 4' },
    { title: '11. Explainable AI (XAI)', desc: 'Grad-CAM and feature attribution heatmaps providing transparent physical interpretability for meteorologists.', tag: 'Interpretability' },
    { title: '12. Command Dashboard', desc: 'High-density decision-support console delivering real-time maps, trend telemetry, and automated alerts.', tag: 'Presentation' },
  ];

  const USER_PERSONAS = [
    {
      role: 'Meteorologists & Weather Scientists',
      icon: Radio,
      desc: 'Operational forecasters requiring objective, automated verification of storm structure, eye formation, and intensity trends to complement NWP models.',
      color: 'border-cyan-500/30 text-cyan-400 bg-cyan-950/20'
    },
    {
      role: 'Disaster Management Authorities (NDMA / SDMAs)',
      icon: Building,
      desc: 'Government decision-makers requiring early situational awareness to mobilize rescue teams, prepare cyclone shelters, and coordinate evacuations.',
      color: 'border-blue-500/30 text-blue-400 bg-blue-950/20'
    },
    {
      role: 'Emergency Response Teams (NDRF / Coast Guard)',
      icon: LifeBuoy,
      desc: 'First responders requiring spatial track trajectories, landfall point estimates, and localized wind hazard cones.',
      color: 'border-amber-500/30 text-amber-400 bg-amber-950/20'
    },
    {
      role: 'Researchers & Earth Scientists',
      icon: GraduationCap,
      desc: 'Academic institutions studying tropical cyclogenesis, rapid intensification mechanisms, and deep learning benchmarks on satellite data.',
      color: 'border-purple-500/30 text-purple-400 bg-purple-950/20'
    },
    {
      role: 'General Public (Secondary)',
      icon: Users,
      desc: 'Coastal citizens and communities seeking accessible, non-alarmist hazard awareness and verified educational updates.',
      color: 'border-emerald-500/30 text-emerald-400 bg-emerald-950/20'
    },
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono mb-1">
          <BookOpen className="w-4 h-4" />
          <span>SYSTEM ARCHITECTURE & OPERATIONAL FRAMEWORK</span>
        </div>
        <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight">Methodology & Design Principles</h1>
        <p className="text-sm text-slate-400 max-w-3xl mt-1">
          Technical specifications, planned machine learning pipeline flow, intended user stakeholder analysis, and operational disclaimers for Smart India Hackathon 2026.
        </p>
      </div>

      {/* Mandatory Disclaimer Callout */}
      <div className="p-5 lg:p-6 rounded-2xl bg-rose-950/40 border border-rose-800/80 space-y-2">
        <div className="flex items-center gap-3 text-rose-400">
          <ShieldAlert className="w-6 h-6 shrink-0" />
          <h2 className="text-base font-bold uppercase tracking-wider">Official Prototype Disclaimer</h2>
        </div>
        <p className="text-sm text-rose-200 leading-relaxed pl-9">
          <strong>"This platform is a research and decision-support prototype. It does not replace official meteorological forecasts, warnings, or emergency instructions."</strong>
        </p>
        <p className="text-xs text-rose-300/80 leading-relaxed pl-9">
          All tropical cyclone alerts, advisories, and evacuation orders must strictly be followed in accordance with the official directives issued by the India Meteorological Department (IMD), the Ministry of Earth Sciences, and the National Disaster Management Authority (NDMA).
        </p>
      </div>

      {/* Planned End-to-End Pipeline */}
      <div className="glass-panel p-6 lg:p-8 rounded-2xl border border-slate-800 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">Planned AI/ML Pipeline Architecture</h2>
            <p className="text-xs text-slate-400">Step-by-step data transformation and inference hierarchy</p>
          </div>
          <span className="text-xs font-mono text-cyan-400 px-2.5 py-1 rounded bg-cyan-950/60 border border-cyan-800/50">
            End-to-End Flow
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 relative">
          {PIPELINE_STEPS.map((step, idx) => (
            <div 
              key={idx}
              className="p-4 rounded-xl bg-meteo-900 border border-slate-800 hover:border-cyan-500/40 transition-all flex flex-col justify-between space-y-2"
            >
              <div>
                <div className="flex items-center justify-between text-[11px] font-mono mb-1.5">
                  <span className="text-cyan-400 font-bold">{step.tag}</span>
                  <span className="text-slate-500">#{idx + 1}</span>
                </div>
                <h3 className="text-sm font-semibold text-white mb-1">{step.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Intended User Personas */}
      <div className="space-y-4">
        <div>
          <h2 className="text-lg font-bold text-white">Primary & Secondary Stakeholders</h2>
          <p className="text-xs text-slate-400">Target operational beneficiaries of CycloneAI decision support</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {USER_PERSONAS.map((persona, idx) => {
            const Icon = persona.icon;
            return (
              <div 
                key={idx}
                className={`p-5 rounded-xl border ${persona.color} flex flex-col justify-between space-y-3`}
              >
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="text-sm font-bold text-white leading-snug">{persona.role}</h3>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{persona.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
