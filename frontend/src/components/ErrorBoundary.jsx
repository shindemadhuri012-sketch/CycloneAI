import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('CycloneAI Component Render Error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 text-center glass-panel rounded-2xl border border-rose-900/60 bg-rose-950/20 my-6 max-w-xl mx-auto">
          <div className="p-3.5 rounded-2xl bg-rose-900/40 border border-rose-700/50 text-rose-400 mb-4 shadow-inner">
            <AlertTriangle className="w-8 h-8 stroke-[1.5]" />
          </div>
          <h3 className="text-lg font-semibold text-slate-100 mb-1">
            {this.props.fallbackTitle || 'Component Display Error'}
          </h3>
          <p className="text-xs text-slate-400 max-w-md leading-relaxed mb-4">
            {this.state.error?.message || 'An unexpected rendering error occurred while visualizing cyclone data.'}
          </p>
          <button
            onClick={this.handleReset}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md shadow-cyan-600/20 cursor-pointer transition-all"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reload Module</span>
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
