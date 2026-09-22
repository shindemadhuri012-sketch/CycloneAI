import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import ErrorBoundary from './components/ErrorBoundary';
import DashboardPage from './pages/DashboardPage';
import SatelliteAnalysisPage from './pages/SatelliteAnalysisPage';
import ClassificationPage from './pages/ClassificationPage';
import IntensityPage from './pages/IntensityPage';
import TrackPredictionPage from './pages/TrackPredictionPage';
import HistoricalPage from './pages/HistoricalPage';
import ModelPerformancePage from './pages/ModelPerformancePage';
import MethodologyPage from './pages/MethodologyPage';
import ExplainabilityPage from './pages/ExplainabilityPage';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<ErrorBoundary fallbackTitle="Dashboard Display Notice"><DashboardPage /></ErrorBoundary>} />
        <Route path="satellite-analysis" element={<ErrorBoundary fallbackTitle="Satellite Module Notice"><SatelliteAnalysisPage /></ErrorBoundary>} />
        <Route path="classification" element={<ErrorBoundary fallbackTitle="Classification Module Notice"><ClassificationPage /></ErrorBoundary>} />
        <Route path="intensity" element={<ErrorBoundary fallbackTitle="Intensity Module Notice"><IntensityPage /></ErrorBoundary>} />
        <Route path="track" element={<ErrorBoundary fallbackTitle="Track Forecast Module Notice"><TrackPredictionPage /></ErrorBoundary>} />
        <Route path="explainability" element={<ErrorBoundary fallbackTitle="Explainability Module Notice"><ExplainabilityPage /></ErrorBoundary>} />
        <Route path="historical" element={<ErrorBoundary fallbackTitle="Historical Catalog Notice"><HistoricalPage /></ErrorBoundary>} />
        <Route path="model-performance" element={<ErrorBoundary fallbackTitle="Model Performance Notice"><ModelPerformancePage /></ErrorBoundary>} />
        <Route path="methodology" element={<ErrorBoundary fallbackTitle="Methodology Notice"><MethodologyPage /></ErrorBoundary>} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}
