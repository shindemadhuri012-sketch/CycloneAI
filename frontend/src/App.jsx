import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
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
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="satellite-analysis" element={<SatelliteAnalysisPage />} />
        <Route path="classification" element={<ClassificationPage />} />
        <Route path="intensity" element={<IntensityPage />} />
        <Route path="track" element={<TrackPredictionPage />} />
        <Route path="explainability" element={<ExplainabilityPage />} />
        <Route path="historical" element={<HistoricalPage />} />
        <Route path="model-performance" element={<ModelPerformancePage />} />
        <Route path="methodology" element={<MethodologyPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}
