/**
 * Main App component
 */

import React from 'react';
import { Routes, Route } from 'react-router-dom';
import AdminDashboard from './pages/AdminDashboard';
import MobileCamera from './pages/MobileCamera';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/mobile" element={<MobileCamera />} />
        <Route path="/*" element={<AdminDashboard />} />
      </Routes>
    </ErrorBoundary>
  );
}

export default App;
