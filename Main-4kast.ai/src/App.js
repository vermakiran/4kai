import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './components/dashboard';
import ImportData from './components/Import';
import ForecastSettings from './components/forecastsettings';
import PlannerWorkbench from './components/PlannerWorkbench.js';
import MasterData from './components/masterdata.js';
import IAMPage from './components/IAM.js';
import Notifications from './components/Notifications';
import Settings from './components/Settings';
import Profile from './components/Profile';
import Login from './components/Login';
import ForgotPassword from './components/ForgotPassword';
import SignUp from './components/SignUp';
import ProtectedRoute from './components/ProtectedRoute';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Default route redirects to /login */}
        <Route path="/" element={<Navigate to="/login" replace />} />
        
        {/* Login-related routes (no app-container layout) */}
        <Route path="/login" element={<Login />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/signup" element={<SignUp />} />

        {/* Protected routes (with app-container layout) */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <Dashboard />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/import"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <ImportData />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/forecast-settings"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <ForecastSettings />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/planner-workbench"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <PlannerWorkbench />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/master-data"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <MasterData />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/iam"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <IAMPage />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/notifications"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <Notifications />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <Settings />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <div className="app-container">
                <Sidebar />
                <div className="main-content">
                  <Navbar />
                  <div className="content">
                    <Profile />
                  </div>
                </div>
              </div>
            </ProtectedRoute>
          }
        />
      </Routes>
    </Router>
  );
}
export default App;