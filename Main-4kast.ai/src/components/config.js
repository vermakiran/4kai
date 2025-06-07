// src/config.js
export const BASE_URL = 'http://127.0.0.1:5000';// Base URL for the server
export const LOGIN_ENDPOINT = `${BASE_URL}/api/auth/token`;   // Full login endpoint
export const UPLOAD_FILE_ENDPOINT = `${BASE_URL}/api/engine/uploadfile`; // Full upload file endpoint
export const UPLOAD_CLEANED_DATA_ENDPOINT = `${BASE_URL}/api/engine/upload-cleaned-data`; // New endpoint for cleaned data and model
export const MODELS_ENDPOINT = `${BASE_URL}/api/engine/models`; // Endpoint for fetching available forecasting models
export const RUN_FORECAST_ENDPOINT = `${BASE_URL}/api/engine/run-forecast`; // Endpoint for running forecasts
export const DASHBOARD_ENDPOINT = `${BASE_URL}/api/engine/dashboard-data`; // Endpoint for dashboard
export const PLANNERWB_ENDPOINT = `${BASE_URL}/api/engine/planner-data`; // Endpoint for Planner-Workbench


// export const BASE_URL = 'https://4kast-backend-v2.cfapps.eu10-005.hana.ondemand.com';  Base URL for the server