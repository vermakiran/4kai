// src/config.js
import axios from 'axios';

// Base URL configuration
export const BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';

// API endpoints
export const LOGIN_ENDPOINT = `${BASE_URL}/api/auth/token`;
export const UPLOAD_FILE_ENDPOINT = `${BASE_URL}/api/engine/uploadfile`;
export const UPLOAD_CLEANED_DATA_ENDPOINT = `${BASE_URL}/api/engine/upload-cleaned-data`;
export const MODELS_ENDPOINT = `${BASE_URL}/api/engine/models`;
export const RUN_FORECAST_ENDPOINT = `${BASE_URL}/api/engine/run-forecast`;
export const DASHBOARD_ENDPOINT = `${BASE_URL}/api/engine/dashboard-data`;
export const PLANNERWB_ENDPOINT = `${BASE_URL}/api/engine/planner-data`;
export const LIST_USERS_ENDPOINT = `${BASE_URL}/api/engine/list-users`;
export const CREATE_USER_ENDPOINT = `${BASE_URL}/api/engine/create-user`;
export const UPDATE_USER_STATUS_ENDPOINT = `${BASE_URL}/api/engine/user-status`;
export const DELETE_USER_ENDPOINT = `${BASE_URL}/api/engine/user`;

// Create axios instance with default config
export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = document.cookie.split('; ').find(row => row.startsWith('authToken='))?.split('=')[1];
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear cookies and redirect to login
      document.cookie.split(";").forEach((c) => {
        document.cookie = c
          .replace(/^ +/, "")
          .replace(/=.*/, `=;expires=${new Date().toUTCString()};path=/`);
      });
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// export const BASE_URL = 'https://4kast-backend-v2.cfapps.eu10-005.hana.ondemand.com';  Base URL for the server