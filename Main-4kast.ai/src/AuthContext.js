// src/AuthContext.js
import React, { createContext, useState, useContext } from 'react';
import Cookies from 'js-cookie';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return !!Cookies.get('authToken');
  });
  const [user, setUser] = useState(() => {
    const storedUsername = Cookies.get('username');
    return storedUsername ? { username: storedUsername } : null;
  });

  const login = (username, token) => {
    setIsAuthenticated(true);
    setUser({ username });
    // Store token in cookie with secure settings
    Cookies.set('authToken', token, { secure: true, sameSite: 'Strict' });
    Cookies.set('username', username, { secure: true, sameSite: 'Strict' });
  };

  const logout = () => {
    setIsAuthenticated(false);
    setUser(null);
    // Remove all auth-related cookies
    Cookies.remove('authToken');
    Cookies.remove('username');
  };

  const getAuthToken = () => {
    return Cookies.get('authToken');
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, logout, getAuthToken }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);