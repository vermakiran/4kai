// src/components/Login.js
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';
import { FaEye, FaEyeSlash } from 'react-icons/fa';
import Cookies from 'js-cookie';
import { LOGIN_ENDPOINT } from './config';
import '../App.css';

const Login = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  // Optional: Check cookies on component mount
  useEffect(() => {
    const storedUsername = Cookies.get('username');
    const storedToken = Cookies.get('authToken');
    console.log('Cookies on mount:', { storedUsername, storedToken });
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    if (errors[name]) {
      setErrors({
        ...errors,
        [name]: ''
      });
    }
  };

  const validateForm = () => {
    let tempErrors = {};
    if (!formData.username) tempErrors.username = 'Username is required';
    else if (formData.username.length < 3) tempErrors.username = 'Username must be at least 3 characters';
    
    if (!formData.password) tempErrors.password = 'Password is required';
    else if (formData.password.length < 6) tempErrors.password = 'Password must be at least 6 characters';

    setErrors(tempErrors);
    return Object.keys(tempErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    if (validateForm()) {
      try {
        const response = await fetch(LOGIN_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            username: formData.username,
            password: formData.password
          }),
        });
        if (response.ok) {
          const data = await response.json();
          const token = data.access_token;

          // Call login from AuthContext first to set localStorage
          login(formData.username);

          // Then set cookies as backup
          Cookies.set('username', formData.username, {
            expires: rememberMe ? 7 : 1,
            secure: true,
            sameSite: 'Strict'
          });
          if (token) {
            Cookies.set('authToken', token, {
              expires: rememberMe ? 7 : 1,
              secure: true,
              sameSite: 'Strict'
            });
          }

          navigate('/dashboard');
        } else {
          const errorData = await response.json();
          const errorMessage = errorData?.message || 'Invalid credentials';
          setErrors({ submit: errorMessage });
        }
      } catch (error) {
        console.error('Login error:', error);
        setErrors({ submit: 'Login failed. Please check your connection and try again.' });
      }
    }
    setIsSubmitting(false);
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const handleRememberMeChange = (e) => {
    setRememberMe(e.target.checked);
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-image"></div> {/* Placeholder for financial forecasting image */}
        <div className="login-box">
          <h2 className="login-title">Welcome to 4kast.ai</h2>
          <p className="login-subtitle">Please enter your credentials to login</p>
          
          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                type="text"
                id="username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Enter your username"
                className={errors.username ? 'input-error' : ''}
              />
              {errors.username && <span className="error">{errors.username}</span>}
            </div>

            <div className="form-group password-group">
              <label htmlFor="password">Password</label>
              <div className="password-wrapper">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Enter your password"
                  className={errors.password ? 'input-error' : ''}
                />
                <span className="password-toggle" onClick={togglePasswordVisibility}>
                  {showPassword ? <FaEyeSlash /> : <FaEye />}
                </span>
              </div>
              {errors.password && <span className="error">{errors.password}</span>}
            </div>

            <div className="form-options">
              <label className="remember-me">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={handleRememberMeChange}
                />
                Remember me
              </label>
              <a href="#" className="forgot-password" onClick={() => navigate('/forgot-password')}>
                Forgot Password?
              </a>
            </div>

            {errors.submit && <span className="error submit-error">{errors.submit}</span>}

            <button 
              type="submit" 
              className="login-button"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Logging in...' : 'Login'}
            </button>
          </form>
        </div>             
      </div>         
    </div>           
  );
};

export default Login;