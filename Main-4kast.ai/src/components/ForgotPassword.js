// src/components/ForgotPassword.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const ForgotPassword = () => {
  const [comment, setComment] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      console.log('Sending email to sarveshu.et20@rvce.edu.in with comment:', comment);
      const newPassword = 'newPassword123'; // Simulated new password
      console.log(`New password generated for user: ${newPassword}`);
      setMessage('Password reset request sent! Check your email for the new password.');
      setTimeout(() => navigate('/login'), 3000); // Redirect back to login after 3 seconds
    } catch (error) {
      setMessage('Failed to send request. Please try again.');
    }
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-box">
          <h2 className="login-title">Forgot Password</h2>
          <p className="login-subtitle">Enter a comment to request a password reset</p>
          
          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="comment">Comment</label>
              <textarea
                id="comment"
                name="comment"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="Enter your comment here..."
                className="comment-input"
                rows="4"
              />
            </div>

            {message && <span className="submit-message">{message}</span>}

            <button type="submit" className="login-button">
              Send
            </button>
          </form>

          <p className="signup-link">
            <a href="#" onClick={() => navigate('/login')}>Back to Login</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;