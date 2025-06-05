// src/components/SignUp.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const SignUp = () => {
  const [step, setStep] = useState(1); // Step 1: User details, Step 2: Pricing plans
  const [formData, setFormData] = useState({
    name: '',
    dob: '',
    companyName: '',
    email: '',
  });
  const [errors, setErrors] = useState({});
  const [selectedPlan, setSelectedPlan] = useState(null);
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

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
    if (!formData.name) tempErrors.name = 'Name is required';
    if (!formData.dob) tempErrors.dob = 'Date of Birth is required';
    if (!formData.companyName) tempErrors.companyName = 'Company Name is required';
    if (!formData.email) tempErrors.email = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(formData.email)) tempErrors.email = 'Email is invalid';

    setErrors(tempErrors);
    return Object.keys(tempErrors).length === 0;
  };

  const handleDetailsSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      setStep(2); // Move to pricing plans
    }
  };

  const handlePlanSelect = (plan) => {
    setSelectedPlan(plan);
    // Simulate sending login details to the user's email
    console.log(`Sending login details to ${formData.email}: Username: ${formData.email}, Password: [REDACTED]`);
    setMessage(`Thank you for choosing the ${plan} plan! Your login details have been sent to ${formData.email}.`);
    setTimeout(() => navigate('/login'), 3000); // Redirect back to login after 3 seconds
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-box">
          {step === 1 ? (
            <>
              <h2 className="login-title">Sign Up for 4kast.ai</h2>
              <p className="login-subtitle">Enter your details to create an account</p>
              
              <form onSubmit={handleDetailsSubmit} className="login-form">
                <div className="form-group">
                  <label htmlFor="name">Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="Enter your name"
                    className={errors.name ? 'input-error' : ''}
                  />
                  {errors.name && <span className="error">{errors.name}</span>}
                </div>

                <div className="form-group">
                  <label htmlFor="dob">Date of Birth</label>
                  <input
                    type="date"
                    id="dob"
                    name="dob"
                    value={formData.dob}
                    onChange={handleChange}
                    className={errors.dob ? 'input-error' : ''}
                  />
                  {errors.dob && <span className="error">{errors.dob}</span>}
                </div>

                <div className="form-group">
                  <label htmlFor="companyName">Company Name</label>
                  <input
                    type="text"
                    id="companyName"
                    name="companyName"
                    value={formData.companyName}
                    onChange={handleChange}
                    placeholder="Enter your company name"
                    className={errors.companyName ? 'input-error' : ''}
                  />
                  {errors.companyName && <span className="error">{errors.companyName}</span>}
                </div>

                <div className="form-group">
                  <label htmlFor="email">Email</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="Enter your email"
                    className={errors.email ? 'input-error' : ''}
                  />
                  {errors.email && <span className="error">{errors.email}</span>}
                </div>

                <button type="submit" className="login-button">
                  Next
                </button>
              </form>

              <p className="signup-link">
                Already have an account? <a href="#" onClick={() => navigate('/login')}>Login</a>
              </p>
            </>
          ) : (
            <>
              <h2 className="login-title">Choose Your Plan</h2>
              <p className="login-subtitle">Select a pricing plan to continue</p>

              <div className="pricing-plans">
                <div className="plan-card">
                  <h3>3 Months</h3>
                  <p>$99</p>
                  <button
                    className="plan-button"
                    onClick={() => handlePlanSelect('3 Months')}
                  >
                    Select
                  </button>
                </div>
                <div className="plan-card">
                  <h3>6 Months</h3>
                  <p>$179</p>
                  <button
                    className="plan-button"
                    onClick={() => handlePlanSelect('6 Months')}
                  >
                    Select
                  </button>
                </div>
                <div className="plan-card">
                  <h3>12 Months</h3>
                  <p>$299</p>
                  <button
                    className="plan-button"
                    onClick={() => handlePlanSelect('12 Months')}
                  >
                    Select
                  </button>
                </div>
              </div>

              {message && <span className="submit-message">{message}</span>}

              <p className="signup-link">
                <a href="#" onClick={() => setStep(1)}>Back</a>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default SignUp;