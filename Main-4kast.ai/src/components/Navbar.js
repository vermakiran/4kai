import React, { useState, useEffect } from "react";
import { Navbar, Container, Nav, Form, FormControl } from "react-bootstrap";
import { FaBell, FaCog, FaUser } from "react-icons/fa";
import { Link } from "react-router-dom"; // Import Link for navigation
import Cookies from 'js-cookie'; // Import Cookies to access cookie data
import "../App.css"; 

function CustomNavbar() {
  const [username, setUsername] = useState("");

  // Fetch username from cookie on component mount
  useEffect(() => {
    const storedUsername = Cookies.get('username');
    if (storedUsername) {
      setUsername(storedUsername);
    } else {
      setUsername("Guest"); // Default if no username is found
    }
    console.log('Username from cookie in Navbar:', storedUsername);
  }, []);

  return (
    <Navbar expand="lg" className="custom-navbar">
      <Container fluid>
        {/* Left Side: Logo */}
        <Navbar.Brand href="#home" className="navbar-logo">
          4kast.ai<span className="red-dot"></span>
        </Navbar.Brand>

        {/* Center: Search Bar */}
        <Form className="search-bar">
          <FormControl
            type="search"
            placeholder="Search anything here..."
            className="search-input"
          />
        </Form>

        {/* Right Side: Icons */}
        <Nav className="navbar-icons">
          <Link to="/notifications" className="icon-link"><FaBell className="icon" /></Link>
          <Link to="/settings" className="icon-link"><FaCog className="icon" /></Link>
          <Link to="/profile" className="icon-link">
            <div className="user-profile">
              <FaUser className="user-icon" />
              <span className="navbar-user">Hi, {username || "Guest"}</span>
            </div>
          </Link>
        </Nav>
      </Container>
    </Navbar>
  );
}

export default CustomNavbar;