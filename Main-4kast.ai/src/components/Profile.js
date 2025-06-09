import React, { useState, useEffect } from "react";
import "../App.css"; // Ensure your CSS file is correctly linked
import { FaUserCircle } from "react-icons/fa";
import Cookies from 'js-cookie'; // Import Cookies to access cookie data

function ProfileDashboard() {
  const [profileImage, setProfileImage] = useState(null);
  const [imageInput, setImageInput] = useState("");
  const [fileInput, setFileInput] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [newTask, setNewTask] = useState("");
  const [username, setUsername] = useState(""); // State to store username from cookie

  // Fetch username from cookie on component mount
  useEffect(() => {
    const storedUsername = Cookies.get('username');
    if (storedUsername) {
      setUsername(storedUsername);
    } else {
      setUsername("Guest"); // Default if no username is found
    }
    console.log('Username from cookie:', storedUsername);
  }, []);

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setProfileImage(URL.createObjectURL(file));
    }
  };

  // Function to add a task
  const addTask = () => {
    if (newTask.trim() !== "") {
      setTasks([...tasks, newTask]);
      setNewTask("");
    }
  };

  // Function to remove a task
  const removeTask = (index) => {
    const updatedTasks = tasks.filter((_, i) => i !== index);
    setTasks(updatedTasks);
  };

  // Function to handle image URL input
  const handleImageURLUpload = () => {
    if (imageInput.trim() !== "") {
      setProfileImage(imageInput);
      setImageInput("");
    }
  };

  // Function to remove the profile picture
  const removeProfilePicture = () => {
    setProfileImage(null);
  };

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-content">
          <div className="user-welcome">
            <h2>Welcome back,</h2>
            <h1>{username || "Guest"}</h1>
          </div>
          <div className="user-profile">
            {profileImage ? (
              <img src={profileImage} alt="Profile" className="header-profile-img" />
            ) : (
              <FaUserCircle className="header-user-icon" />
            )}
          </div>
        </div>
      </div>

      {/* Profile & Skills Section */}
      <div className="dashboard-content">
        {/* Profile Card */}
        <div className="profile-card">
          <div className="profile-image-container">
            {profileImage ? (
              <img src={profileImage} alt="Profile" className="profile-img" />
            ) : (
              <FaUserCircle className="default-avatar three-d-avatar" />
            )}
          </div>

          <h3>My Profile</h3>
          <p><strong>Name:</strong> {username || "Guest"}</p>
          <p><strong>Email:</strong> {username ? `${username}@example.com` : "guest@example.com"}</p>

          {/* Image Upload Section */}
          <div className="upload-section">
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileUpload} 
              style={{ display: "none" }} 
              ref={(fileInput) => setFileInput(fileInput)} 
            />
            <button className="upload-btn" onClick={() => fileInput.click()}>
              Upload Image
            </button>
            <input
              type="text"
              placeholder="Paste image URL"
              value={imageInput}
              onChange={(e) => setImageInput(e.target.value)}
            />
            <button className="upload-btn" onClick={handleImageURLUpload}>
              Upload from URL
            </button>
            {profileImage && (
              <button className="remove-btn" onClick={removeProfilePicture}>
                Remove Photo
              </button>
            )}
          </div>
        </div>

        <div className="reminder-card three-d-effect">
          <h3>⏳ Task Reminders</h3>
          <div className="task-input">
            <input
              type="text"
              placeholder="Enter a task..."
              value={newTask}
              onChange={(e) => setNewTask(e.target.value)}
            />
            <button className="add-task-btn" onClick={addTask}>Add Task</button>
          </div>
          <ul className="task-list">
            {tasks.map((task, index) => (
              <li key={index} className="task-item">
                {task}
                <button className="remove-task-btn" onClick={() => removeTask(index)}>❌</button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ProfileDashboard;