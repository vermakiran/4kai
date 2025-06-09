import React, { useState } from "react";
import "../App.css"

// Initial People Data
const initialPeople = [
  { id: 1, name: "John Smith", email: "john.smith@4kast.ai", role: "Admin", empId: "4K001" },
  { id: 2, name: "Sarah Johnson", email: "sarah.johnson@4kast.ai", role: "Planner", empId: "4K002" },
  { id: 3, name: "Michael Chen", email: "michael.chen@4kast.ai", role: "Planner", empId: "4K003" }
];

// List of Roles
const roles = ["Admin", "Planner"];

const IAMPage = () => {
  const [people, setPeople] = useState(initialPeople);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newRole, setNewRole] = useState(roles[0]);
  const [newEmpId, setNewEmpId] = useState("");

  // Function to Add a New Person
  const handleAddPerson = () => {
    if (newName && newEmail && newEmpId) {
      const newPerson = {
        id: Date.now(),
        name: newName,
        email: newEmail,
        role: newRole,
        empId: newEmpId,
      };
      setPeople(prevPeople => [...prevPeople, newPerson]);
      setNewName("");
      setNewEmail("");
      setNewEmpId("");
    } else {
      alert("Please fill in all fields.");
    }
  };

  return (
    <div className="container">
      <h1 className="page-title">IAM - Identity and Access Management</h1>

      {/* Add Person Form */}
      <div className="card">
        <h2 className="card-title">👥 People Management</h2>
        <div className="form-section">
          <div className="input-group">
            <input type="text" placeholder="Full Name" value={newName} onChange={(e) => setNewName(e.target.value)} />
            <input type="email" placeholder="Email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
            <select value={newRole} onChange={(e) => setNewRole(e.target.value)}>
              {roles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
            <input
              type="text"
              placeholder="Employee ID"
              value={newEmpId}
              onChange={(e) => setNewEmpId(e.target.value)}
            />
          </div>
          <button onClick={handleAddPerson}>Add Person</button>
        </div>

        {/* People List */}
        <div className="table-container">
          <h3 className="section-title">📋 Employee List</h3>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Role</th>
                <th>Employee ID</th>
              </tr>
            </thead>
            <tbody>
              {people.map((person) => (
                <tr key={person.id}>
                  <td>{person.name}</td>
                  <td>{person.email}</td>
                  <td>{person.role}</td>
                  <td>{person.empId}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default IAMPage;
 
 