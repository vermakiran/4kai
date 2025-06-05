import React, { useState } from "react";
import Papa from "papaparse";
import "../App.css"
// Initial People Data
const initialPeople = [
  { id: 1, name: "Alice Johnson", email: "alice@example.com", department: "HR", role: "Manager", empId: "12301001" },
  { id: 2, name: "Bob Smith", email: "bob@example.com", department: "IT", role: "Operator", empId: "12302002" },
  { id: 3, name: "Charlie Brown", email: "charlie@example.com", department: "Sales", role: "Manager", empId: "12303003" },
  { id: 4, name: "David Lee", email: "david@example.com", department: "HR", role: "Operator", empId: "12301004" },
  { id: 5, name: "Emma Wilson", email: "emma@example.com", department: "IT", role: "Manager", empId: "12302005" },
];
 
// List of Departments & Roles
const departments = ["HR", "IT", "Sales", "Marketing", "Finance", "Logistics", "R&D", "Customer Support", "Operations"];
const roles = ["Admin", "User", "Super User"];
 
 
// Mapping Departments to Warehouse IDs
const warehouseIds = {
  HR: "01",
  IT: "02",
  Sales: "03",
  Marketing: "04",
  Finance: "05",
  Logistics: "06",
  "R&D": "07",
  "Customer Support": "08",
  Operations: "09",
};
 
const IAMPage = () => {
  const [people, setPeople] = useState(initialPeople);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newDepartment, setNewDepartment] = useState(departments[0]);
  const [newRole, setNewRole] = useState(roles[0]);
  const [customEmpNum, setCustomEmpNum] = useState("");
 
  // Function to Generate Employee ID
  const generateEmployeeId = (department, empNum) => {
    const orgId = "123";
    const warehouseId = warehouseIds[department] || "00";
    const empNumber = empNum.padStart(3, "0");
    return `${orgId}${warehouseId}${empNumber}`;
  };
 
  // Function to Add a New Person
  const handleAddPerson = () => {
    if (newName && newEmail && customEmpNum.length === 3) {
      const newPerson = {
        id: people.length + 1,
        id: Date.now(),
        name: newName,
        email: newEmail,
        department: newDepartment,
        role: newRole,
        empId: generateEmployeeId(newDepartment, customEmpNum),
      };
      setPeople(prevPeople => [...prevPeople, newPerson]);
      setNewName("");
      setNewEmail("");
      setCustomEmpNum("");
    } else {
      alert("Please enter a valid 3-digit Employee Number.");
    }
  };
 
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (result) => {
          const newPeople = result.data // Changed variable name to newPeople
            .filter(row =>
              row.Name &&
              row.Email &&
              row.Department &&
              row.Role &&
              row["Employee ID"]
            )
            .map((row, index) => {
              const empIdDigits = String(row["Employee ID"]).slice(-3).padStart(3, "0");
 
              return {
                id: Date.now() + index,
                name: row.Name,
                email: row.Email,
                department: row.Department,
                role: row.Role,
                empId: generateEmployeeId(row.Department, empIdDigits),
              };
            });
 
          if (newPeople.length > 0) { // Changed to newPeople
            setPeople(prev => [...prev, ...newPeople]); // Changed to newPeople
          } else {
            alert("No valid records found in CSV");
          }
 
          event.target.value = "";
        },
        error: (err) => {
          console.error('CSV Error:', err);
          alert('Error parsing CSV file');
        }
      });
    }
  };
 
  // Function to Download CSV
  const handleDownloadCSV = () => {
    const csvData = [
      ["Name", "Email", "Department", "Role", "Employee ID"],
      ...people.map((p) => [p.name, p.email, p.department, p.role, p.empId]),
    ];
    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "employees.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };
 
  // Function to Get Role Counts for Each Department
  const getRoleCounts = (dept) => {
    return {
      managers: people.filter((p) => p.department === dept && p.role === "Manager").length,
      operators: people.filter((p) => p.department === dept && p.role === "Operator").length,
    };
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
            <select value={newDepartment} onChange={(e) => setNewDepartment(e.target.value)}>
              {departments.map((dept) => (
                <option key={dept} value={dept}>
                  {dept}
                </option>
              ))}
            </select>
            <select value={newRole} onChange={(e) => setNewRole(e.target.value)}>
              {roles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
            <input
              type="text"
              placeholder="Last 3 digits on emp id"
              value={customEmpNum}
              maxLength="3"
              onChange={(e) => setCustomEmpNum(e.target.value.replace(/\D/, ""))}
            />
          </div>
          <button onClick={handleAddPerson}>Add Person</button>
        </div>
 
        <div className="file-upload-container">
          <label className="custom-file-upload">
            <input type="file" accept=".csv" onChange={handleFileUpload} />
            Upload CSV
          </label>
          <button onClick={handleDownloadCSV}>Download CSV</button>
        </div>
 
        {/* People List */}
        <div className="table-container">
          <h3 className="section-title">📋 Employee List</h3>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Department</th>
                <th>Role</th>
                <th>Employee ID</th>
              </tr>
            </thead>
            <tbody>
              {people.map((person) => (
                <tr key={person.id}>
                  <td>{person.name}</td>
                  <td>{person.email}</td>
                  <td>{person.department}</td>
                  <td>{person.role}</td>
                  <td>{person.empId}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
 
      {/* Department Overview */}
      <div className="card">
        <h2 className="card-title">🏢 Department Overview</h2>
        <div className="department-grid">
          {departments.map((dept) => {
            const { managers, operators } = getRoleCounts(dept);
            return (
              <div key={dept} className="department-card">
                <h3>{dept}</h3>
                <p>✅ {managers + operators} Employees</p>
                <p>👔 {managers} Managers</p>
                <p>⚙️ {operators} Operators</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
 
export default IAMPage;
 
 