import React, { useState, useEffect, useRef } from "react";
import Cookies from "js-cookie";
import {
  LIST_USERS_ENDPOINT,
  CREATE_USER_ENDPOINT,
  UPDATE_USER_STATUS_ENDPOINT,
  DELETE_USER_ENDPOINT,
} from "./config";

// List of Roles
const roles = ["Admin", "Planner"];

function RowMenu({ onDeactivate, onDelete }) {
  const [open, setOpen] = useState(false);
  const [openUp, setOpenUp] = useState(false);
  const menuRef = useRef();
  const btnRef = useRef();

  useEffect(() => {
    function handleClick(e) {
      if (open && menuRef.current && !menuRef.current.contains(e.target) && btnRef.current && !btnRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  // When menu opens, check if there is enough space below, else open upwards
  useEffect(() => {
    if (open && btnRef.current && menuRef.current) {
      const btnRect = btnRef.current.getBoundingClientRect();
      const menuHeight = 90; // Adjust if you change the menu content/height
      const spaceBelow = window.innerHeight - btnRect.bottom;
      if (spaceBelow < menuHeight + 8) {
        setOpenUp(true);
      } else {
        setOpenUp(false);
      }
    }
  }, [open]);

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <button
        ref={btnRef}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "4px 10px",
          fontSize: "22px",
          color: "#555",
          borderRadius: "50%",
          transition: "background 0.2s",
        }}
        aria-label="Open menu"
        onClick={() => setOpen((o) => !o)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
      >
        ⋮
      </button>
      {open && (
        <div
          ref={menuRef}
          style={{
            position: "absolute",
            right: 0,
            [openUp ? "bottom" : "top"]: 32, // open up or down
            zIndex: 10,
            background: "#fff",
            border: "1px solid #f1f1f2",
            borderRadius: 12,
            minWidth: 140,
            boxShadow: "0 6px 32px #0002",
            padding: "4px 0",
            fontFamily: "inherit",
          }}
        >
          <button
            onClick={() => {
              onDeactivate();
              setOpen(false);
            }}
            style={{
              display: "block",
              width: "100%",
              background: "none",
              border: "none",
              textAlign: "left",
              padding: "12px 22px 12px 18px",
              cursor: "pointer",
              color: "#3b3b3b",
              fontSize: "15px",
              borderBottom: "1px solid #f4f4f6",
              borderRadius: "12px 12px 0 0",
            }}
          >
            Deactivate
          </button>
          <button
            onClick={() => {
              onDelete();
              setOpen(false);
            }}
            style={{
              display: "block",
              width: "100%",
              background: "none",
              border: "none",
              textAlign: "left",
              padding: "12px 22px 12px 18px",
              cursor: "pointer",
              color: "#e53e3e",
              fontSize: "15px",
              borderRadius: "0 0 12px 12px",
            }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}


function ConfirmModal({
  open,
  onCancel,
  onConfirm,
  title,
  message,
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  confirmColor = "#e53e3e",
}) {
  if (!open) return null;
  return (
    <div
      style={{
        position: "fixed",
        zIndex: 1000,
        left: 0,
        top: 0,
        width: "100vw",
        height: "100vh",
        background: "rgba(0,0,0,0.13)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          background: "#fff",
          padding: 36,
          borderRadius: 20,
          boxShadow: "0 10px 44px #0003",
          minWidth: 350,
          maxWidth: 380,
        }}
      >
        <h3 style={{ marginBottom: 10, fontWeight: 700, fontSize: 21 }}>
          {title}
        </h3>
        <p style={{ marginBottom: 28, color: "#444", fontSize: 16 }}>
          {message}
        </p>
        <div style={{ display: "flex", gap: 18, justifyContent: "flex-end" }}>
          {cancelLabel && (
            <button
              onClick={onCancel}
              style={{
                padding: "8px 28px",
                borderRadius: 8,
                border: "1px solid #bae6fd",
                background: "#e0f2fe",
                color: "#2563eb",
                fontWeight: 500,
                fontSize: 16,
                cursor: "pointer",
                transition: "background 0.13s, color 0.13s",
              }}
            >
              {cancelLabel}
            </button>
          )}
          <button
            onClick={onConfirm}
            style={{
              background: confirmColor,
              color: "#fff",
              border: "none",
              padding: "8px 28px",
              borderRadius: 8,
              cursor: "pointer",
              fontWeight: 600,
              fontSize: 16,
              boxShadow: `0 2px 12px ${confirmColor}22`,
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}


function StyledCheckbox({ checked, onChange }) {
  return (
    <label
      style={{
        display: "inline-flex",
        alignItems: "center",
        cursor: "pointer",
        width: 22,
        height: 22,
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        style={{
          display: "none",
        }}
      />
      <span
        style={{
          display: "inline-block",
          width: 22,
          height: 22,
          borderRadius: "7px",
          border: checked ? "2px solid #111827" : "2px solid #c3c7cd",
          background: checked ? "#111827" : "#fff",
          transition: "all 0.16s",
          boxShadow: checked
            ? "0 2px 6px #1118272c"
            : "0 1px 2px #a6abb72b",
          position: "relative",
        }}
      >
        {checked && (
          <svg
            width="14"
            height="14"
            style={{
              position: "absolute",
              left: 3,
              top: 2,
              pointerEvents: "none",
            }}
          >
            <polyline
              points="2,7 6,11 12,3"
              style={{
                fill: "none",
                stroke: "#fff",
                strokeWidth: 2.3,
                strokeLinecap: "round",
                strokeLinejoin: "round",
              }}
            />
          </svg>
        )}
      </span>
    </label>
  );
}

export default function IAMPage() {
  const [people, setPeople] = useState([]);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newRole, setNewRole] = useState(roles[0]);
  const [newEmpId, setNewEmpId] = useState("");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteUserId, setDeleteUserId] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false); 
  const [showErrorModal, setShowErrorModal] = useState({ open: false, message: "" });


  const getAuthHeaders = () => {
    const token = Cookies.get("authToken");
    if (!token) {
      console.error("Authentication token not found. Redirecting to login...");
      return null;
    }
    return {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    };
  };

  useEffect(() => {
    const fetchUsers = async () => {
      const headers = getAuthHeaders();
      if (!headers) return;
      try {
        const response = await fetch(LIST_USERS_ENDPOINT, { headers });
        if (!response.ok) throw new Error("Failed to fetch users");
        const data = await response.json();
        setPeople(data);
      } catch (error) {
        console.error("Error fetching users:", error);
      }
    };
    fetchUsers();
  }, []);

  const handleAddPerson = async () => {
    if (!(newName && newEmail && newEmpId)) {
      alert("Please fill in all fields.");
      return;
    }
    const headers = getAuthHeaders();
    if (!headers) return;
    const payload = {
      full_name: newName,
      email: newEmail,
      role: newRole,
      employee_id: newEmpId,
      password: "TempPassword123!",
    };
    try {
      const res = await fetch(CREATE_USER_ENDPOINT, {
        method: "POST",
        headers: headers,
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errText = await res.text();
        if (errText.includes("unique constraint violation") && errText.includes("EMPLOYEEID")) {
          setShowErrorModal({ open: true, message: "Employee ID already exists. Please use a unique Employee ID." });
        } else {
          setShowErrorModal({ open: true, message: "Add failed: " + errText });
        }
        return;
      }

      const createdUser = await res.json();
      setPeople((prev) => [...prev, createdUser]);
      setNewName("");
      setNewEmail("");
      setNewEmpId("");
      setNewRole(roles[0]);
    } catch (err) {
      setShowErrorModal({ open: true, message: "Add failed: " + err.message });
    }
  };

  const handleToggleStatus = async (userid, newStatus) => {
    const headers = getAuthHeaders();
    if (!headers) return;
    const patchHeaders = { Authorization: headers.Authorization };
    try {
      const url = `${UPDATE_USER_STATUS_ENDPOINT}/${userid}?isactive=${newStatus}`;
      const res = await fetch(url, {
        method: "PATCH",
        headers: patchHeaders,
      });
      if (!res.ok) throw new Error(await res.text());
      setPeople((prev) =>
        prev.map((p) =>
          p.userid === userid ? { ...p, isactive: newStatus } : p
        )
      );
    } catch (err) {
      alert("Status update failed: " + err.message);
    }
  };

  const openDeleteDialog = (userid) => {
    setDeleteUserId(userid);
    setShowDeleteModal(true);
  };

  const handleDeleteUser = async () => {
    if (!deleteUserId) return;
    const headers = getAuthHeaders();
    if (!headers) return;
    try {
      const res = await fetch(`${DELETE_USER_ENDPOINT}/${deleteUserId}`, {
        method: "DELETE",
        headers: { Authorization: headers.Authorization },
      });
      if (!res.ok) throw new Error(await res.text());
      setPeople((prev) => prev.filter((p) => p.userid !== deleteUserId));
      setShowDeleteModal(false);
      setDeleteUserId(null);
    } catch (err) {
      alert("Delete failed: " + err.message);
      setShowDeleteModal(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        fontFamily:
          'Inter, "SF Pro Display", "Segoe UI", Arial, system-ui, sans-serif',
      }}
    >
      <div
        style={{
          margin: "34px 0 18px 0",
          padding: "28px 30px 18px 30px",
          background: "#fff",
          borderRadius: "22px",
          boxShadow: "0 2px 24px #cfd5e322",
          border: "1px solid #eff0f4",
        }}
      >
        <h2 style={{ fontWeight: 700, fontSize: 27, margin: 0, color: "#15191f" }}>
          Identity & Access Management
        </h2>
        <div
          style={{
            color: "#adb0b6",
            fontSize: 16,
            margin: "4px 0 26px 1px",
          }}
        >
        </div>
        <div
          style={{
            display: "flex",
            gap: 20,
            alignItems: "center",
            marginBottom: 20,
          }}
        >
          <input
            type="text"
            placeholder="Full name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            style={{
              padding: "10px 16px",
              borderRadius: 9,
              border: "1.5px solid #e6e8ef",
              fontSize: 16,
              width: 190,
              background: "#fcfdfe",
              fontWeight: 500,
            }}
          />
          <input
            type="email"
            placeholder="Email"
            value={newEmail}
            onChange={(e) => setNewEmail(e.target.value)}
            style={{
              padding: "10px 16px",
              borderRadius: 9,
              border: "1.5px solid #e6e8ef",
              fontSize: 16,
              width: 220,
              background: "#fcfdfe",
              fontWeight: 500,
            }}
          />
          <select
            value={newRole}
            onChange={(e) => setNewRole(e.target.value)}
            style={{
              padding: "10px 16px",
              borderRadius: 9,
              border: "1.5px solid #e6e8ef",
              fontSize: 16,
              minWidth: 120,
              background: "#fcfdfe",
              fontWeight: 500,
            }}
          >
            {roles.map((role) => (
              <option key={role}>{role}</option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Employee ID"
            value={newEmpId}
            onChange={(e) => setNewEmpId(e.target.value)}
            style={{
              padding: "10px 16px",
              borderRadius: 9,
              border: "1.5px solid #e6e8ef",
              fontSize: 16,
              width: 150,
              background: "#fcfdfe",
              fontWeight: 500,
            }}
          />
          <button
            onClick={() => setShowAddModal(true)}
            style={{
              background: "#111827",
              color: "#fff",
              fontWeight: 600,
              border: "none",
              borderRadius: 10,
              padding: "11px 34px",
              fontSize: 16,
              marginLeft: 8,
              cursor: "pointer",
              letterSpacing: 0.1,
              boxShadow: "0 2px 12px #11182713",
            }}
          >
            Add user
          </button>
        </div>
      </div>

      <div
        style={{
          marginTop: 22,
          padding: "28px 30px 22px 30px",
          background: "#fff",
          borderRadius: "22px",
          boxShadow: "0 2px 24px #cfd5e322",
          border: "1px solid #eff0f4",
        }}
      >
        <h2
          style={{
            fontWeight: 700,
            fontSize: 21,
            margin: "0 0 18px 0",
            color: "#222",
            letterSpacing: 0,
          }}
        >
          Current users
        </h2>
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              borderSpacing: 0,
              fontSize: 16,
              background: "#fff",
              borderRadius: 18,
              overflow: "hidden",
              boxShadow: "0 2px 18px #ebedf526",
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "2.5px solid #edeef2",
                  color: "#232324",
                  fontWeight: 700,
                  background: "#fafbfc",
                  fontSize: 17,
                  letterSpacing: 0.03,
                }}
              >
                <th style={{ textAlign: "left", padding: "13px 16px" }}>Name</th>
                <th style={{ textAlign: "left", padding: "13px 16px" }}>Email</th>
                <th style={{ textAlign: "left", padding: "13px 16px" }}>Role</th>
                <th style={{ textAlign: "left", padding: "13px 16px" }}>Employee ID</th>
                <th style={{ textAlign: "left", padding: "13px 16px" }}>Org ID</th>
                <th style={{ textAlign: "center", padding: "13px 16px" }}>Active</th>
                <th style={{ width: 46 }}></th>
              </tr>
            </thead>
            <tbody>
              {people.map((p, i) => (
                <tr
                  key={p.userid}
                  style={{
                    borderBottom: "1.5px solid #f3f4f8",
                    background: i % 2 ? "#fcfcfd" : "#fff",
                    transition: "background 0.16s",
                  }}
                  onMouseEnter={e =>
                    (e.currentTarget.style.background = "#f4f8fd")
                  }
                  onMouseLeave={e =>
                    (e.currentTarget.style.background = i % 2 ? "#fcfcfd" : "#fff")
                  }
                >
                  <td style={{ padding: "13px 16px" }}>{p.full_name}</td>
                  <td
                    style={{
                      padding: "13px 16px",
                      fontFamily: "monospace",
                      color: "#324472",
                      letterSpacing: 0.03,
                    }}
                  >
                    {p.email}
                  </td>
                  <td style={{ padding: "13px 16px" }}>{p.role}</td>
                  <td style={{ padding: "13px 16px" }}>{p.employee_id}</td>
                  <td style={{ padding: "13px 16px" }}>{`${p.org_id}`}</td>
                  <td style={{ textAlign: "center", padding: "13px 16px" }}>
                    <StyledCheckbox
                      checked={p.isactive}
                      onChange={() =>
                        handleToggleStatus(p.userid, !p.isactive)
                      }
                    />
                  </td>
                  <td style={{ textAlign: "center" }}>
                    <RowMenu
                      onDeactivate={() => handleToggleStatus(p.userid, false)}
                      onDelete={() => openDeleteDialog(p.userid)}
                    />
                  </td>
                </tr>
              ))}
              {people.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ textAlign: "center", color: "#888" }}>
                    No users found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <ConfirmModal
        open={showDeleteModal}
        onCancel={() => {
          setShowDeleteModal(false);
          setDeleteUserId(null);
        }}
        onConfirm={handleDeleteUser}
        title="Delete user?"
        message="Are you sure you want to delete this user? This cannot be undone."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        confirmColor="#e53e3e"
      />

      <ConfirmModal
        open={showAddModal}
        onCancel={() => setShowAddModal(false)}
        onConfirm={() => {
          setShowAddModal(false);
          handleAddPerson();
        }}
        title="Add new user?"
        message="Are you sure you want to add this user?"
        confirmLabel="Add"
        cancelLabel="Cancel"
        confirmColor="#10b981"
      />

      <ConfirmModal
        open={showErrorModal.open}
        onCancel={() => setShowErrorModal({ open: false, message: "" })}
        onConfirm={() => setShowErrorModal({ open: false, message: "" })}
        title="Error"
        message={showErrorModal.message}
        confirmLabel="OK"
        cancelLabel=""    // Only show one button for error modal
        confirmColor="#2563eb"
      />

    </div>
  );
}
