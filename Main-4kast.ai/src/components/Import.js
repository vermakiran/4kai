import React, { useState, useRef, useEffect, useCallback } from "react";
import Cookies from 'js-cookie';
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { UPLOAD_FILE_ENDPOINT, BASE_URL } from './config';
import "../App.css";

// Custom Alert Component
const CustomAlert = ({ message, type, isOpen, onClose, onConfirm }) => {
  if (!isOpen) return null;

  return (
    <div className="alert-overlay">
      <div className="alert-container">
        <div className={`alert-content ${type}`}>
          <p>{message}</p>
          {type === 'confirm' ? (
            <div className="alert-buttons">
              <button className="confirm-btn" onClick={onConfirm}>OK</button>
              <button className="cancel-btn" onClick={onClose}>Cancel</button>
            </div>
          ) : (
            <div className="alert-buttons">
              <button className="confirm-btn" onClick={onClose}>OK</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

function ImportData() {
  const [importLog, setImportLog] = useState("");
  const [fileList, setFileList] = useState([]);
  const [previewData, setPreviewData] = useState(null);
  const [previewHeaders, setPreviewHeaders] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const fileInputRef = useRef(null);
  const updateFileInputRef = useRef(null);
  
  // Alert state
  const [alert, setAlert] = useState({
    isOpen: false,
    message: "",
    type: "info", // 'info', 'success', 'error', 'confirm'
    onConfirm: null
  });

  const showAlert = (message, type = "info", onConfirm = null) => {
    setAlert({
      isOpen: true,
      message,
      type,
      onConfirm
    });
  };

  const closeAlert = () => {
    setAlert(prev => ({
      ...prev,
      isOpen: false,
      onConfirm: null
    }));
  };

  const handleConfirm = () => {
    if (alert.onConfirm) {
      alert.onConfirm();
    }
    closeAlert();
  };

  // Token retrieval function
const getAuthToken = useCallback(() => {
  const token = Cookies.get("authToken");
  if (!token) {
    console.error("Auth Token not found");
    showAlert("Please log in to continue.", "error");
    return null;
  }
  console.log("Retrieved Token:", token);  // Debugging line
  return token;
}, []);  // No dependencies since it's a simple getter

  const API_ENDPOINT = UPLOAD_FILE_ENDPOINT;

  console.log("API Endpoint configured as:", API_ENDPOINT);

const fetchFiles = useCallback(async () => {
  try {
    console.log("Attempting to fetch files...");

    // Get the token from cookies
    const token = getAuthToken();
    if (!token) {
      console.error("No auth token found, stopping fetch.");
      return;
    }
    debugger;

    const response = await fetch(`${BASE_URL}/api/engine/files`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    });

    console.log("Fetch response status:", response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Error fetching files: ${response.status} - ${errorText}`);
      showAlert(`Error fetching files: ${errorText}`, "error");
      return;
    }

    const data = await response.json();
    console.log("Files fetched successfully:", data.files);
    setFileList(data.files || []);
  } catch (error) {
    console.error("Error in fetchFiles:", error.message);
    showAlert("Error fetching files. Please try again.", "error");
  }
}, [getAuthToken]);  

useEffect(() => {
  // Log to check if useEffect is running repeatedly
  console.log("Fetching files on component mount");
  fetchFiles();
}, [fetchFiles]);  

  
  const handleImportClick = () => {
debugger;
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const fileType = file.name.split(".").pop().toLowerCase();
    if (fileType !== "csv" && fileType !== "xlsx" && fileType !== "xls") {
      showAlert("Unsupported file type. Please upload CSV or Excel files only.", "error");
      fileInputRef.current.value = "";
      return;
    }
    // Get the auth token from cookies
    const token = getAuthToken();
    if (!token) return;
  
    if (fileList.length > 0) {
      showAlert(
        `Uploading a new file will replace the current file "${fileList[0]}". Do you want to continue?`,
        "confirm",
        async () => {
          debugger;
          try {
            // Delete existing files
            for (const existingFile of fileList) {
              const response = await fetch(`${BASE_URL}/api/engine/deletefile`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  "Authorization": `Bearer ${token}`,  // Include the auth token
                },
                body: JSON.stringify({ filename: existingFile }),
              });
  
              if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to delete file: ${response.status} - ${errorText}`);
              }
            }
  
            // Upload new file
            await uploadFile(file);
            showAlert("File uploaded successfully!", "success");
          } catch (error) {
            console.error("Error replacing file:", error);
            showAlert("Error replacing file. Please try again.", "error");
          }
        }
      );
    } else {
      await uploadFile(file);
    }
  
    fileInputRef.current.value = "";
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const token = getAuthToken();
      if (!token) return;
  
      const response = await fetch(`${BASE_URL}/api/engine/uploadfile`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,  // Include the token properly
        },
        body: formData,
      });
  
      const result = await response.json();
  
      if (response.ok) {
        setImportLog(`File "${file.name}" uploaded successfully.`);
        showAlert("File uploaded successfully!", "success");
        fetchFiles();
      } else {
        throw new Error(result.error || "Upload failed");
      }
    } catch (error) {
      console.error("Upload error:", error);
      showAlert(`Upload failed: ${error.message}`, "error");
    }
  };
  

  const handlePreview = async (filename) => {
    try {

      const token = getAuthToken();
      if (!token) return;

      const url = `${BASE_URL}/api/engine/download/${encodeURIComponent(filename)}`;
      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`,  // Include the auth token
        },
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Preview failed with status ${response.status}: ${errorText}`);
      }

      const fileType = filename.split(".").pop().toLowerCase();
      if (fileType === "csv") {
        const text = await response.text();
        Papa.parse(text, {
          complete: (result) => {
            const headers = result.data[0] || [];
            const rows = result.data.slice(1, 11);
            setPreviewHeaders(headers);
            setPreviewData(rows);
            setShowPreview(true);
          },
          header: false,
          skipEmptyLines: true,
          error: (error) => {
            throw new Error(`Error parsing CSV: ${error.message}`);
          },
        });
      } else if (fileType === "xlsx" || fileType === "xls") {
        const blob = await response.blob();
        const arrayBuffer = await blob.arrayBuffer();
        const workbook = XLSX.read(new Uint8Array(arrayBuffer), { type: "array" });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        const headers = jsonData[0] || [];
        const rows = jsonData.slice(1, 11);
        setPreviewHeaders(headers);
        setPreviewData(rows);
        setShowPreview(true);
      }
    } catch (error) {
      console.error("Full preview error:", error);
      alert(`Error previewing file: ${error.message}`);
    }
  };

  const handleDelete = async (filename) => {
    showAlert(
      `Are you sure you want to delete "${filename}"?`,
      "confirm",
      async () => {
        try {
          // Retrieve the auth token from cookies
          const token = getAuthToken();
          if (!token) return;

          const url = `${BASE_URL}/api/engine/deletefile`;
          const response = await fetch(url, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,  // Include the token in the header
            },
            body: JSON.stringify({ filename: filename }),
          });
  
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Delete failed with status ${response.status}: ${errorText}`);
          }
  
          await response.json();
          showAlert(`File "${filename}" deleted successfully!`, "success");
  
          // Refresh the file list after deletion
          fetchFiles();
        } catch (error) {
          console.error("Delete error:", error);
          showAlert(`Error deleting file: ${error.message}`, "error");
        }
      }
    );
  };

  const closePreview = () => {
    setShowPreview(false);
    setPreviewData(null);
    setPreviewHeaders([]);
  };

  const handleUpdateClick = () => {
    if (fileList.length === 0) {
      showAlert("No existing file to update. Please upload a file first.", "error");
      return;
    }
    updateFileInputRef.current.click();
  };

  const handleUpdateFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const fileType = file.name.split(".").pop().toLowerCase();
    if (fileType !== "csv" && fileType !== "xlsx" && fileType !== "xls") {
      showAlert("Unsupported file type. Please upload CSV or Excel files only.", "error");
      updateFileInputRef.current.value = "";
      return;
    }
  
    try {
      // Get the auth token from cookies
      const token = getAuthToken();
      if (!token) return;
  
      // First confirm with the user
      showAlert(
        `This will update the current file "${fileList[0]}" with new data. Continue?`,
        "confirm",
        async () => {
          try {
            // Read the new file
            const newFileContent = await readFileContent(file);


            // Fetch the existing file content
            const existingFileResponse = await fetch(`${BASE_URL}/api/engine/download/${encodeURIComponent(fileList[0])}`, {
              method: "GET",
              headers: {
                "Authorization": `Bearer ${token}`,  // Include the token
              },
            });
            if (!existingFileResponse.ok) {
              throw new Error("Failed to read existing file");
            }
            const existingFileContent = await existingFileResponse.text();
  
            // Parse both files
            const existingData = Papa.parse(existingFileContent, { header: true }).data;
            const newData = Papa.parse(newFileContent, { header: true }).data;
  
            // Combine the data
            const combinedData = [...existingData, ...newData];
  
            // Create updated file content
            const updatedCsv = Papa.unparse(combinedData);
            const updatedBlob = new Blob([updatedCsv], { type: "text/csv" });
  
            // Delete the existing file first
            const deleteResponse = await fetch(`${BASE_URL}/api/engine/deletefile`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`,  // Include the token
              },
              body: JSON.stringify({ filename: fileList[0] }),
            });
            if (!deleteResponse.ok) {
              const errorText = await deleteResponse.text();
              throw new Error(`Delete failed with status ${deleteResponse.status}: ${errorText}`);
            }
  
            // Create form data with the same filename
            const formData = new FormData();
            formData.append("file", updatedBlob, fileList[0]);
  
            // Upload the updated file
            const response = await fetch(API_ENDPOINT, {
              method: "POST",
              headers: {
                "Authorization": `Bearer ${token}`,  // Include the token in the upload request
              },
              body: formData,
            });
  
            const result = await response.json();
  
            if (response.ok) {
              setImportLog(`File "${fileList[0]}" updated successfully with new data`);
              showAlert("File updated successfully!", "success");
              fetchFiles();
            } else {
              throw new Error(result.error || "Update failed");
            }
          } catch (error) {
            console.error("Update error:", error);
            showAlert(`Update failed: ${error.message}`, "error");
          }
        }
      );
    } catch (error) {
      console.error("Update error:", error);
      showAlert(`Update failed: ${error.message}`, "error");
    }
  
    updateFileInputRef.current.value = "";
  };

  const readFileContent = (file) => {
    return new Promise((resolve, reject) => {
      if (file.name.endsWith('.csv')) {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
      } else {
        // Handle Excel files
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const data = new Uint8Array(e.target.result);
            const workbook = XLSX.read(data, { type: 'array' });
            const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
            const csvContent = XLSX.utils.sheet_to_csv(firstSheet);
            resolve(csvContent);
          } catch (error) {
            reject(error);
          }
        };
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
      }
    });
  };

  const handleDownload = async (filename) => {
    try {
      const token = getAuthToken();
      if (!token) return;

      const response = await fetch(`${BASE_URL}/api/engine/download/${encodeURIComponent(filename)}`, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`,  // Include the token
        },
      });
  
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Download failed with status ${response.status}: ${errorText}`);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
  
      showAlert(`File "${filename}" downloaded successfully!`, "success");
    } catch (error) {
      console.error("Download error:", error);
      showAlert(`Error downloading file: ${error.message}`, "error");
    }
  };

  return (
    <div className="import-data-container">
      <CustomAlert
        isOpen={alert.isOpen}
        message={alert.message}
        type={alert.type}
        onClose={closeAlert}
        onConfirm={handleConfirm}
      />
      <div className="template-container">
        <div className="template-card">
          <h3 className="template-title">Upload Data</h3>
          <div className="template-content">
            <label>Import from</label>
            <select className="input-field">
              <option>Local device</option>
              <option>Online</option>
            </select>

            <div className="btn-group">
              <button className="cancel-btn">Cancel</button>
              <button className="save-btn" onClick={handleImportClick}>
                Import
              </button>
              <button 
                className="update-btn" 
                onClick={handleUpdateClick}
                title="Append new data to existing file"
              >
                Update
              </button>
            </div>

            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              onChange={handleFileChange}
              accept=".csv,.xlsx,.xls"
            />
            <input
              type="file"
              ref={updateFileInputRef}
              style={{ display: "none" }}
              onChange={handleUpdateFileChange}
              accept=".csv,.xlsx,.xls"
            />
          </div>
        </div>
      </div>

      <div className="import-log">
        <h3>Import log</h3>
        <p>{importLog || "No files uploaded yet"}</p>
      </div>

      <div className="file-list" style={{ marginTop: "20px" }}>
        <h3 className="UploadedFiles"><b>Current File</b></h3>
        {fileList.length > 0 ? (
          <div>
            <p style={{ marginBottom: "10px", color: "#666" }}>
              Note: Uploading a new file will replace the current one.
            </p>
            <div style={{ display: "flex", alignItems: "center" }}>
              <button
                onClick={() => handleDelete(fileList[0])}
                style={{
                  background: "none",
                  border: "none",
                  fontSize: "16px",
                  cursor: "pointer",
                  color: "#ff0000",
                  marginRight: "8px",
                  padding: "0",
                }}
                title={`Delete ${fileList[0]}`}
              >
                ✕
              </button>
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  handlePreview(fileList[0]);
                }}
                style={{ color: "#007bff", cursor: "pointer", marginRight: "8px" }}
              >
                {fileList[0]}
              </a>
              <button
                onClick={() => handleDownload(fileList[0])}
                className="download-btn"
                title="Download file"
              >
                ⬇
              </button>
            </div>
          </div>
        ) : (
          <p>No file uploaded yet</p>
        )}
      </div>

      {showPreview && previewData && (
        <div className="preview-overlay" style={{
          position: "fixed",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: "600px",
          height: "400px",
          background: "#fff",
          padding: "20px",
          border: "1px solid #ddd",
          borderRadius: "4px",
          boxShadow: "0 0 10px rgba(0,0,0,0.3)",
          zIndex: 1000,
          display: "flex",
          flexDirection: "column",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
            <h3 style={{ margin: 0 }}>Preview: First 10 Rows</h3>
            <button
              onClick={closePreview}
              style={{
                background: "none",
                border: "none",
                fontSize: "20px",
                cursor: "pointer",
                color: "#ff0000",
                padding: "0",
                lineHeight: "1",
              }}
              title="Close"
            >
              ✕
            </button>
          </div>
          <div style={{ flex: 1, overflowX: "auto", overflowY: "auto" }}>
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: "600px" }}>
              <thead>
                <tr>
                  {previewHeaders.map((header, index) => (
                    <th key={index} style={{ border: "1px solid #ddd", padding: "8px", background: "#f5f5f5" }}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewData.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex} style={{ border: "1px solid #ddd", padding: "8px" }}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImportData;