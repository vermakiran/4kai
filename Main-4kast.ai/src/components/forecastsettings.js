import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Cookies from "js-cookie";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { UPLOAD_CLEANED_DATA_ENDPOINT, MODELS_ENDPOINT, BASE_URL, RUN_FORECAST_ENDPOINT } from './config';
import "../App.css";

// Add custom CSS for forecast displays
const customStyles = `
  .future-forecasts table, .imported-data-table table, .model-results table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    font-size: 14px;
    font-family: Arial, sans-serif;
  }
  
  .future-forecasts th, .future-forecasts td,
  .imported-data-table th, .imported-data-table td,
  .model-results th, .model-results td {
    border: 1px solid #d4d4d4;
    padding: 6px 8px;
    height: 21px;
    min-width: 96px;
    max-width: 200px;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    background-color: #ffffff;
  }
  
  .future-forecasts th, .imported-data-table th, .model-results th {
    background-color: #f8f9fa;
    font-weight: normal;
    color: #000000;
    height: 20px;
    border-bottom: 2px solid #d4d4d4;
  }
  
  .future-forecasts tr:hover td, .imported-data-table tr:hover td, .model-results tr:hover td {
    background-color: #f5f5f5;
  }
  
  .future-forecasts td:hover, .imported-data-table td:hover, .model-results td:hover {
    background-color: #e8e8e8;
  }
  
  .future-forecasts input, .imported-data-table input, .model-results input {
    width: 100%;
    height: 20px;
    border: none;
    padding: 2px 4px;
    font-size: 14px;
    font-family: Arial, sans-serif;
    background: transparent;
  }
  
  .future-forecasts input:focus, .imported-data-table input:focus, .model-results input:focus {
    outline: 2px solid #1a73e8;
    background: #ffffff;
  }
  
  /* Add scrolling for wide tables */
  .table-container {
    overflow-x: auto;
    max-width: 100%;
  }
  
  /* Excel-like selection styling */
  .selected-cell {
    outline: 2px solid #1a73e8;
    outline-offset: -2px;
  }
  
  /* Rest of your existing styles... */
  .item-selector, .combination-selector {
    margin-bottom: 15px;
  }
  
  .item-selector select, .combination-selector select {
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #ccc;
    min-width: 250px;
  }
  
  .item-forecast, .combo-forecast {
    margin-bottom: 30px;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: white;
  }
  
  .unknown-forecast-type {
    padding: 15px;
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
    border-radius: 4px;
    color: #856404;
  }
  
  .unknown-forecast-type pre {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 4px;
    overflow-x: auto;
  }

  .download-button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background-color 0.3s ease;
    margin-top: 15px;
  }

  .download-button:hover {
    background-color: #45a049;
  }

  .download-button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }

  .download-button svg {
    width: 16px;
    height: 16px;
  }

  .forecast-results-section {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 20px;
    margin-top: 20px;
  }

  .forecast-results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
  }

  .forecast-results-title {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin: 0;
  }
`;

// Helper function to generate future dates from current date
const generateFutureDatesFromCurrent = (horizon, frequency = 'daily') => {
  const currentDate = new Date();
  const futureDates = [];
  
  for (let i = 1; i <= horizon; i++) {
    const futureDate = new Date(currentDate);
    
    switch (frequency.toLowerCase()) {
      case 'daily':
        futureDate.setDate(currentDate.getDate() + i);
        break;
      case 'weekly':
        futureDate.setDate(currentDate.getDate() + (i * 7));
        break;
      case 'monthly':
        futureDate.setMonth(currentDate.getMonth() + i);
        break;
      case 'quarterly':
        futureDate.setMonth(currentDate.getMonth() + (i * 3));
        break;
      case 'yearly':
        futureDate.setFullYear(currentDate.getFullYear() + i);
        break;
      default:
        futureDate.setDate(currentDate.getDate() + i);
    }
    
    futureDates.push(futureDate);
  }
  
  return futureDates;
};

// Helper function to detect data frequency from historical dates
const detectDataFrequency = (dates) => {
  if (!dates || dates.length < 2) return 'daily';
  
  const sortedDates = dates.map(d => new Date(d)).sort((a, b) => a - b);
  const diffs = [];
  
  for (let i = 1; i < Math.min(sortedDates.length, 10); i++) {
    const diff = (sortedDates[i] - sortedDates[i-1]) / (1000 * 60 * 60 * 24); // days
    diffs.push(diff);
  }
  
  const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  
  if (avgDiff <= 1.5) return 'daily';
  if (avgDiff >= 6 && avgDiff <= 8) return 'weekly';
  if (avgDiff >= 28 && avgDiff <= 32) return 'monthly';
  if (avgDiff >= 88 && avgDiff <= 95) return 'quarterly';
  if (avgDiff >= 360 && avgDiff <= 370) return 'yearly';
  
  return 'daily'; // fallback
};

const ForecastSettings = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [timeBucket, setTimeBucket] = useState("Daily");
  const [forecastHorizon, setForecastHorizon] = useState("0");
  const [forecastLock, setForecastLock] = useState("0");
  const [granularity, setGranularity] = useState("Overall");
  const [aggregation, setAggregation] = useState("Sum");
  const [forecastMethod, setForecastMethod] = useState("Best Fit");
  const [canRunModel, setCanRunModel] = useState(false);

  const [uploadedDatasets, setUploadedDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [importedData, setImportedData] = useState([]);
  const [columnNames, setColumnNames] = useState([]);
  const [showColumns, setShowColumns] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [columnMappings, setColumnMappings] = useState({});
  const [timeDependentVariables, setTimeDependentVariables] = useState([]);
  const [omittedColumns, setOmittedColumns] = useState([]);
  const [models, setModels] = useState([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [dataSummary, setDataSummary] = useState(null);
  const [showSummary, setShowSummary] = useState(false);
  const columnSelectionRef = useRef(null);
  const uploadButtonRef = useRef(null);
  const [forecastResults, setForecastResults] = useState(null);
  const [showForecastResults, setShowForecastResults] = useState(false);

  const mandatoryFields = ["Date", "Demand", "StoreID", "ProductID"];

  const measureOptions = ["Sales History", "Inventory Levels", "Demand Trends"];

  const granularityOptions = [
    "Overall",
    "ProductID-wise",
    "StoreID-ProductID Combination",
  ];
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
  
     // Token retrieval function
   const getAuthToken = useCallback(() => {
     const token = Cookies.get("authToken");
     if (!token) {
       console.error("Auth Token not found");
       showAlert("Please log in to continue.", "error");
       return null;
     }
    //  console.log("Retrieved Token:", token);  // Debugging line
     return token;
   }, []);  // No dependencies since it's a simple getter

    const fetchFiles = async () => {
      try {
        // Get the token from cookies
        const token = getAuthToken();
        if (!token) {
          console.error("No auth token found, stopping fetch.");
          return;
        }
        const response = await fetch(`${BASE_URL}/api/engine/files`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
          },
        });
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Failed to fetch file list: ${response.status} - ${errorText}`);
        }
        const data = await response.json();
        // console.log(data); // Debugging line
        setUploadedDatasets(data.files || []);
      } catch (error) {
        console.error("Error fetching files:", error);
        showAlert("Error fetching files. Please try again.", "error");
      }
    };

  useEffect(() => {
    fetchFiles();
    
    // Fetch available forecasting models
    const fetchModels = async () => {
      setIsLoadingModels(true);
      try {
        const token = getAuthToken();
        if (!token) {
          console.error("No auth token found, stopping fetch.");
          return;
        }
        const response = await fetch(MODELS_ENDPOINT, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
          },
        });
        // const response = await fetch(MODELS_ENDPOINT);
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Failed to fetch models: ${response.status} - ${errorText}`);
        }
        const data = await response.json();
        console.log(data); // Debugging line
        setModels(data.models || []);
      } catch (error) {
        console.error("Error fetching models:", error);
        // Fallback to default models if API fails
        setModels([
          { name: "Seasonal History", createdBy: "Arjun", date: "7-Nov-2024" },
          { name: "Intermittent History", createdBy: "Bhavana", date: "7-Nov-2024" },
          { name: "Advanced Forecasting", createdBy: "Vishnu", date: "7-Nov-2024" },
        ]);
      } finally {
        setIsLoadingModels(false);
      }
    };
    fetchModels();
  }, []);

  const handleDatasetChange = async (event) => {
    const fileName = event.target.value;
    setSelectedDataset(fileName);
    setImportedData([]);
    setColumnNames([]);
    setShowColumns(false);
    setShowTable(false);
    setColumnMappings({});
    setTimeDependentVariables([]);

    if (fileName) {
      try {
        const token = getAuthToken();
        if (!token) {
          console.error("No auth token found, stopping fetch.");
          return;
        }
        const response = await fetch(`${BASE_URL}/api/engine/download/${encodeURIComponent(fileName)}`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
          },
        });
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Fetch failed with status ${response.status}: ${errorText}`);
        }

        const fileType = fileName.split(".").pop().toLowerCase();
        if (fileType === "csv") {
          const text = await response.text();
          Papa.parse(text, {
            complete: (result) => {
              if (result.data && result.data.length > 0) {
                const headers = result.data.length > 1 && typeof result.data[0] === "object"
                  ? Object.keys(result.data[0] || {})
                  : result.data[0] || [];
                const rows = result.data.slice(1).filter(row => row && Object.keys(row).length > 0);
                setColumnNames(Array.isArray(headers) ? headers : [headers]);
                setImportedData(rows);
                setShowColumns(true);
                setTimeout(() => {
                  if (columnSelectionRef.current) {
                    columnSelectionRef.current.scrollIntoView({ behavior: "smooth" });
                  }
                }, 100);
              } else {
                showAlert("No data or headers found in the selected CSV file.", "error");
              }
            },
            header: true,
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
          if (jsonData.length > 0) {
            const headers = jsonData[0] || [];
            const rows = jsonData.slice(1).filter(row => row && row.length > 0);
            setColumnNames(Array.isArray(headers) ? headers : [headers]);
            setImportedData(rows);
            setShowColumns(true);
            setTimeout(() => {
              if (columnSelectionRef.current) {
                columnSelectionRef.current.scrollIntoView({ behavior: "smooth" });
              }
            }, 100);
          } else {
            showAlert("No data or headers found in the selected Excel file.", "error");
          }
        }
      } catch (error) {
        console.error("Full preview error:", error);
        showAlert(`Error previewing file: ${error.message}`, "error");
      }
    }
  };

  const handleDeleteColumn = (column) => {
    if (!Array.isArray(columnNames) || !columnNames.includes(column)) return;
    const updatedColumns = columnNames.filter((col) => col !== column);
    setColumnNames(updatedColumns);
    setOmittedColumns([...omittedColumns, { name: column, data: importedData.map(row => row[column]) }]);
  };

  const handleRetrieveColumn = (column) => {
    const omittedColumn = omittedColumns.find(col => col.name === column);
    if (!omittedColumn) return;
    
    const updatedOmittedColumns = omittedColumns.filter(col => col.name !== column);
    const updatedColumns = [...columnNames, column];
    const updatedData = importedData.map((row, index) => ({
      ...row,
      [column]: omittedColumn.data[index]
    }));
    
    setOmittedColumns(updatedOmittedColumns);
    setColumnNames(updatedColumns);
    setImportedData(updatedData);
  };

  const handleMappingChange = (mandatoryField, selectedColumn) => {
    setColumnMappings({
      ...columnMappings,
      [mandatoryField]: selectedColumn,
    });
  };

  const areMandatoryFieldsMapped = () => {
    return mandatoryFields.every(field =>
      columnMappings[field] && columnNames.includes(columnMappings[field])
    );
  };

  const handleShowPreview = () => {
    if (areMandatoryFieldsMapped()) {
      setShowTable(true);
    } else {
      showAlert("Please map the fields first", "error");
    }
  };

  const handleFieldChange = (index, field, value) => {
    const updatedData = [...importedData];
    updatedData[index][field] = value;
    setImportedData(updatedData);
  };

  const handleTimeDependentVariablesChange = (event) => {
    const { value, checked } = event.target;
    setTimeDependentVariables((prev) =>
      checked ? [...prev, value] : prev.filter((item) => item !== value)
    );
    if (checked && uploadButtonRef.current) {
      uploadButtonRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleClearTimeDependentVariables = () => {
    setTimeDependentVariables([]);
  };

  const resetForm = () => {
    setTimeBucket("Daily");
    setForecastHorizon("0");
    setForecastLock("0");
    setGranularity("Overall");
    setAggregation("Sum");
    setForecastMethod("Best Fit");
    setSelectedDataset("");
    setImportedData([]);
    setColumnNames([]);
    setShowColumns(false);
    setShowTable(false);
    setColumnMappings({});
    setTimeDependentVariables([]);
    setSelectedModel("");
  };

  const handleUploadDataForCleaning = async () => {
    if (!areMandatoryFieldsMapped()) {
      showAlert("Please map all mandatory fields before uploading.", "error");
      return;
    }

    try {
      // Create a new dataset with only the mapped fields and time-dependent variables
      const finalData = importedData.map(row => {
        const cleanedRow = {};
        // First add mandatory fields with proper mapping
        mandatoryFields.forEach(field => {
          if (columnMappings[field]) {
            cleanedRow[field] = row[columnMappings[field]];
          }
        });
        // Then add time-dependent variables
        timeDependentVariables.forEach(varName => {
          if (row[varName] !== undefined) {
            cleanedRow[varName] = row[varName];
          }
        });
        return cleanedRow;
      }).filter(row => {
        // Filter out rows with missing mandatory fields
        return mandatoryFields.every(field => row[field] !== undefined && row[field] !== '');
      });

      if (finalData.length === 0) {
        throw new Error("No valid data rows after cleaning. Please check your data and mappings.");
      }

      console.log("Cleaned data sample:", finalData.slice(0, 5));

      const csv = Papa.unparse(finalData);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const formData = new FormData();
      const timestamp = new Date().toISOString().replace(/[:.-]/g, "_");
      const newFilename = `${selectedDataset.split('.')[0]}_cleaned_${timestamp}.csv`;
      formData.append("file", blob, newFilename);
      
      // Add metadata with complete information
      formData.append("granularity", granularity || "Overall");
      formData.append("timeBucket", timeBucket || "Daily");
      formData.append("forecastHorizon", forecastHorizon || "30");
      formData.append("columnMappings", JSON.stringify(columnMappings || {}));
      formData.append("timeDependentVariables", JSON.stringify(timeDependentVariables || []));
      formData.append("organizationId", organizationId || "default");  // Add organization ID

      // Debug: Log form data contents
      console.log("FormData contents:");
      for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + (pair[1] instanceof Blob ? 'Blob data' : pair[1]));
      }

      setIsLoading(true);
      const token = getAuthToken();
      if (!token) {
        console.error("No auth token found, stopping fetch.");
        return;
      }
      
      // Debug: Log request details
      console.log("Making request to:", UPLOAD_CLEANED_DATA_ENDPOINT);
      console.log("With token:", token.substring(0, 10) + "...");
      
      const response = await fetch(UPLOAD_CLEANED_DATA_ENDPOINT, {
        method: "POST",
        headers: {
          // Don't set Content-Type when sending FormData
          // Browser will automatically set it with the correct boundary
          "Authorization": `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed with status ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      
      if (result.status === "success") {
        showAlert(`Data processed successfully! New file '${newFilename}' created.`);
        // Refresh the datasets list to include the new file
        fetchFiles();
        // Show statistics or summary if available
        if (result.summary) {
          setDataSummary(result.summary);
          setShowSummary(true);
        }
      } else {
        throw new Error(result.message || "Unknown error occurred");
      }
    } catch (error) {
      console.error("Processing error:", error);
      showAlert(`Error processing data: ${error.message}`, "error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunModel = async () => {
    if (!canRunModel) {
      showAlert("Please fill all required fields first.", "error");
      return;
    }
    
    setIsLoading(true);
    
    try {
      const token = getAuthToken();
      if (!token) {
        console.error("No auth token found");
        return;
      }

      // Prepare the request body
      const requestBody = {
        filename: selectedDataset,
        granularity: granularity,
        forecastHorizon: parseInt(forecastHorizon),
        timeBucket: timeBucket,
        forecastLock: parseInt(forecastLock),
        selectedModels: forecastMethod === "Best Fit" ? [] : [selectedModel], // Empty array for Best Fit, selected model otherwise
        timeDependentVariables: timeDependentVariables,
        columnMappings: columnMappings
      };

      console.log("Sending forecast request:", requestBody);
      
      const response = await fetch(RUN_FORECAST_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify(requestBody)
      });
      
      // First check if the response is OK
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API responded with status ${response.status}: ${errorText}`);
      }
      
      // Try to parse the JSON response
      const result = await response.json();
      
      // Check if the response indicates an error
      if (result.status === "error") {
        throw new Error(result.message || "Unknown error occurred");
      }
      
      // COMPREHENSIVE DEBUG - Print full structure to identify the problem
      console.log("==== COMPREHENSIVE DEBUG OUTPUT ====");
      console.log("Forecast type:", result.forecast_type);
      console.log("Selected models:", selectedModel);
      console.log("FULL future_forecasts:", result.future_forecasts);
      console.log("future_forecasts type:", typeof result.future_forecasts);
      console.log("Is future_forecasts array?", Array.isArray(result.future_forecasts));
      console.log("future_forecasts stringified:", JSON.stringify(result.future_forecasts, null, 2));
      
      // Deep dive into the data structure
      if (result.forecast_type === "Item-wise") {
        console.log("Item-wise forecast detected!");
        
        // What are the top-level keys? These should be product IDs
        const topLevelKeys = Object.keys(result.future_forecasts);
        console.log("Top level keys (should be product IDs):", topLevelKeys);
        
        // Check if topLevelKeys contains model names - improved detection
        const containsModelName = topLevelKeys.some(key => 
          selectedModel === key || 
          ['SES', 'ARIMA', 'SARIMA', 'Prophet', 'HWES', 'Random Forest', 'LSTM', 'XGBoost', 'Croston', 'GRU'].includes(key)
        );
        
        if (containsModelName) {
          console.log("WARNING: Top level keys are model names, not product IDs!");
          
          // Search for any data that might have product IDs
          let uniqueProductIds = [];
          
          // Check original data if available
          if (result.original_data) {
            console.log("Extracting product IDs from original data");
            uniqueProductIds = [...new Set(result.original_data.map(row => row.ProductID))];
          }
          
          // Fallback: Try to extract product ID from the CSV file name or metadata
          if (uniqueProductIds.length === 0 && selectedDataset) {
            console.log("Trying to extract product ID hint from dataset name:", selectedDataset);
            // This is a fallback mechanism only
          }
          
          // If we still don't have IDs, use a meaningful label
          if (uniqueProductIds.length === 0) {
            console.log("No product IDs could be extracted, using 'aggregated'");
            uniqueProductIds = ["aggregated"];
          }
          
          // Restructure the forecasts
          const restructuredForecasts = {};
          
          // Use the actual data we received but restructure it properly
          uniqueProductIds.forEach(productId => {
            restructuredForecasts[productId] = {};
            
            // For each model, add its forecasts under this product ID
            for (const modelName of topLevelKeys) {
              restructuredForecasts[productId][modelName] = result.future_forecasts[modelName];
            }
          });
          
          console.log("Restructured forecasts:", restructuredForecasts);
          result.future_forecasts = restructuredForecasts;
        }
        
        // Let's check one of the items
        if (topLevelKeys.length > 0) {
          const firstKey = topLevelKeys[0];
          const firstItem = result.future_forecasts[firstKey];
          console.log(`First item (${firstKey}):`, firstItem);
          
          // Is this a nested structure?
          if (typeof firstItem === 'object' && !Array.isArray(firstItem)) {
            console.log(`Item ${firstKey} contains nested model data:`, Object.keys(firstItem));
          } else {
            console.log(`Item ${firstKey} contains direct forecast array, length:`, Array.isArray(firstItem) ? firstItem.length : "Not an array");
          }
        }
      } else if (result.forecast_type === "Store-Item Combination") {
        // Similar check for Store-Item
        console.log("Store-Item Combination forecast detected!");
        
        const topLevelKeys = Object.keys(result.future_forecasts);
        console.log("Top level keys (should be store-item combos):", topLevelKeys);
        
        // Check if topLevelKeys look like store-item combos or if they are model names
        const hasProperComboFormat = topLevelKeys.some(key => key.includes(' - '));
        const containsModelName = topLevelKeys.some(key => 
          selectedModel === key || 
          ['SES', 'ARIMA', 'SARIMA', 'Prophet', 'HWES', 'Random Forest', 'LSTM', 'XGBoost', 'Croston', 'GRU'].includes(key)
        );
        
        if (!hasProperComboFormat && containsModelName) {
          console.log("WARNING: Top level keys are model names, not store-item combos!");
          
          // Use a single representative combo ID
          const restructuredForecasts = {};
          restructuredForecasts["aggregated_data"] = {};
          
          // For each model, add its forecasts under the combo
          for (const modelName of topLevelKeys) {
            restructuredForecasts["aggregated_data"][modelName] = result.future_forecasts[modelName];
          }
          
          console.log("Restructured forecasts:", restructuredForecasts);
          result.future_forecasts = restructuredForecasts;
        }
      }
      
      // Include column mappings in the forecast results
      result.columnMappings = columnMappings;
      
      // If we got this far, the request was successful
      setIsLoading(false);
      
      if (result.status === "success") {
        // Store forecast results in state or redirect to results page
        setForecastResults(result);
        setShowForecastResults(true);
        showAlert(`Model ${selectedModel} execution completed successfully!`);
      } else {
        showAlert(`Warning: ${result.message || "Unknown status returned from API"}`);
      }
    } catch (fetchError) {
      console.error("API request error:", fetchError);
      showAlert(`Error communicating with the server: ${fetchError.message}`, "error");
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const requiredFilled = forecastHorizon && forecastLock && selectedDataset;
    // Only check selectedModel if forecastMethod is "Select From List"
    const modelRequired = forecastMethod === "Select From List" ? selectedModel : true;
    setCanRunModel(requiredFilled && modelRequired);
  }, [selectedModel, forecastHorizon, forecastLock, selectedDataset, forecastMethod]);

  const handleDownloadForecast = async () => {
    try {
      if (!forecastResults) {
        showAlert("No forecast results available to download.", "error");
        return;
      }
      
      const exportData = {
        results: forecastResults.results,
        future_forecasts: forecastResults.future_forecasts,
        dates: forecastResults.dates,
        forecast_type: forecastResults.forecast_type || map_granularity_to_forecast_type(granularity),
        future_dates: forecastResults.future_dates
      };
      
      // If CSV data is already in the response, use that directly
      if (forecastResults.csv_data) {
        downloadCSV(forecastResults.csv_data);
        return;
      }
      const token = getAuthToken();
      if (!token) {
        console.error("No auth token found, stopping fetch.");
        return;
      }
      
      // Otherwise call the export endpoint
      const response = await fetch(`${BASE_URL}/export-forecast`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify(exportData)
      });
      
      if (!response.ok) {
        throw new Error(`Export failed with status ${response.status}`);
      }
      
      // Check if the response is JSON (error) or CSV (success)
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        // Handle error response
        const errorData = await response.json();
        throw new Error(errorData.message || "Export failed");
      }
      
      // Success - download the CSV
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "forecast_results.csv";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error downloading forecast:", error);
      showAlert(`Error downloading forecast: ${error.message}`, "error");
    }
  };
  
  // Helper function to map granularity to forecast_type
  const map_granularity_to_forecast_type = (granularity) => {
    const mapping = {
      "Overall": "Overall",
      "ProductID-wise": "Item-wise",
      "StoreID-ProductID Combination": "Store-Item Combination"
    };
    return mapping[granularity] || "Overall";
  };
  
  // Helper function to download CSV data directly
  const downloadCSV = (csvData) => {
    try {
      // Convert the data to CSV format
      let csvContent = "data:text/csv;charset=utf-8,";
      
      // Add headers
      if (csvData.length > 0) {
        const headers = Object.keys(csvData[0]);
        csvContent += headers.join(",") + "\n";
        
        // Add rows
        csvData.forEach(row => {
          const rowContent = headers.map(header => {
            // Handle values with commas
            const value = row[header] !== null && row[header] !== undefined ? row[header].toString() : "";
            return value.includes(",") ? `"${value}"` : value;
          }).join(",");
          csvContent += rowContent + "\n";
        });
      }
      
      // Create download link
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "forecast_results.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error("Error creating CSV:", error);
      showAlert(`Error creating CSV: ${error.message}`, "error");
    }
  };

  const handleForecastHorizonChange = (e) => {
    const value = e.target.value;
    setForecastHorizon(value);
    // If forecast lock is greater than new horizon, reset it
    if (parseInt(forecastLock) > parseInt(value)) {
      setForecastLock("0");
    }
  };

  const handleForecastLockChange = (e) => {
    const value = e.target.value;
    const horizon = parseInt(forecastHorizon);
    const lock = parseInt(value);
    
    if (lock >= horizon) {
      showAlert("Forecast lock period must be less than forecast horizon", "error");
      return;
    }
    
    setForecastLock(value);
  };

  // Add organization calendar state
  const [organizationId, setOrganizationId] = useState('default');
  const [weekStartDay, setWeekStartDay] = useState(0); // 0=Monday
  const [organizationCalendars, setOrganizationCalendars] = useState([]);
  const [showCalendarConfig, setShowCalendarConfig] = useState(false);

  // Week day options for dropdown
  const weekDayOptions = [
    { value: 0, label: 'Monday' },
    { value: 1, label: 'Tuesday' },
    { value: 2, label: 'Wednesday' },
    { value: 3, label: 'Thursday' },
    { value: 4, label: 'Friday' },
    { value: 5, label: 'Saturday' },
    { value: 6, label: 'Sunday' }
  ];

  // Load organization calendars on component mount
  useEffect(() => {
    fetchOrganizationCalendars();
  }, []);

  const fetchOrganizationCalendars = async () => {
    try {
      const response = await fetch('/api/engine/organization/calendars', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setOrganizationCalendars(data.calendars || []);
      }
    } catch (error) {
      console.error('Error fetching organization calendars:', error);
    }
  };

  const saveOrganizationCalendar = async () => {
    try {
      const response = await fetch('/api/engine/organization/calendar', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          organization_id: organizationId,
          week_start_day: weekStartDay
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Calendar configuration saved: Week starts on ${result.week_start_name} for organization ${organizationId}`);
        fetchOrganizationCalendars(); // Refresh the list
        setShowCalendarConfig(false);
      } else {
        const error = await response.json();
        alert(`Error: ${error.message}`);
      }
    } catch (error) {
      console.error('Error saving organization calendar:', error);
      alert('Error saving calendar configuration');
    }
  };

  return (
    <div className="forecast-container">
      <style>{customStyles}</style>
      <h3 className="forecast-title">Forecast Settings</h3>

      {/* Alert Component */}
      {alert.isOpen && (
        <div className={`alert alert-${alert.type}`} style={{
          padding: "10px 15px",
          margin: "10px 0",
          borderRadius: "4px",
          backgroundColor: alert.type === "error" ? "#f8d7da" : 
                          alert.type === "success" ? "#d4edda" : 
                          alert.type === "warning" ? "#fff3cd" : "#cce5ff",
          color: alert.type === "error" ? "#721c24" : 
                alert.type === "success" ? "#155724" : 
                alert.type === "warning" ? "#856404" : "#004085",
          border: `1px solid ${alert.type === "error" ? "#f5c6cb" : 
                              alert.type === "success" ? "#c3e6cb" : 
                              alert.type === "warning" ? "#ffeeba" : "#b8daff"}`,
          position: "relative",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}>
          <span>{alert.message}</span>
          <button 
            onClick={() => setAlert({...alert, isOpen: false})} 
            style={{
              background: "none",
              border: "none",
              fontSize: "16px",
              cursor: "pointer",
              color: "inherit"
            }}
          >
            ×
          </button>
        </div>
      )}

      <div className="forecast-methodology-section" style={{ marginBottom: "20px", marginTop: "20px" }}>
        <h4 style={{ fontSize: "18px", marginBottom: "10px" }}><strong>Forecast Methodology :</strong></h4>
        <div className="dropdown-container">
          <select 
            value={forecastMethod} 
            onChange={(e) => {
              setForecastMethod(e.target.value);
              if (e.target.value === "Best Fit") {
                setSelectedModel(""); // Clear selected model when Best Fit is chosen
              }
            }} 
            className="centered-dropdown"
            style={{ minWidth: "200px" }}
          >
            <option value="Best Fit">Best Fit</option>
            <option value="Select From List">Select From List</option>
          </select>
        </div>
      </div>

      {/* Dataset Selection Dropdown */}
      <div className="forecast-details" style={{ marginBottom: "20px" }}>
        <label style={{ fontSize: "18px" }}><strong>Available Datasets :</strong></label>
        <select
          value={selectedDataset}
          onChange={handleDatasetChange}
          className="input-field"
          disabled={uploadedDatasets.length === 0}
        >
          <option value="">Select a Dataset</option>
          {uploadedDatasets.map((dataset, index) => (
            <option key={index} value={dataset}>{dataset}</option>
          ))}
        </select>
        {uploadedDatasets.length === 0 && (
          <p style={{ color: "#DC3545", fontSize: "14px", marginTop: "5px" }}>
            No datasets uploaded. Please upload data in the Import section.
          </p>
        )}
      </div>

      {/* Column Selection and Mapping */}
      {showColumns && Array.isArray(columnNames) && columnNames.length > 0 && (
        <div ref={columnSelectionRef} className="column-selection">
          <h3>Select and Map Columns</h3>

          <div className="available-columns">
            <h4>Available Columns:</h4>
            <div className="columns-container">
              {columnNames.map((col, index) => (
                <div key={index} className="column-item">
                  {col}
                  <button className="delete-btn" onClick={() => handleDeleteColumn(col)}>
                    ✖
                  </button>
                </div>
              ))}
            </div>
          </div>

          {omittedColumns.length > 0 && (
            <div className="omitted-columns">
              <h4>Omitted Columns:</h4>
              <div className="columns-container">
                {omittedColumns.map((col, index) => (
                  <div key={index} className="column-item">
                    {col.name}
                    <button className="retrieve-btn" onClick={() => handleRetrieveColumn(col.name)}>
                      ↺
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mapping-section">
            <h4>Map Mandatory Fields:</h4>
            <div className="mapping-container">
              {mandatoryFields.map((field) => (
                <div key={field} className="mapping-item">
                  <label>{field}:</label>
                  <select
                    className="mapping-dropdown"
                    onChange={(e) => handleMappingChange(field, e.target.value)}
                    value={columnMappings[field] || ""}
                  >
                    <option value="">Select column...</option>
                    {Array.isArray(columnNames) && columnNames.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </div>

          <div className="time-dependent-variables">
            <h4>Select Additional Time-Dependent Variables:</h4>
            <div className="checkbox-container">
              {Array.isArray(columnNames) && columnNames.map((col) => (
                <label key={col} className="checkbox-label">
                  <input
                    type="checkbox"
                    value={col}
                    checked={timeDependentVariables.includes(col)}
                    onChange={handleTimeDependentVariablesChange}
                  />
                  {col}
                </label>
              ))}
            </div>
            <button className="clear-btn" onClick={handleClearTimeDependentVariables}>
              Clear Selection
            </button>
            <button
              className={`preview-btn ${!areMandatoryFieldsMapped() ? "disabled" : ""}`}
              onClick={handleShowPreview}
              style={{ marginLeft: "20px" }}
            >
              Show Preview
            </button>
          </div>
        </div>
      )}

      {/* Preview Table */}
      {showTable && Array.isArray(importedData) && importedData.length > 0 && Array.isArray(columnNames) && columnNames.length > 0 && (
        <div className="imported-data-table">
          <h3>Preview First 5 Rows</h3>
          <table>
            <thead>
              <tr>
                {columnNames.map((col, index) => (
                  <th key={index} scope="col">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {importedData.slice(0, 5).map((row, index) => (
                <tr key={index}>
                  {columnNames.map((col, colIndex) => (
                    <td key={colIndex}>
                      <input
                        type="text"
                        value={row[col] || ""}
                        onChange={(e) => handleFieldChange(index, col, e.target.value)}
                        aria-label={`View ${col} for row ${index + 1}`}
                        readOnly={true} // Make fields non-editable after preview
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <button
            ref={uploadButtonRef}
            className="upload-data-for-cleaning"
            onClick={handleUploadDataForCleaning}
          >
            Upload Data for Cleaning
          </button>
        </div>
      )}

      {/* Granularity Dropdown (Moved Here) */}
      {showTable && (
        <div className="granularity-section" style={{ marginTop: "20px" }}>
          <h4 style={{ fontSize: "18px" }}><strong>Granularity :</strong></h4>
          <div className="dropdown-container">
            <select
              value={granularity}
              onChange={(e) => setGranularity(e.target.value)}
              className="centered-dropdown"
            >
              {granularityOptions.map((option, index) => (
                <option key={index} value={option}>{option}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Model Selection - Only show if "Select From List" is chosen */}
      {forecastMethod === "Select From List" && (
        <div className="forecast-details">
          <label style={{ fontSize: "18px" }}><strong>Model Name :</strong></label>
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isLoadingModels}
          >
            <option value="">Select a Model</option>
            {isLoadingModels ? (
              <option value="" disabled>Loading models...</option>
            ) : (
              models.map((model, index) => (
                <option key={index} value={model}>{model}</option>
              ))
            )}
          </select>
          {isLoadingModels && <span style={{ marginLeft: '10px', fontSize: '14px', color: '#666' }}>Loading...</span>}
        </div>
      )}

      <div className="top-buttons">
        <button className="reset-button" onClick={resetForm}>Reset</button>
      </div>

      <div className="time-settings">
        <h4 style={{ fontSize: "18px" }}><strong>Time Settings :</strong></h4>
        <div className="settings-card-group" style={{ display: "flex", gap: "10px" }}>
          <div className="settings-card" style={{ flex: "0.9" }}>
            <h5 style={{ fontSize: "16px" }}>Time Bucket</h5>
            <div className="input-group">
              <select style={{ width: "120px" }} value={timeBucket} onChange={(e) => setTimeBucket(e.target.value)}>
                <option value="Daily">Daily</option>
                <option value="Weekly">Weekly</option>
                <option value="Monthly">Monthly</option>
                <option value="Yearly">Yearly</option>
              </select>
            </div>
          </div>

          <div className="settings-card">
            <h5 style={{ fontSize: "16px" }}>Forecast Parameters</h5>
            <div className="input-group">
              <label>Forecast Horizon:</label>
              <input 
                type="number" 
                value={forecastHorizon} 
                onChange={handleForecastHorizonChange}
                min="0"
              />
            </div>
            <div className="input-group">
              <label>Forecast Lock Period:</label>
              <input 
                type="number" 
                value={forecastLock} 
                onChange={handleForecastLockChange}
                min="0"
                max={parseInt(forecastHorizon) - 1}
              />
            </div>
          </div>
          
          <div className="settings-card">
            <h5 style={{ fontSize: "16px" }}>Organization Calendar</h5>
            <div className="input-group">
              <label>Organization:</label>
              <input 
                type="text" 
                value={organizationId} 
                onChange={(e) => setOrganizationId(e.target.value)}
                placeholder="Organization ID"
                style={{ width: "120px" }}
              />
            </div>
            <div className="input-group">
              <button 
                onClick={() => setShowCalendarConfig(!showCalendarConfig)}
                style={{ 
                  padding: "5px 10px", 
                  backgroundColor: "#007bff", 
                  color: "white", 
                  border: "none", 
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontSize: "12px"
                }}
              >
                {showCalendarConfig ? "Hide" : "Configure"} Calendar
              </button>
            </div>
          </div>
        </div>
        
        {/* Organization Calendar Configuration Panel */}
        {showCalendarConfig && (
          <div className="calendar-config-panel" style={{ 
            marginTop: "15px", 
            padding: "15px", 
            border: "1px solid #ddd", 
            borderRadius: "5px",
            backgroundColor: "#f9f9f9"
          }}>
            <h5 style={{ fontSize: "16px", marginBottom: "10px" }}>
              <strong>Calendar Configuration for {organizationId}</strong>
            </h5>
            
            <div style={{ display: "flex", gap: "15px", alignItems: "center" }}>
              <div className="input-group">
                <label>Week starts on:</label>
                <select 
                  value={weekStartDay} 
                  onChange={(e) => setWeekStartDay(parseInt(e.target.value))}
                  style={{ width: "120px" }}
                >
                  {weekDayOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
              
              <button 
                onClick={saveOrganizationCalendar}
                style={{ 
                  padding: "8px 15px", 
                  backgroundColor: "#28a745", 
                  color: "white", 
                  border: "none", 
                  borderRadius: "4px",
                  cursor: "pointer"
                }}
              >
                Save Calendar
              </button>
            </div>
            
            {/* Display existing organization calendars */}
            {organizationCalendars.length > 0 && (
              <div style={{ marginTop: "15px" }}>
                <h6 style={{ fontSize: "14px", marginBottom: "8px" }}>
                  <strong>Existing Organization Calendars:</strong>
                </h6>
                <div style={{ maxHeight: "100px", overflowY: "auto" }}>
                  {organizationCalendars.map((calendar, index) => (
                    <div key={index} style={{ 
                      fontSize: "12px", 
                      padding: "4px 8px", 
                      margin: "2px 0",
                      backgroundColor: "white",
                      border: "1px solid #eee",
                      borderRadius: "3px"
                    }}>
                      <strong>{calendar.organization_id}:</strong> Week starts on {calendar.week_start_name}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="forecast-parameters active">
        <h4 style={{ fontSize: "18px" }}><strong>Forecast Parameters :</strong></h4>
        <div className="forecast-methodology">
          <div className="dropdown-container">
            <label style={{ fontSize: "16px" }}><strong>Aggregation Level: </strong></label>
            <select value={aggregation} onChange={(e) => setAggregation(e.target.value)} className="centered-dropdown">
              <option value="Sum">Sum</option>
              <option value="Average">Average</option>
              <option value="Max">Max</option>
            </select>
          </div>
        </div>
      </div>

      {/* Run Model Button - Show for both Best Fit and Select From List */}
      {((forecastMethod === "Best Fit") || (forecastMethod === "Select From List" && selectedModel)) && selectedDataset && (
        <div className="run-model-card">
          <h4>Model Execution</h4>
          <button
            className={`run-button ${canRunModel ? "active" : "disabled"}`}
            onClick={handleRunModel}
            disabled={!canRunModel}
          >
            Run Forecasting Model
          </button>
          {!canRunModel && <p className="validation-message">Please fill all required parameters</p>}
        </div>
      )}

      {/* Add loading indicator */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Processing data... Please wait</p>
        </div>
      )}
      
      {/* Add data summary section */}
      {showSummary && dataSummary && (
        <div className="data-summary">
          <h3>Data Processing Summary</h3>
          <button className="close-btn" onClick={() => setShowSummary(false)}>×</button>
          <div className="summary-content">
            {Object.entries(dataSummary).map(([key, value]) => (
              <div key={key} className="summary-item">
                <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Forecast Results Section */}
      {showForecastResults && forecastResults && (
        <div className="forecast-results-section">
          <div className="forecast-results-header">
            <h3 className="forecast-results-title">Forecast Results</h3>
            <button
              className="download-button"
              onClick={handleDownloadForecast}
              disabled={!forecastResults}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 16l-4-4h3V4h2v8h3l-4 4zm9-13h-6v2h4.5c.83 0 1.5.67 1.5 1.5v13c0 .83-.67 1.5-1.5 1.5h-13c-.83 0-1.5-.67-1.5-1.5v-13c0-.83.67-1.5 1.5-1.5H11V3H3v18h18V3z"/>
              </svg>
              Download CSV
            </button>
          </div>
          <div className="results-content">
            <h4>Model Performance</h4>
            {Object.entries(forecastResults.results).map(([modelName, metrics]) => (
              <div key={modelName} className="model-results">
                <h5>{modelName}</h5>
                {typeof metrics === 'string' ? (
                  <p className="error-message">{metrics}</p>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Training</th>
                        <th>Validation</th>
                        <th>Test</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>RMSE</td>
                        <td>{metrics.train && metrics.train[0] ? metrics.train[0].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.val && metrics.val[0] ? metrics.val[0].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.test && metrics.test[0] ? metrics.test[0].toFixed(2) : 'N/A'}</td>
                      </tr>
                      <tr>
                        <td>MAPE</td>
                        <td>{metrics.train && metrics.train[1] ? metrics.train[1].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.val && metrics.val[1] ? metrics.val[1].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.test && metrics.test[1] ? metrics.test[1].toFixed(2) : 'N/A'}</td>
                      </tr>
                      <tr>
                        <td>MAE</td>
                        <td>{metrics.train && metrics.train[2] ? metrics.train[2].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.val && metrics.val[2] ? metrics.val[2].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.test && metrics.test[2] ? metrics.test[2].toFixed(2) : 'N/A'}</td>
                      </tr>
                      <tr>
                        <td>Bias</td>
                        <td>{metrics.train && metrics.train[3] ? metrics.train[3].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.val && metrics.val[3] ? metrics.val[3].toFixed(2) : 'N/A'}</td>
                        <td>{metrics.test && metrics.test[3] ? metrics.test[3].toFixed(2) : 'N/A'}</td>
                      </tr>
                    </tbody>
                  </table>
                )}
              </div>
            ))}
            
            <h4>Future Forecasts</h4>
            <div className="future-forecasts">
              {forecastResults.forecast_type === "Overall" ? (
                // Overall forecast type display
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    {Object.keys(forecastResults.future_forecasts || {}).map(modelName => (
                      <th key={modelName}>{modelName}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                    {Array.from({ length: parseInt(forecastHorizon) || 30 }).map((_, index) => {
                      // Detect frequency and generate future dates from current date
                      const dataFrequency = detectDataFrequency(forecastResults.dates);
                      const futureDates = generateFutureDatesFromCurrent(parseInt(forecastHorizon) || 30, dataFrequency);
                      const futureDate = futureDates[index];
                      
                      return (
                    <tr key={index}>
                          <td>{futureDate ? futureDate.toISOString().split('T')[0] : `Future_${index + 1}`}</td>
                          {Object.entries(forecastResults.future_forecasts || {}).map(([modelName, forecasts]) => (
                            <td key={modelName}>
                              {Array.isArray(forecasts) && index < forecasts.length ? 
                                typeof forecasts[index] === 'number' ? 
                                  forecasts[index].toFixed(2) : forecasts[index]
                                : 'N/A'}
                            </td>
                      ))}
                    </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : forecastResults.forecast_type === "Item-wise" ? (
                // Item-wise forecast type display
                <div className="item-wise-forecast">
                  <div className="item-selector">
                    <label>Select Product:</label>
                    <select onChange={(e) => {
                      const itemId = e.target.value;
                      document.getElementById(`item-forecast-${itemId}`)?.scrollIntoView({ behavior: 'smooth' });
                    }}>
                      {(() => {
                        // Get all product IDs (top-level keys)
                        const productIds = Object.keys(forecastResults.future_forecasts || {});
                        
                        // Log for debugging
                        console.log("Product IDs for dropdown:", productIds);
                        
                        // Map them to options, with nice display format
                        return productIds.map(productId => {
                          // Format the display value for special cases
                          let displayValue = productId;
                          
                          if (productId === "default_product") {
                            displayValue = "All Products";
                          } else if (productId === "all_items") {
                            displayValue = "All Items";
                          } else if (productId === "aggregated" || productId === "aggregated_data") {
                            displayValue = "Aggregated Data";
                          }
                          
                          return (
                            <option key={productId} value={productId}>{displayValue}</option>
                          );
                        });
                      })()}
                    </select>
                  </div>
                  
                  {(() => {
                    // Get all entries from future_forecasts
                    const entries = Object.entries(forecastResults.future_forecasts || {});
                    
                    // Map each entry to a forecast display
                    return entries.map(([productId, modelForecasts]) => {
                      console.log(`Rendering item ${productId} with forecasts:`, modelForecasts);
                      
                      // Check data structure - is it nested or flat?
                      const isNestedStructure = typeof modelForecasts === 'object' && !Array.isArray(modelForecasts);
                      
                      // Get model names based on structure
                      const modelNames = isNestedStructure 
                        ? Object.keys(modelForecasts) 
                        : [selectedModel || "Forecast"];
                      
                      console.log(`Item ${productId} modelNames:`, modelNames);
                      
                      return (
                        <div key={productId} id={`item-forecast-${productId}`} className="item-forecast">
                          <h5>Product: {
                            productId === "default_product" ? "All Products" : 
                            productId === "all_items" ? "All Items" : 
                            productId === "aggregated" ? "Aggregated Data" : 
                            productId
                          }</h5>
                          <table>
                            <thead>
                              <tr>
                                <th>Date</th>
                                {modelNames.map(modelName => (
                                  <th key={modelName}>{modelName}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {Array.from({ length: parseInt(forecastHorizon) || 30 }).map((_, index) => {
                                // Detect frequency and generate future dates from current date
                                const dataFrequency = detectDataFrequency(forecastResults.dates);
                                const futureDates = generateFutureDatesFromCurrent(parseInt(forecastHorizon) || 30, dataFrequency);
                                const futureDate = futureDates[index];
                                
                                return (
                                  <tr key={index}>
                                    <td>{futureDate ? futureDate.toISOString().split('T')[0] : `Future_${index + 1}`}</td>
                                    {modelNames.map(modelName => {
                                      // Get forecasts based on structure
                                      const forecasts = isNestedStructure
                                        ? modelForecasts[modelName]
                                        : modelForecasts;
                                      
                                      return (
                                        <td key={modelName}>
                                          {Array.isArray(forecasts) && index < forecasts.length
                                            ? typeof forecasts[index] === 'number'
                                              ? forecasts[index].toFixed(2)
                                              : forecasts[index]
                                            : 'N/A'}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                );
                              })}
                </tbody>
              </table>
            </div>
                      );
                    });
                  })()}
                </div>
              ) : forecastResults.forecast_type === "Store-Item Combination" ? (
                // Store-Item Combination forecast type display
                <div className="store-item-forecast">
                  <div className="combination-selector">
                    <label>Select Store-Item Combination:</label>
                    <select onChange={(e) => {
                      const comboId = e.target.value;
                      document.getElementById(`combo-forecast-${comboId}`)?.scrollIntoView({ behavior: 'smooth' });
                    }}>
                      {(() => {
                        // Get all combo IDs (top-level keys)
                        const comboIds = Object.keys(forecastResults.future_forecasts || {});
                        
                        // Log for debugging
                        console.log("Combo IDs for dropdown:", comboIds);
                        
                        // Map them to options
                        return comboIds.map(comboId => {
                          // Display the store-item combination more nicely
                          let displayName = comboId;
                          
                          if (comboId === "default_store - default_product") {
                            displayName = "All Store-Item Combinations";
                          } else if (comboId === "all_combinations") {
                            displayName = "All Combinations";
                          } else if (comboId === "aggregated_data") {
                            displayName = "Aggregated Data";
                          } else {
                            const parts = comboId.split(' - ');
                            if (parts.length === 2) {
                              displayName = `Store: ${parts[0]}, Item: ${parts[1]}`;
                            }
                          }
                          
                          return (
                            <option key={comboId} value={comboId}>{displayName}</option>
                          );
                        });
                      })()}
                    </select>
                  </div>
                  
                  {(() => {
                    // Get all entries from future_forecasts
                    const entries = Object.entries(forecastResults.future_forecasts || {});
                    
                    // Map each entry to a forecast display
                    return entries.map(([comboId, modelForecasts]) => {
                      console.log(`Rendering combination ${comboId} with forecasts:`, modelForecasts);
                      
                      // Check for special combo names
                      let displayTitle = comboId;
                      
                      if (comboId === "default_store - default_product") {
                        displayTitle = "All Store-Item Combinations";
                      } else if (comboId === "all_combinations") {
                        displayTitle = "All Combinations";
                      } else if (comboId === "aggregated_data") {
                        displayTitle = "Aggregated Data";
                      } else {
                        const parts = comboId.split(' - ');
                        if (parts.length === 2) {
                          const storeId = parts[0];
                          const itemId = parts[1];
                          displayTitle = `Store: ${storeId}, Item: ${itemId}`;
                        }
                      }
                      
                      // Check data structure - is it nested or flat?
                      const isNestedStructure = typeof modelForecasts === 'object' && !Array.isArray(modelForecasts);
                      
                      // Get model names based on structure
                      const modelNames = isNestedStructure 
                        ? Object.keys(modelForecasts) 
                        : [selectedModel || "Forecast"];
                      
                      console.log(`Combo ${comboId} modelNames:`, modelNames);
                      
                      return (
                        <div key={comboId} id={`combo-forecast-${comboId}`} className="combo-forecast">
                          <h5>{displayTitle}</h5>
                          <table>
                            <thead>
                              <tr>
                                <th>Date</th>
                                {modelNames.map(modelName => (
                                  <th key={modelName}>{modelName}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {Array.from({ length: parseInt(forecastHorizon) || 30 }).map((_, index) => {
                                // Detect frequency and generate future dates from current date
                                const dataFrequency = detectDataFrequency(forecastResults.dates);
                                const futureDates = generateFutureDatesFromCurrent(parseInt(forecastHorizon) || 30, dataFrequency);
                                const futureDate = futureDates[index];
                                
                                return (
                                  <tr key={index}>
                                    <td>{futureDate ? futureDate.toISOString().split('T')[0] : `Future_${index + 1}`}</td>
                                    {modelNames.map(modelName => {
                                      // Get forecasts based on structure
                                      const forecasts = isNestedStructure
                                        ? modelForecasts[modelName]
                                        : modelForecasts;
                                      
                                      return (
                                        <td key={modelName}>
                                          {Array.isArray(forecasts) && index < forecasts.length
                                            ? typeof forecasts[index] === 'number'
                                              ? forecasts[index].toFixed(2)
                                              : forecasts[index]
                                            : 'N/A'}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      );
                    });
                  })()}
                </div>
              ) : (
                // Fallback display if forecast type is unknown
                <div className="unknown-forecast-type">
                  <p>Unknown forecast type: {forecastResults.forecast_type || "Not specified"}</p>
                  <p>Raw data:</p>
                  <pre>{JSON.stringify(forecastResults.future_forecasts, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ForecastSettings;