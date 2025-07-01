import React, { useState, useEffect } from "react";
import moment from 'moment';
import { useAuth } from '../AuthContext';
import { PLANNERWB_ENDPOINT } from "./config"; 
import Cookies from "js-cookie";
import PageTitle from './PageTitle';

function aggregateTableData(tableData, timeBucket) {
  if (!tableData) return [];

  let groupFormat = "YYYY-MM-DD";
  if (timeBucket === "Monthly") groupFormat = "YYYY-MM";
  else if (timeBucket === "Yearly") groupFormat = "YYYY";

  const agg = {};

  tableData.forEach(row => {
    const groupKey = moment(row.date).format(groupFormat);
    const { store, product } = row;
    const key = `${store}||${product}||${groupKey}`;

    if (!agg[key]) {
      agg[key] = {
        store,
        product,
        date: groupKey,
        actual_quantity: 0,
        planner_entered_actuals: 0,
        forecast_quantity:  0,
        MANUALDEMAND: 0,
        final_qty: 0,
        count: 0
      };
    }

    agg[key].actual_quantity += Number(row.actual_quantity) || 0;
    agg[key].planner_entered_actuals += Number(row.planner_entered_actuals) || 0;
    agg[key].forecast_quantity += Number(row.forecast_quantity) || 0;
    agg[key].MANUALDEMAND += Number(row.MANUALDEMAND) || 0;
    agg[key].count += 1;
  });

  Object.values(agg).forEach(group => {
    group.final_qty = group.MANUALDEMAND;
  });

  return Object.values(agg);
}

const PlannerWorkbench = () => {
  const { user } = useAuth();
  const [selectedProduct, setSelectedProduct] = useState("");
  const [selectedLocation, setSelectedLocation] = useState("");
  const [timeBucket, setTimeBucket] = useState("Monthly");
  const [startDate, setStartDate] = useState(moment().format("YYYY-MM-DD"));
  const [endDate, setEndDate] = useState(moment().add(3, 'months').format("YYYY-MM-DD"));
  const [tableData, setTableData] = useState(null);
  const [editedCells, setEditedCells] = useState({});
  const [isEditing, setIsEditing] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveReason, setSaveReason] = useState("");
  const [saveComment, setSaveComment] = useState("");
  const token = Cookies.get("authToken");
  
  useEffect(() => {
    console.log('Auth user:', user);
  }, [user]);

  function groupData(data) {
    const grouped = {};
    data.forEach(row => {
      const { store, product, date } = row;
      if (!grouped[store]) grouped[store] = {};
      if (!grouped[store][product]) grouped[store][product] = {};
      grouped[store][product][date] = row;
    });
    return grouped;
  }

  const [products, setProducts] = useState([]);
  const [stores, setStores] = useState([]);

  const measures = [
    { key: "actual_quantity", label: "Actual Quantity" },
    { key: "planner_entered_actuals", label: "Planner-Entered Actuals" }, // The NEW row
    { key: "forecast_quantity", label: "Forecast Quantity" },
    { key: "MANUALDEMAND", label: "Manual Forecast" },
    { key: "final_qty", label: "Demand Planning Final Qty" }
  ];

  useEffect(() => {
    setTableData(null);
    fetch(`${PLANNERWB_ENDPOINT}`, {
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    })
    .then(res => {
      if (!res.ok) throw new Error("Failed to fetch planner data");
      return res.json();
    })
    .then(json => {
      const normalizedData = (json.data || []).map(row => ({
        ...row,
        MANUALDEMAND: row.MANUALDEMAND ?? row.manual_forecast ?? undefined
      }));
      setTableData(normalizedData);

      if (normalizedData.length) {
        const products = [...new Set(normalizedData.map(row => row.product))].sort();
        const stores = [...new Set(normalizedData.map(row => row.store))].sort();
        setProducts(products);
        setStores(stores);
      }
    })
    .catch(err => {
      setTableData([]);
      setProducts([]);
      setStores([]);
      console.error(err);
    });
  }, [token]);

  const filteredTableDataRaw = tableData
    ? tableData.filter(row =>
        (!selectedProduct || row.product === selectedProduct) &&
        (!selectedLocation || row.store === selectedLocation)
      )
    : [];

  const filteredTableData = aggregateTableData(filteredTableDataRaw, timeBucket);
  const groupedTableData = filteredTableData ? groupData(filteredTableData) : {};
  const allDates = filteredTableData
    ? [...new Set(filteredTableData.map(r => r.date))].sort()
    : [];

  const handleCellEdit = (store, product, date, metric, value) => {
    // Only allow edits for these two specific rows
    if (metric !== "MANUALDEMAND" && metric !== "planner_entered_actuals") return;

    const parsedValue = value === "" ? null : parseFloat(value);
    if (value !== '' && (isNaN(parsedValue) || parsedValue < 0)) return;

    const key = `${store}||${product}||${date}`;

    setEditedCells(prev => ({
      ...prev,
      [key]: {
        ...prev[key], // Keep any other edits for the same cell
        store: store,
        product: product,
        date: date,
        [metric]: parsedValue // Use the metric key to store the edit correctly
      }
    }));
  };

  const handleSaveClick = () => {
    setShowSaveDialog(true);
  };

  const handleSaveCancel = () => {
    setShowSaveDialog(false);
    setSaveReason("");
    setSaveComment("");
  };

  const handleSaveConfirm = async () => {
    // This function will be updated later to match the new backend API payload
    try {
      const token = Cookies.get("authToken");
      const updates = Object.values(editedCells).map(edit => ({
        ...edit,
        user: user?.username || localStorage.getItem('username') || "unknown",
        reason: saveReason,
        comment: saveComment,
      }));
      const response = await fetch(`${PLANNERWB_ENDPOINT}`, {
        method: "POST",
        headers: { 
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json" 
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error("Failed to update forecast in backend");
      }

      // Refresh data from the server
      handleRefresh();

      // Clean up edit state and dialog
      setEditedCells({});
      setIsEditing(false);
      setShowSaveDialog(false);
      setSaveReason("");
      setSaveComment("");
    } catch (error) {
      alert(error.message);
    }
  };


  const handleRefresh = () => {
    // Re-trigger the initial data fetch
    fetch(`${PLANNERWB_ENDPOINT}`, { headers: { "Authorization": `Bearer ${token}` }})
      .then(res => res.json())
      .then(json => {
        const normalizedData = (json.data || []).map(row => ({
          ...row,
          MANUALDEMAND: row.MANUALDEMAND ?? row.manual_forecast ?? undefined
        }));
        setTableData(normalizedData);
        setEditedCells({});
        setIsEditing(false);
      })
      .catch(err => {
        setTableData([]);
        console.error(err);
      });
  };

  
  const renderWorksheet = () => {
    if (!tableData) return <p>Loading data...</p>;
    if (tableData.length === 0) return <p>No data available for the selected filters.</p>;

    return (
      <div className="worksheet-container" style={{ width: '100%', overflowX: 'auto', maxWidth: '100%' }}>
        <div className="worksheet">
          <h3>Demand Planning Worksheet</h3>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Entity</th>
                  <th>Measure</th>
                  {allDates.map(date => (
                    <th key={date}>
                      {timeBucket === "Daily" && moment(date).format("DD MMM YYYY")}
                      {timeBucket === "Monthly" && moment(date, "YYYY-MM").format("MMM YYYY")}
                      {timeBucket === "Yearly" && date}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(groupedTableData).map(([store, products]) => (
                  <React.Fragment key={store}>
                    {Object.entries(products).map(([product, dateRows]) =>
                      measures.map((measure, mIdx) => (
                        <tr key={`${store}-${product}-${measure.key}`}>
                          {mIdx === 0 && (
                            <td rowSpan={measures.length}><b>{store}-{product}</b></td>
                          )}
                          <td>{measure.label}</td>
                          {allDates.map(date => {
                            const row = dateRows[date] || {};
                            const cellKey = `${store}||${product}||${date}`;

                            const isEditable = isEditing &&
                                (measure.key === "MANUALDEMAND" || measure.key === "planner_entered_actuals");
                            
                            const displayValue = editedCells[cellKey]?.[measure.key] ?? row[measure.key] ?? "";

                            if (isEditable) {
                              return (
                                <td key={date} style={{ backgroundColor: '#fff9e6' }}>
                                  <input
                                    type="number"
                                    min="0"
                                    value={displayValue}
                                    onChange={e =>
                                      handleCellEdit(store, product, date, measure.key, e.target.value)
                                    }
                                    style={{
                                      width: "80px",
                                      padding: "2px 4px",
                                      textAlign: "right"
                                    }}
                                  />
                                </td>
                              );
                            }

                            return (
                              <td key={date} style={{textAlign: "right"}}>
                                {displayValue !== "" ? displayValue : "-"}
                              </td>
                            );
                          })}
                        </tr>
                      ))
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };


  return (
    <>
      <PageTitle title="Planner Workbench" />
      <div className="planner-container">
        <div className="demand-plan">
          <h3>Demand Plan</h3>

          {/* Display Settings */}
          <div className="card display-settings">
            <h3 className="card-header">Display Settings</h3>
            <div className="card-body">
              <div className="input-group">
                <label>Product</label>
                <select
                  value={selectedProduct}
                  onChange={(e) => setSelectedProduct(e.target.value)}
                >
                  <option value="">All Products</option>
                  {products.map(product => (
                    <option key={product} value={product}>{product}</option>
                  ))}
                </select>
              </div>
              <div className="input-group">
                <label>Location</label>
                <select
                  value={selectedLocation}
                  onChange={(e) => setSelectedLocation(e.target.value)}
                >
                  <option value="">All Locations</option>
                  {stores.map(store => (
                    <option key={store} value={store}>{store}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Time Horizon */}
          <div className="card time-horizon">
            <h3 className="card-header">Time Horizon</h3>
            <div className="card-body">
              <div className="input-group">
                <label>Time Bucket</label>
                <select
                  value={timeBucket}
                  onChange={(e) => setTimeBucket(e.target.value)}
                >
                  <option value="Daily">Daily</option>
                  <option value="Monthly">Monthly</option>
                  <option value="Yearly">Yearly</option>
                </select>
              </div>
              <div className="input-group">
                <label>Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="action-buttons">
            <button
              className={`edit-button ${isEditing ? "active" : ""}`}
              onClick={() => setIsEditing(!isEditing)}
            >
              {isEditing ? "Cancel Edit" : "Edit Forecast"}
            </button>
            <button
              className="save-button"
              onClick={handleSaveClick}
              disabled={!isEditing || Object.keys(editedCells).length === 0}
            >
              Save
            </button>
            <button
              className="refresh-button"
              onClick={handleRefresh}
            >
              Refresh
            </button>
          </div>

          {/* Worksheet */}
          {renderWorksheet()}

          {/* --- CHANGE 4: REMOVE THE OLD ACTUALS ENTRY WORKSHEET --- */}
          {/* The entire card div that was here has been deleted. */}
          
        </div>

        {/* Save Dialog */}
        {showSaveDialog && (
          <div className="save-dialog-overlay">
            <div className="save-dialog">
              <h3>Save Changes</h3>
              <div className="dialog-content">
                <div className="input-group">
                  <label>Reason:</label>
                  <input
                    type="text"
                    value={saveReason}
                    onChange={(e) => setSaveReason(e.target.value)}
                    placeholder="Enter reason for change"
                  />
                </div>
                <div className="input-group">
                  <label>Comment:</label>
                  <textarea
                    value={saveComment}
                    onChange={(e) => setSaveComment(e.target.value)}
                    placeholder="Enter additional comments"
                  />
                </div>
              </div>
              <div className="dialog-actions">
                <button onClick={handleSaveCancel}>Cancel</button>
                <button 
                  onClick={handleSaveConfirm}
                  disabled={!saveReason.trim()}
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

// Add the required CSS (no changes here)
const styles = `
  .save-dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  .save-dialog {
    background: white;
    padding: 20px;
    border-radius: 8px;
    width: 400px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  }
  .dialog-content { margin: 20px 0; }
  .dialog-content .input-group { margin-bottom: 15px; }
  .dialog-content label { display: block; margin-bottom: 5px; font-weight: bold; }
  .dialog-content input,
  .dialog-content textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
  .dialog-content textarea { height: 80px; resize: vertical; }
  .dialog-actions { display: flex; justify-content: flex-end; gap: 10px; }
  .dialog-actions button { padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer; }
  .dialog-actions button:first-child { background-color: #f0f0f0; }
  .dialog-actions button:last-child { background-color: #007bff; color: white; }
`;

const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default PlannerWorkbench;