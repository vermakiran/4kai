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
        forecast_quantity:  0,
        MANUALDEMAND: 0,
        final_qty: 0,
        count: 0
      };
    }

    agg[key].actual_quantity += Number(row.actual_quantity) || 0;
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
  const [overrideInfo, setOverrideInfo] = useState({});
  const token = Cookies.get("authToken");


  useEffect(() => {
    console.log('Auth user:', user);
    console.log('LocalStorage username:', localStorage.getItem('username'));
    console.log('Cookies:', document.cookie);
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
  { key: "forecast_quantity", label: "Forecast Quantity" },
  { key: "MANUALDEMAND", label: "Manual Forecast" },
  { key: "final_qty", label: "Demand Planning Final Qty" }
];


useEffect(() => {
  debugger;
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
        MANUALDEMAND: row.MANUALDEMAND !== undefined
        ? row.MANUALDEMAND
        : row.manual_forecast !== undefined
          ? row.manual_forecast
          : undefined
      }));

      console.log("Normalized planner data:", normalizedData);

      setTableData(normalizedData);
      // Derive unique product/store lists from API result
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
}, []);

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
  if (metric !== "MANUALDEMAND") return;
  const parsedValue = value === "" ? null : parseInt(value, 10);
  if (parsedValue < 0 || isNaN(parsedValue)) return;

  setEditedCells(prev => ({
    ...prev,
    [`${store}||${product}||${date}`]: {
      store: store,
      product: product,
      date: date,
      MANUALDEMAND: parsedValue
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

    const fresh = await fetch(`${PLANNERWB_ENDPOINT}`, {
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      }
    });

    const freshJson = await fresh.json();
    const normalizedData = (freshJson.data || []).map(row => ({
      ...row,
      MANUALDEMAND: row.MANUALDEMAND !== undefined
        ? row.MANUALDEMAND
        : row.manual_forecast !== undefined
          ? row.manual_forecast
          : undefined
    }));

    setTableData(normalizedData);

    // 4. Clean up edit state and dialog
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
    setTableData(null); // Optional: Show loading UI
    fetch(`${PLANNERWB_ENDPOINT}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch planner data");
        return res.json();
      })
      .then(json => {
        setTableData(json.data || []);
        setEditedCells({});
        setIsEditing(false);
      })
      .catch(err => {
        setTableData([]);
        setEditedCells({});
        setIsEditing(false);
        console.error(err);
      });
  };


  const renderCell = (entityId, metric, value, index) => {
    const key = `${entityId}-${metric}-${index}`;
    const isEditable = metric === "Manual Forecast";
    const isFinal = metric === "Demand Planning Final Qty";
    const override = overrideInfo[`${entityId}-Manual Forecast-${index}`];

    return (
      <td key={index} style={{
        width: '96px',
        minWidth: '96px',
        maxWidth: '200px',
        height: '21px',
        padding: '6px 8px',
        textAlign: 'right',
        border: '1px solid #d4d4d4',
        backgroundColor: isEditable ? '#f8f9fa' : '#fff',
        position: 'relative',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis'
      }}>
        {isEditable && isEditing ? (
          <input
            type="number"
            min="0"
            value={editedCells[key] !== undefined ? editedCells[key] : value ?? ""}
            onChange={(e) => handleCellEdit(entityId, metric, index, e.target.value)}
            style={{
              width: '100%',
              height: '20px',
              padding: '2px 4px',
              textAlign: 'right',
              border: 'none',
              backgroundColor: '#fff',
              fontSize: '14px',
              fontFamily: 'Arial, sans-serif'
            }}
          />
        ) : (
          <div 
            style={{
              fontSize: '14px',
              fontFamily: 'Arial, sans-serif',
              position: 'relative'
            }}
            className={isFinal && override ? 'override-value' : ''}
          >
            {value?.toLocaleString() ?? "0"}
            {isFinal && override && (
              <div className="override-tooltip">
                <div>Overridden by: {override.overriddenBy}</div>
                <div>Reason: {override.reason}</div>
                <div>Comment: {override.comment}</div>
              </div>
            )}
          </div>
        )}
      </td>
    );
  };

  const renderWorksheet = () => {
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
                    {Object.entries(products).map(([product, dateRows], productIdx) =>
                      measures.map((measure, mIdx) => (
                        <tr key={`${store}-${product}-${measure.key}`}>
                          {mIdx === 0 && (
                            <td rowSpan={measures.length}><b>{store}-{product}</b></td>
                          )}
                          <td>{measure.label}</td>
                          {allDates.map(date => {
                            const row = dateRows[date] || {};
                            const cellKey = `${store}||${product}||${date}`;
                            // Manual Forecast is editable
                            if (measure.key === "MANUALDEMAND") {
                              return (
                                <td key={date}>
                                  {isEditing ? (
                                    <input
                                      type="number"
                                      min="0"
                                      value={
                                        editedCells[cellKey]?.MANUALDEMAND !== undefined
                                          ? editedCells[cellKey].MANUALDEMAND
                                          : row.MANUALDEMAND ?? ""
                                      }
                                      onChange={e =>
                                        handleCellEdit(store, product, date, "MANUALDEMAND", e.target.value)
                                      }
                                      style={{
                                        width: "80px",
                                        padding: "2px 4px",
                                        textAlign: "right"
                                      }}
                                    />
                                  ) : (
                                    row.MANUALDEMAND !== undefined && row.MANUALDEMAND !== ""
                                      ? row.MANUALDEMAND
                                      : "-"
                                  )}
                                </td>
                              );
                            }
                            // All other measures are read-only
                            return (
                              <td key={date}>
                                {row[measure.key] !== undefined && row[measure.key] !== "" ? row[measure.key] : "-"}
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
                  <option value="none">No Product</option>
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
                  <option value="none">No Store</option>
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
              disabled={!isEditing}
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
        </div>

        {/* Add the save dialog */}
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

// Add the required CSS
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

  .dialog-content {
    margin: 20px 0;
  }

  .dialog-content .input-group {
    margin-bottom: 15px;
  }

  .dialog-content label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
  }

  .dialog-content input,
  .dialog-content textarea {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
  }

  .dialog-content textarea {
    height: 80px;
    resize: vertical;
  }

  .dialog-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }

  .dialog-actions button {
    padding: 8px 16px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
  }

  .dialog-actions button:first-child {
    background-color: #f0f0f0;
  }

  .dialog-actions button:last-child {
    background-color: #007bff;
    color: white;
  }

  .override-value {
    position: relative;
    cursor: help;
  }

  .override-tooltip {
    display: none;
    position: absolute;
    background: white;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 4px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    z-index: 100;
    width: 200px;
    left: 50%;
    transform: translateX(-50%);
    top: -80px;
  }

  .override-value:hover .override-tooltip {
    display: block;
  }
`;

// Add styles to the document
const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default PlannerWorkbench;