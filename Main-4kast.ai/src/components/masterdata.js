import React, { useState } from "react";

const MasterData = () => {
  const [selectedMaster, setSelectedMaster] = useState("Product");
  const [showTable, setShowTable] = useState(false);
  const [editMode, setEditMode] = useState(false);

  const demoData = [
    { slNo: 1, productCode: "1080", description: "Galaxy Flute Single", category: "CAT001", brand: "ALG100", family: "FAM001", uom: "EA" },
    { slNo: 2, productCode: "1099", description: "Maltesers", category: "CAT001", brand: "ALG101", family: "FAM001", uom: "EA" },
    { slNo: 3, productCode: "2318", description: "Thomas Cat Litter", category: "CAT001", brand: "ALG102", family: "FAM001", uom: "EA" },
    { slNo: 4, productCode: "7219", description: "Cat Litter", category: "CAT002", brand: "ALG103", family: "FAM001", uom: "EA" },
    { slNo: 5, productCode: "1982", description: "Hersheys", category: "CAT002", brand: "ALG105", family: "FAM001", uom: "EA" },
  ];

  const handleAddUpdate = () => {
    setShowTable(true);
  };

  const handleEdit = () => {
    setEditMode(!editMode);
  };

  const handleRemove = () => {
    setShowTable(false);
    setEditMode(false);
  };

  const handleSave = () => {
    const confirmSave = window.confirm("Do you really want to save?");
    if (confirmSave) {
      alert("Saved successfully!");
    } else {
      window.location.reload();
    }
  };

  return (
    <div className="master-data-container">
      <h2>Master Data</h2>
      <div className="dropdown-container">
        <label>Select Master Data: </label>
        <select value={selectedMaster} onChange={(e) => setSelectedMaster(e.target.value)}>
          <option value="Product">Product</option>
          <option value="Location">Location</option>
        </select>
      </div>

      {showTable ? (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                {editMode && <th>Select</th>}
                <th>SL.NO</th>
                <th>Product Code</th>
                <th>Product Description</th>
                <th>Product Category</th>
                <th>Brand</th>
                <th>Product Family</th>
                <th>Base UoM</th>
              </tr>
            </thead>
            <tbody>
              {demoData.map((row, index) => (
                <tr key={index}>
                  {editMode && <td><input type="checkbox" /></td>}
                  <td>{row.slNo}</td>
                  <td>{row.productCode}</td>
                  <td>{row.description}</td>
                  <td>{row.category}</td>
                  <td>{row.brand}</td>
                  <td>{row.family}</td>
                  <td>{row.uom}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-box">
          <p>Select your Master Data</p>
        </div>
      )}

      <div className="button-group">
        <button className="btn edit" onClick={handleEdit} disabled={!showTable}>Edit</button>
        <button className="btn add-update" onClick={handleAddUpdate}>Add/Update</button>
        <button className="btn remove" onClick={handleRemove} disabled={!showTable}>Remove</button>
        <button className="btn save" onClick={handleSave} disabled={!showTable}>Save</button>
      </div>
    </div>
  );
};

export default MasterData;
