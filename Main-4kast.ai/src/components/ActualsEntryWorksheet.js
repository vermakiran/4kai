import React, { useState, useMemo, useEffect } from 'react';
import moment from 'moment';

const ActualsEntryWorksheet = ({ forecastData, onSave }) => {
  const [actualsEdits, setActualsEdits] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  // When new forecast data is loaded from the parent, clear any old edits
  useEffect(() => {
    setActualsEdits({});
  }, [forecastData]);

  // This logic groups the flat array from the API into a nested structure for easy rendering
  const { groupedData, allDates } = useMemo(() => {
    if (!forecastData || forecastData.length === 0) {
      return { groupedData: {}, allDates: [] };
    }
    const grouped = {};
    const dates = new Set();
    forecastData.forEach(row => {
      const key = `${row.store || 'N/A'}||${row.product || 'N/A'}`;
      if (!grouped[key]) {
        grouped[key] = { store: row.store, product: row.product, dates: {} };
      }
      grouped[key].dates[row.date] = {
        forecast: row.forecast_quantity,
        actual: row.actual_quantity,
      };
      dates.add(row.date);
    });
    const sortedDates = Array.from(dates).sort();
    return { groupedData: grouped, allDates: sortedDates };
  }, [forecastData]);

  const handleEdit = (key, date, value) => {
    const numericValue = value === '' ? null : parseFloat(value);
    if (value !== '' && (isNaN(numericValue) || numericValue < 0)) {
        // Prevent non-numeric or negative values
        return; 
    }
    
    setActualsEdits(prev => ({
      ...prev,
      [key]: {
        ...prev[key],
        [date]: numericValue,
      },
    }));
  };
  
  const handleSaveClick = async () => {
    setIsLoading(true);
    const updates = [];
    Object.entries(actualsEdits).forEach(([entityKey, dateEdits]) => {
      const [store, product] = entityKey.split('||');
      Object.entries(dateEdits).forEach(([date, value]) => {
        // We only send updates for cells that were actually edited to a non-null value
        if (value !== null && value !== undefined) {
            updates.push({
                store: store === 'N/A' ? null : store,
                product: product,
                date: date,
                actual_units: value,
            });
        }
      });
    });

    try {
        await onSave(updates); // Call the onSave function passed down from the parent
        setActualsEdits({});   // Clear local edits on successful save
    } catch (error) {
        console.error("Failed to save actuals:", error);
        alert("Error: Could not save actuals. Please check the console.");
    } finally {
        setIsLoading(false);
    }
  };

  if (!forecastData || forecastData.length === 0) {
    return <p style={{ padding: '20px', color: '#666' }}>Load a forecast to begin entering actuals.</p>;
  }

  return (
    <div className="worksheet-container" style={{ marginTop: '20px' }}>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Entity</th>
              <th>Measure</th>
              {allDates.map(date => (
                <th key={date}>{moment(date).format("DD MMM YYYY")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(groupedData).map(([key, entityData]) => {
              const { store, product } = entityData;
              return (
                <React.Fragment key={key}>
                  {/* Row 1: Forecast Quantity (Read-only) */}
                  <tr>
                    <td rowSpan={3} style={{ fontWeight: 'bold', verticalAlign: 'middle' }}>{`${store} - ${product}`}</td>
                    <td>Forecast Quantity</td>
                    {allDates.map(date => (
                      <td key={`${key}-forecast-${date}`} style={{ textAlign: 'right', backgroundColor: '#f8f9fa', color: '#6c757d' }}>
                        {entityData.dates[date]?.forecast?.toFixed(2) ?? '-'}
                      </td>
                    ))}
                  </tr>
                  {/* Row 2: Actual Quantity (Editable) */}
                  <tr>
                    <td style={{ fontWeight: 500 }}>Actual Quantity</td>
                    {allDates.map(date => {
                      const initialValue = entityData.dates[date]?.actual;
                      const editedValue = actualsEdits[key]?.[date];
                      const displayValue = editedValue !== undefined ? editedValue : initialValue;

                      return (
                        <td key={`${key}-actual-${date}`} style={{ backgroundColor: '#fff3cd', padding: '2px' }}>
                          <input
                            type="number"
                            value={displayValue ?? ''}
                            onChange={(e) => handleEdit(key, date, e.target.value)}
                            style={{ width: '80px', padding: '4px', textAlign: 'right', border: '1px solid #ccc', borderRadius: '3px' }}
                            placeholder="Enter..."
                            min="0"
                          />
                        </td>
                      );
                    })}
                  </tr>
                  {/* Row 3: Variance (Calculated) */}
                  <tr>
                    <td style={{ fontStyle: 'italic' }}>Variance (%)</td>
                    {allDates.map(date => {
                      const forecast = entityData.dates[date]?.forecast;
                      const actual = actualsEdits[key]?.[date] ?? entityData.dates[date]?.actual;
                      let variance = '-';
                      let color = 'inherit';

                      if (typeof forecast === 'number' && typeof actual === 'number') {
                        if(forecast > 0) {
                            const diff = ((actual - forecast) / forecast) * 100;
                            variance = `${diff.toFixed(1)}%`;
                            color = diff < 0 ? '#dc3545' : '#198754'; // Red for negative, green for positive
                        } else if (actual > 0) {
                            variance = '∞%'; // Infinite variance if forecast was 0 but sales occurred
                            color = '#198754';
                        }
                      }
                      
                      return (
                         <td key={`${key}-variance-${date}`} style={{ textAlign: 'right', color: color, fontWeight: 'bold' }}>
                           {variance}
                         </td>
                      );
                    })}
                  </tr>
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ textAlign: 'right', marginTop: '20px' }}>
        <button 
          className="save-button"
          onClick={handleSaveClick}
          disabled={Object.keys(actualsEdits).length === 0 || isLoading}
        >
          {isLoading ? 'Saving...' : 'Save Actuals'}
        </button>
      </div>
    </div>
  );
};

export default ActualsEntryWorksheet;