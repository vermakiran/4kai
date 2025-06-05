import React, { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, PieChart, Pie, Cell, Legend, LabelList
} from "recharts";
import Heatmap from "react-heatmap-grid";
import { DASHBOARD_ENDPOINT } from "./config";
import Cookies from "js-cookie";
import CountUp from 'react-countup';

// Color palette
const COLORS = [
  "#002855", "#00509E", "#007BFF", "#1A1F71",
  "#00BFFF", "#4682B4", "#274472", "#0A1172"
];

// Chart Modal Component
function ChartModal({ open, onClose, children }) {
  if (!open) return null;
  return (
    <div className="chart-modal-backdrop" onClick={onClose}>
      <div className="chart-modal-content" onClick={e => e.stopPropagation()}>
        <button className="chart-modal-close" onClick={onClose}>×</button>
        {children}
      </div>
    </div>
  );
}

const defaultKpiData = {
  total_demand: 0,
  mape: 0,
  rmse: 0,
  bias: 0,
  fva: 0,
  num_products: 0,
  weighted_mape: 0,
  latest_runid: 0,
  chartData: {}
};

function WaterfallChart({ data }) {
  let cumulative = 0;
  const processed = data.map((d) => {
    const start = cumulative;
    cumulative += d.impact;
    return {
      ...d,
      start,
      end: cumulative,
      isTotal: d.name === "Total"
    };
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={processed}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip formatter={val => `${val}`} />
        <Bar dataKey="start" stackId="a" fill="transparent" />
        <Bar
          dataKey={d => d.end - d.start}
          stackId="a"
          fill="#007BFF"
          isAnimationActive={false}
        >
          <LabelList dataKey={d => d.end - d.start} position="top" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// Utility functions
function safeArray(data) {
  return Array.isArray(data) ? data : [];
}

// Normalize heatmap for Top-N only
function normalizeHeatmapTop(heatmapRaw, yLabels, xLabels) {
  return yLabels.map(store =>
    xLabels.map(product => {
      const entry = heatmapRaw.find(
        e => e.product === product && e.store === store
      );
      return entry ? entry.avg_mape : 0;
    })
  );
}

function Dashboard() {
  // State for filters
  const [products, setProducts] = useState([]);
  const [models, setModels] = useState([]);
  const [minDate, setMinDate] = useState("");
  const [maxDate, setMaxDate] = useState("");
  const [storeList, setStoreList] = useState([]);

  // Filter panel (pending)
  const [pendingProduct, setPendingProduct] = useState("");
  const [pendingModel, setPendingModel] = useState("");
  const [pendingStartDate, setPendingStartDate] = useState("");
  const [pendingEndDate, setPendingEndDate] = useState("");
  const [pendingStore, setPendingStore] = useState("");

  // Active filters
  const [selectedProduct, setSelectedProduct] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedStartDate, setSelectedStartDate] = useState("");
  const [selectedEndDate, setSelectedEndDate] = useState("");
  const [selectedStore, setSelectedStore] = useState("");

  // Data
  const [kpiData, setKpiData] = useState(defaultKpiData);
  const [chartData, setChartData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedChart, setExpandedChart] = useState(null);

  // Top-N for heatmap
  const topProducts = safeArray(chartData.topProducts);
  const topStores = safeArray(chartData.topStores);
  const heatmapDataTop = safeArray(chartData.heatmapData_top);

  const normalizedHeatmapTop =
    topStores.length && topProducts.length
      ? normalizeHeatmapTop(heatmapDataTop, topStores, topProducts)
      : [];

  // Initial API call to populate filters and defaults
  useEffect(() => {
    const fetchInitial = async () => {
      setLoading(true);
      setError(null);
      try {
        const token = Cookies.get("authToken");
        const response = await fetch(DASHBOARD_ENDPOINT, {
          headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
          }
        });
        if (!response.ok) throw new Error("Failed to fetch dashboard data");
        const data = await response.json();

        setProducts(data.data.productList || []);
        setModels(data.data.modelList || []);
        setMinDate(data.data.minDate || "");
        setMaxDate(data.data.maxDate || "");
        setStoreList(data.data.storeList || []);

        // Set pending and selected to default values (first item or min/max dates)
        const defaultProduct = (data.data.productList && data.data.productList[0]) || "";
        const defaultModel = (data.data.modelList && data.data.modelList[0]) || "";
        const defaultStart = data.data.minDate || "";
        const defaultEnd = data.data.maxDate || "";

        setPendingProduct(defaultProduct);
        setPendingModel(defaultModel);
        setPendingStartDate(defaultStart);
        setPendingEndDate(defaultEnd);
        setPendingStore("");

        setSelectedProduct(defaultProduct);
        setSelectedModel(defaultModel);
        setSelectedStartDate(defaultStart);
        setSelectedEndDate(defaultEnd);
        setSelectedStore("");

        setKpiData(data.data.kpiData || defaultKpiData);
        setChartData((data.data.kpiData && data.data.kpiData.chartData) || {});
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchInitial();
  }, []);

  // Fetch dashboard data every time filters change (Apply)
  useEffect(() => {
    if (!selectedProduct && !selectedModel && !selectedStartDate && !selectedEndDate) return;
    const fetchDashboardData = async () => {
      setLoading(true);
      setError(null);
      try {
        let url = DASHBOARD_ENDPOINT;
        const params = [];
        if (selectedProduct) params.push(`product=${encodeURIComponent(selectedProduct)}`);
        if (selectedStartDate) params.push(`start=${encodeURIComponent(selectedStartDate)}`);
        if (selectedEndDate) params.push(`end=${encodeURIComponent(selectedEndDate)}`);
        if (selectedStore) params.push(`store=${encodeURIComponent(selectedStore)}`);
        if (params.length) url += "?" + params.join("&");

        const token = Cookies.get("authToken");
        const response = await fetch(url, {
          headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
          }
        });
        if (!response.ok) throw new Error("Failed to fetch dashboard data");
        const data = await response.json();

        setKpiData(data.data.kpiData || defaultKpiData);
        setChartData((data.data.kpiData && data.data.kpiData.chartData) || {});
        setProducts(data.data.productList || []);
        setModels(data.data.modelList || []);
        setMinDate(data.data.minDate || "");
        setMaxDate(data.data.maxDate || "");
        setStoreList(data.data.storeList || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboardData();
    // eslint-disable-next-line
  }, [selectedProduct, selectedModel, selectedStartDate, selectedEndDate, selectedStore]);

  // Chart data with fallback
  const lineData = safeArray(chartData.lineData);
  const barData = safeArray(chartData.barData);
  const pieDataAll = safeArray(chartData.pieData_all);
  const waterfallData = safeArray(chartData.waterfallData);

  // Handlers for filters
  const handleApply = () => {
    setSelectedProduct(pendingProduct);
    setSelectedModel(pendingModel);
    setSelectedStore(pendingStore);
    setSelectedStartDate(pendingStartDate);
    setSelectedEndDate(pendingEndDate);
  };
  const handleReset = () => {
    setPendingProduct(products[0] || "");
    setPendingModel(models[0] || "");
    setPendingStartDate(minDate || "");
    setPendingEndDate(maxDate || "");
    setPendingStore(storeList[0] || "");
    setSelectedStore(storeList[0] || "");

    setSelectedProduct(products[0] || "");
    setSelectedModel(models[0] || "");
    setSelectedStartDate(minDate || "");
    setSelectedEndDate(maxDate || "");
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div style={{ color: 'red' }}>Error: {error}</div>;

  function getHeatmapColor(value) {
  // Value assumed to be 0-100 (MAPE%).
  // Blue for low, yellow for mid, red for high.
  if (value === null || value === undefined) return "#e0e7ef";
  if (value < 15) return "#3AB795"; // Greenish for high accuracy
  if (value < 30) return "#FFE156"; // Yellow for mid
  return "#FF6464"; // Red for low accuracy
}

// Optional: Show value on hover
function cellTooltip(x, y, value) {
  return `${y} × ${x}: ${value !== null ? value.toFixed(2) + "%" : "No data"}`;
}

  return (
    <div className="dashboard">
      <h2 className="dashboard-title">
        Forecast Analytics Dashboard <span style={{ fontWeight: 400, fontSize: 22 }}>(Forecast ID: {kpiData.latest_forecastid})</span>
      </h2>
      {/* --- KPI CARDS: GRID --- */}
      <div className="kpi-row">
        <div className="kpi-card" style={{ backgroundColor: COLORS[0] }}>
          <h4>Total Forecasted Demand</h4>
          <h2>
            <CountUp end={kpiData.total_demand || 0} prefix="#" separator="," decimals={2} duration={1.2} />
          </h2>
        </div>
        <div className="kpi-card" style={{ backgroundColor: COLORS[1] }}>
          <h4>MAPE</h4>
          <h2>
            <CountUp end={kpiData.mape || 0} suffix="%" decimals={2} duration={1.1} />
          </h2>
        </div>
        <div className="kpi-card" style={{ backgroundColor: COLORS[2] }}>
          <h4>MAE</h4>
          <h2>
            <CountUp end={kpiData.mae || 0} decimals={2} duration={1.1} />
          </h2>
        </div>
        <div className="kpi-card" style={{ backgroundColor: COLORS[3] }}>
          <h4>Forecast Accuracy</h4>
          <h2>
            <CountUp end={kpiData.fva || 0} suffix="%" decimals={2} duration={1.1} />
          </h2>
        </div>
        <div className="kpi-card" style={{ backgroundColor: COLORS[4] }}>
          <h4>Forecast Bias</h4>
          <h2>
            <CountUp end={kpiData.bias || 0} decimals={2} duration={1.1} />
          </h2>
        </div>
        <div className="kpi-card" style={{ backgroundColor: COLORS[5] }}>
          <h4>Number of Products</h4>
          <h2>
            <CountUp end={kpiData.num_products || 0} duration={1.1} />
          </h2>
        </div>
        <div className="kpi-card" style={{ backgroundColor: COLORS[6] }}>
          <h4>Weighted MAPE</h4>
          <h2>
            <CountUp end={kpiData.weighted_mape || 0} suffix="%" decimals={2} duration={1.1} />
          </h2>
        </div>
      </div>

      {/* --- MAIN CONTENT: VERTICAL FILTER SIDEBAR + CHARTS --- */}
      <div className="dashboard-main-content">
        <div className="dashboard-filters-vertical">
          <div className="filter-group">
            <label>Product:</label>
            <select value={pendingProduct} onChange={e => setPendingProduct(e.target.value)}>
              {products.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="filter-group">
            <label>Model:</label>
            <select value={pendingModel} onChange={e => setPendingModel(e.target.value)}>
              {models.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          <div className="filter-group">
            <label>Store:</label>
            <select value={pendingStore} onChange={e => setPendingStore(e.target.value)}>
              <option value="">All Stores</option>
              {storeList.map((store) => (
                <option key={store} value={store}>
                  {store}
                </option>
              ))}
            </select>
          </div>
          <div className="filter-group">
            <label>Start Date:</label>
            <input
              type="date"
              value={pendingStartDate}
              min={minDate}
              max={maxDate}
              onChange={e => setPendingStartDate(e.target.value)}
            />
          </div>
          <div className="filter-group">
            <label>End Date:</label>
            <input
              type="date"
              value={pendingEndDate}
              min={minDate}
              max={maxDate}
              onChange={e => setPendingEndDate(e.target.value)}
            />
          </div>
          <div className="filter-buttons-vertical">
            <button onClick={handleApply}>Apply Filters</button>
            <button onClick={handleReset}>Reset Filters</button>
          </div>
        </div>

        <div className="dashboard-charts-vertical">
          <div className="dashboard-charts-grid">
            {/* Line Chart */}
            <div className="chart-box chart-box-full" style={{ gridColumn: "1 / -1" }}>
              <button
                className="chart-expand-btn"
                title="Expand chart"
                onClick={() => setExpandedChart("line")}
              >⛶</button>
              <h4>Historical vs Forecasted Demand</h4>
              {lineData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={lineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis tickFormatter={value => `${value.toLocaleString()}`} />
                    <Tooltip formatter={value => `${value.toLocaleString()}`} />
                    <Legend />
                    <Line type="monotone" dataKey="value" stroke="#002855" name="Actual" strokeWidth={2} />
                    <Line type="monotone" dataKey="forecast" stroke="#00509E" name="Forecast" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <p>No data available.</p>}
            </div>

            {/* Bar Chart */}
            <div className="chart-box">
              <button
                className="chart-expand-btn"
                title="Expand chart"
                onClick={() => setExpandedChart("bar")}
              >⛶</button>
              <h4>Product Sales Bar Chart</h4>
              {barData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={barData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="sales" fill="#007BFF" name="Sales" />
                  </BarChart>
                </ResponsiveContainer>
              ) : <p>No bar chart data available.</p>}
            </div>

            {/* Waterfall Chart */}
            <div className="chart-box">
              <button
                className="chart-expand-btn"
                title="Expand chart"
                onClick={() => setExpandedChart("waterfall")}
              >⛶</button>
              <h4>Forecast Accuracy Impact by Product</h4>
              {waterfallData.length > 0 ? (
                <WaterfallChart data={waterfallData} />
              ) : <p>No data available for waterfall chart.</p>}
            </div>

            {/* Pie Chart */}
            <div className="chart-box">
              <button
                className="chart-expand-btn"
                title="Expand chart"
                onClick={() => setExpandedChart("pie")}
              >⛶</button>
              <h4>Forecast Accuracy Distribution</h4>
              {pieDataAll.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieDataAll}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}%`}
                      labelLine={true}
                    >
                      {pieDataAll.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : <p>No pie chart data available.</p>}
            </div>
            {/* --- Modals for Expanding Each Chart --- */}
            <ChartModal open={expandedChart === "line"} onClose={() => setExpandedChart(null)}>
              <h3>Historical vs Forecasted Demand</h3>
              <ResponsiveContainer width="100%" height={500}>
                <LineChart data={lineData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={value => `${value.toLocaleString()}`} />
                  <Tooltip formatter={value => `${value.toLocaleString()}`} />
                  <Legend />
                  <Line type="monotone" dataKey="value" stroke="#002855" name="Actual" strokeWidth={2} />
                  <Line type="monotone" dataKey="forecast" stroke="#00509E" name="Forecast" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </ChartModal>

            <ChartModal open={expandedChart === "bar"} onClose={() => setExpandedChart(null)}>
              <h3>Product Sales Bar Chart</h3>
              <ResponsiveContainer width="100%" height={500}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="sales" fill="#007BFF" name="Sales" />
                </BarChart>
              </ResponsiveContainer>
            </ChartModal>

            <ChartModal open={expandedChart === "waterfall"} onClose={() => setExpandedChart(null)}>
              <h3>Forecast Accuracy Impact by Product</h3>
              <div style={{ width: "100%", height: "calc(100% - 2.5rem)" }}>
                <WaterfallChart data={waterfallData} />
              </div>
            </ChartModal>

            <ChartModal open={expandedChart === "pie"} onClose={() => setExpandedChart(null)}>
              <h3>Forecast Accuracy Distribution</h3>
              <ResponsiveContainer width="100%" height={500}>
                <PieChart>
                  <Pie
                    data={pieDataAll}
                    cx="50%"
                    cy="50%"
                    outerRadius={180}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}%`}
                    labelLine={true}
                  >
                    {pieDataAll.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </ChartModal>
          </div>
        </div>
      </div>
    </div>
  );
}
export default Dashboard;