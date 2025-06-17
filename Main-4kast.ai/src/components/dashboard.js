import React, { useState, useMemo } from "react";
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Dialog from '@mui/material/Dialog';
import IconButton from '@mui/material/IconButton';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import CloseIcon from '@mui/icons-material/Close';
import { Typography, InputAdornment, TextField } from '@mui/material';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import GaugeChart from 'react-gauge-chart';
import { ResponsiveLine } from '@nivo/line';
import { format } from 'date-fns';
import { ResponsiveBar } from '@nivo/bar';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import { styled } from '@mui/material/styles';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import DateRangeIcon from '@mui/icons-material/DateRange';
import PublicIcon from '@mui/icons-material/Public';
import CategoryIcon from '@mui/icons-material/Category';
import TodayIcon from '@mui/icons-material/Today';

function Dashboard({ children }) {
  return (
    <Box sx={{ width: '100%', minHeight: '100vh', bgcolor: 'background.default', p: 3 }}>
      <Box sx={{ maxWidth: '1800px', margin: '0 auto', '& > * + *': { mt: 3 } }}>{children}</Box>
    </Box>
  );
}

// ChartCard with expand logic
function ChartCard({ title, children }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Paper sx={{ p: 2, height: '100%', position: 'relative' }}>
        <IconButton size="small" sx={{ position: 'absolute', top: 8, right: 8, zIndex: 2 }} onClick={() => setOpen(true)} aria-label={`Expand ${title}`}>
          <OpenInFullIcon fontSize="small" />
        </IconButton>
        {children}
      </Paper>
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="lg" fullWidth>
        <Box sx={{ position: 'relative', p: 2, bgcolor: 'background.paper' }}>
          <IconButton size="small" sx={{ position: 'absolute', top: 8, right: 8, zIndex: 2 }} onClick={() => setOpen(false)} aria-label={`Close ${title}`}>
            <CloseIcon fontSize="small" />
          </IconButton>
          <Box sx={{ p: 2, minHeight: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{children}</Box>
        </Box>
      </Dialog>
    </>
  );
}

// FilterBar
function FilterBar({ filters, onFilterChange }) {
  const handleChange = (field) => (event) => {
    onFilterChange({ ...filters, [field]: event.target.value });
  };

  return (
    <Paper sx={{ p: 2, mb: 3, borderRadius: 2, boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
      <Typography variant="h2" sx={{ mb: 3, color: '#1976d2', fontWeight: 'bold', textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        4kast.ai Analytics!
      </Typography>

      <Grid container spacing={2} alignItems="center">
        {/* Year Filter */}
        <Grid item xs={12} sm={6} md={2}>
          <FormControl fullWidth size="small" variant="outlined" sx={{ borderRadius: 2, minWidth: 160, py:1.5 }}>
            <InputLabel>Year</InputLabel>
            <Select
              value={filters.year}
              onChange={handleChange('year')}
              label="Year"
              startAdornment={
                <InputAdornment position="start">
                  <TodayIcon sx={{ color: '#2563eb' }} />
                </InputAdornment>
              }
            >
              <MenuItem value={2023}>2023</MenuItem>
              <MenuItem value={2024}>2024</MenuItem>
              <MenuItem value={2025}>2025</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        {/* Region Filter */}
        <Grid item xs={12} sm={6} md={2.5}>
          <FormControl fullWidth size="small" variant="outlined" sx={{ borderRadius: 2, minWidth: 160 }}>
            <InputLabel> Region</InputLabel>
            <Select
              value={filters.region}
              onChange={handleChange('region')}
              label=" Region"
              startAdornment={
                <InputAdornment position="start">
                  <PublicIcon sx={{ color: '#059669' }} />
                </InputAdornment>
              }
            >
              <MenuItem value="All">All</MenuItem>
              <MenuItem value="TX">TX</MenuItem>
              <MenuItem value="NYC">NYC</MenuItem>
              <MenuItem value="CA">CA</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        {/* Product Family Filter */}
        <Grid item xs={12} sm={6} md={2.5}>
          <FormControl fullWidth size="small" variant="outlined" sx={{ borderRadius: 2, minWidth: 160 }}>
            <InputLabel>Product Family</InputLabel>
            <Select
              value={filters.productFamily}
              onChange={handleChange('productFamily')}
              label="Product Family"
              startAdornment={
                <InputAdornment position="start">
                  <CategoryIcon sx={{ color: '#f59e0b' }} />
                </InputAdornment>
              }
            >
              <MenuItem value="All">All</MenuItem>
              <MenuItem value="A">A</MenuItem>
              <MenuItem value="B">B</MenuItem>
              <MenuItem value="C">C</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        {/* Start Date */}
        <Grid item xs={12} sm={6} md={2.5}>
          <TextField
            fullWidth
            size="small"
            type="date"
            label="Start Date"
            variant="outlined"
            value={filters.startDate}
            onChange={handleChange('startDate')}
            InputLabelProps={{ shrink: true }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <DateRangeIcon sx={{ color: '#2563eb' }} />
                </InputAdornment>
              ),
            }}
          />
        </Grid>
        {/* End Date */}
        <Grid item xs={12} sm={6} md={2.5}>
          <TextField
            fullWidth
            size="small"
            type="date"
            label="End Date"
            variant="outlined"
            value={filters.endDate}
            onChange={handleChange('endDate')}
            InputLabelProps={{ shrink: true }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <DateRangeIcon sx={{ color: '#2563eb' }} />
                </InputAdornment>
              ),
            }}
          />
        </Grid>
      </Grid>
    </Paper>
  );
}

// Chart Components
function ForecastActualTimeSeries() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('1M');

  const { startDate, endDate } = useMemo(() => {
    const end = new Date();
    end.setHours(23, 59, 59, 999);
    let start = new Date(end);

    switch (selectedTimeRange) {
      case '5D': start.setDate(end.getDate() - 4); break;
      case '3M': start.setMonth(end.getMonth() - 3); break;
      case '6M': start.setMonth(end.getMonth() - 6); break;
      case 'YTD': start = new Date(end.getFullYear(), 0, 1); break;
      case 'All': start = new Date('2023-01-01'); break;
      case '1M':
      default: start.setMonth(end.getMonth() - 1); break;
    }

    return { startDate: start, endDate: end };
  }, [selectedTimeRange]);

const data = useMemo(() => {
  const pointsHistory = [];
  const pointsForecast = [];
  let currentDate = new Date(startDate);
  let currentValue = 10;

  const totalDays = Math.floor((endDate - startDate) / (1000 * 60 * 60 * 24));
  const splitDate = new Date(startDate);
  splitDate.setDate(startDate.getDate() + Math.floor(totalDays / 2));

  while (currentDate <= endDate) {
    const point = {
      x: new Date(currentDate),
      y: Math.round((currentValue + Math.random() * 5 - 2.5) * 100) / 100,
    };

    if (currentDate < splitDate) {
      pointsHistory.push(point);
    } else if (currentDate.getTime() === splitDate.getTime()) {
      pointsHistory.push(point);     // include this point in both
      pointsForecast.push(point);
    } else {
      pointsForecast.push(point);
    }

    currentValue += Math.random() * 2 - 1;
    currentDate.setDate(currentDate.getDate() + 1);
  }

  return [
    { id: 'Historical', data: pointsHistory },
    { id: 'Forecasted', data: pointsForecast },
  ];
}, [startDate, endDate]);


  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Time Series
        </Typography>
        <ButtonGroup variant="outlined" size="small">
          {['5D', '1M', '3M', '6M', 'YTD', 'All'].map((r) => (
            <Button
              key={r}
              onClick={() => setSelectedTimeRange(r)}
              variant={selectedTimeRange === r ? 'contained' : 'outlined'}
              sx={{
                color: '#1e40af',
                borderColor: '#1e40af',
                '&.MuiButton-contained': {
                  backgroundColor: '#1e40af',
                  color: '#fff',
                },
              }}
            >
              {r}
            </Button>
          ))}
        </ButtonGroup>
      </Box>

      <Box sx={{ height: 300 }}>
        <ResponsiveLine
          data={data}
          xScale={{ type: 'time', format: 'native' }}
          yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
          axisBottom={{
            format: selectedTimeRange === 'All' ? '%Y' : '%b %d',
            tickValues:
              selectedTimeRange === '5D'
                ? 'every day'
                : selectedTimeRange === '1M'
                ? 'every week'
                : selectedTimeRange === '3M'
                ? 'every 2 weeks'
                : selectedTimeRange === '6M'
                ? 'every 45 days'
                : selectedTimeRange === 'YTD'
                ? 'every 2 months'
                : selectedTimeRange === 'All'
                ? 'every year'
                : 'every month',
            tickRotation: -30,
            legend: '',
            legendOffset: 36,
            legendPosition: 'middle',
          }}

          axisLeft={{
            legend: '',
            legendOffset: -40,
            legendPosition: 'middle',
          }}
          enableGridX={false}
          enablePoints={true}
          pointSize={4}
          pointColor="#1e40af"
          pointBorderWidth={1}
          pointBorderColor="#fff"
          useMesh={true}
          curve="monotoneX"
          colors={({ id }) => (id === 'Historical' ? '#1e40af' : '#10b981')}
          markers={[
            {
              axis: 'x',
              value: data[1]?.data?.[0]?.x, // start of forecast
              lineStyle: {
                stroke: '#9ca3af',
                strokeWidth: 2,
                strokeDasharray: '6 6',
              },
              legend: 'Forecast Start',
              legendPosition: 'top-right',
              textStyle: { fill: '#6b7280', fontWeight: 600 },
            },
          ]}

          margin={{ top: 20, right: 20, bottom: 60, left: 50 }}
        />
      </Box>
    </Box>
  );
}


function CorrelationHeatmap() {
  const data = [
    {
      id: 'Demand',
      data: [
        { x: 'Price', y: 0.82 },
        { x: 'Promotion', y: 0.65 },
        { x: 'Holiday', y: 0.45 },
        { x: 'Weather', y: 0.32 },
        { x: 'Inventory', y: 0.18 }
      ]
    },
    {
      id: 'Price',
      data: [
        { x: 'Price', y: 1.0 },
        { x: 'Promotion', y: 0.78 },
        { x: 'Holiday', y: 0.25 },
        { x: 'Weather', y: 0.12 },
        { x: 'Inventory', y: -0.35 }
      ]
    },
    {
      id: 'Promotion',
      data: [
        { x: 'Price', y: 0.78 },
        { x: 'Promotion', y: 1.0 },
        { x: 'Holiday', y: 0.42 },
        { x: 'Weather', y: 0.08 },
        { x: 'Inventory', y: -0.22 }
      ]
    }
  ];

  return (
    <>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Feature Correlation Heatmap
      </Typography>
      <Box sx={{ height: 320 }}>
        <ResponsiveHeatMap
          data={data}
          margin={{ top: 60, right: 90, bottom: 60, left: 90 }}
          valueFormat={(v) => `${(v * 100).toFixed(0)}%`} 
          axisTop={{
            orient: 'top',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: -45,
            legend: '',
          }}
          axisLeft={{
            orient: 'left',
            tickSize: 5,
            tickPadding: 5,
            legend: '',
          }}
          colors={{
            type: 'diverging',
            scheme: 'red_yellow_blue',
            divergeAt: 0.5,
            minValue: -1,
            maxValue: 1,
          }}
          cellBorderWidth={1}
          cellBorderColor="#ffffff"
          labelTextColor={{
            from: 'color',
            modifiers: [['darker', 3]]
          }}
          animate={true}
          enableLabels={true}
          isInteractive={true}
        />
      </Box>
    </>
  );
}


function ErrorDistributionChart() {
  const data = [
    { region: 'TX', 'Under Forecast': 16, 'Over Forecast': 12, 'Accurate': 37 },
    { region: 'Cali', 'Under Forecast': 19, 'Over Forecast': 16, 'Accurate': 24 },
    { region: 'NYC', 'Under Forecast': 18, 'Over Forecast': 14, 'Accurate': 76 },
    { region: 'CHI', 'Under Forecast': 14, 'Over Forecast': 18, 'Accurate': 63 }
  ];
  return (
    <>
      <Typography variant="h6" 
      gutterBottom>Forecast Error Distribution by Region</Typography>
      <Box sx={{ height: 'calc(100% - 40px)' }}>
        <ResponsiveBar data={data} 
        keys={['Under Forecast', 'Over Forecast', 'Accurate']} 
        layout="horizontal"
        indexBy="region" 
        colors={({ id }) => {
          switch (id) {
            case 'Under Forecast': return '#ef4444';     // light red
            case 'Over Forecast': return '#3b82f6';       // light orange
            case 'Accurate': return '#22c55e';            // light green
            default: return '#e5e7eb';                    // fallback neutral
          }
        }}
 />
      </Box>
    </>
  );
}

function DemandVolatilityGauge() {
  const value = 0.65;
  return (
    <>
      <Typography variant="h6" gutterBottom>Demand Volatility Index</Typography>
      <div style={{ width: '100%', height: 260, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <GaugeChart id="demand-volatility-gauge" nrOfLevels={20} percent={value} colors={["#22c55e", "#f59e42", "#ef4444"]} arcWidth={0.3} textColor="#1f2937" needleColor="#2563eb" />
      </div>
      <Typography variant="subtitle1" align="center" sx={{ mt: 2, fontWeight: 600 }}>
        Current Volatility: {Math.round(value * 100)}%
      </Typography>
    </>
  );
}

function ProductSalesBreakdown() {
  const data = [
    { product: 'Product A', sales: 963, forecast: 528 },
    { product: 'Product B', sales: 440, forecast: 1226 },
    { product: 'Product C', sales: 727, forecast: 520 },
    { product: 'Product D', sales: 981, forecast: 1021 },
    { product: 'Product E', sales: 1349, forecast: 755 }
  ];
  return (
    <>
      <Typography variant="h6" gutterBottom>Product Sales vs Forecast</Typography>
      <Box sx={{ height: 'calc(100% - 40px)' }}>
        <ResponsiveBar data={data} keys={['sales', 'forecast']} indexBy="product" colors={["#2563eb", "#818cf8"]} />
      </Box>
    </>
  );
}

function ForecastAccuracyWaterfall() {
  const rawData = [
    { label: 'Total Forecast', value: 0 },
    { label: 'Product A', value: 120 },
    { label: 'Product B', value: -45 },
    { label: 'Product C', value: 78 },
    { label: 'Product D', value: -32 },
    { label: 'Product E', value: 54 },
    { label: 'Total Actual', isTotal: true }
  ];

  let cumulative = 0;
  const chartData = rawData.map((d, i) => {
    const base = cumulative;
    const value = d.value || 0;
    if (!d.isTotal) cumulative += value;

    return {
      label: d.label,
      offset: d.isTotal ? 0 : value > 0 ? base : base + value,
      delta: d.isTotal ? cumulative : Math.abs(value),
      displayValue: d.isTotal ? `+${cumulative}` : value > 0 ? `+${value}` : `${value}`,
      color: d.isTotal ? '#cbd5e1' : value > 0 ? '#3b82f6' : '#60a5fa', // blue tones
    };
  });

  return (
    <>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Forecast Accuracy Analysis
      </Typography>
      <Typography variant="body2" sx={{ mb: 2, color: '#6b7280' }}>
        Breakdown of forecast variance by product
      </Typography>
      <Box sx={{ height: 400 }}>
        <ResponsiveBar
          data={chartData}
          keys={['offset', 'delta']}
          indexBy="label"
          layout="vertical"
          margin={{ top: 20, right: 30, bottom: 80, left: 60 }}
          padding={0.3}
          groupMode="stacked"
          colors={({ id, data }) => {
            if (id === 'offset') return 'rgba(0,0,0,0)';
            return data.color;
          }}
          label={({ id, data }) => {
            if (id === 'delta') return data.displayValue;
            return '';
          }}
          enableLabel={true}
          labelTextColor="#fff"
          borderRadius={2}
          axisBottom={{
            tickRotation: -35,
          }}
          axisLeft={{
            legend: 'Units',
            legendPosition: 'middle',
            legendOffset: -40,
          }}
          tooltip={({ indexValue, value }) => (
            <div style={{ padding: '6px 9px', background: 'white', border: '1px solid #ccc' }}>
              <strong>{indexValue}</strong>: {value}
            </div>
          )}
        />
      </Box>
    </>
  );
}

// KPI Components
const KPIItem = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  borderRadius: 8,
  boxShadow: '0 4px 12px 0 rgba(0,0,0,0.05)',
  transition: 'transform 0.2s, box-shadow 0.2s',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 6px 16px 0 rgba(0,0,0,0.1)',
  },
}));

const KPIValue = styled(Typography)({
  fontSize: '1.8rem',
  fontWeight: 800,
  margin: '8px 0',
  color: '#1e40af',
});

function KPISection({ data }) {
  const kpis = [
    { 
      title: 'Total Demand', 
      value: data.totalDemand.toLocaleString(),
      description: 'Units',
      trend: 'neutral',
      color: '#1e40af',
      icon: '📊'
    },
    { 
      title: 'MAPE', 
      value: `${data.mape}%`,
      description: 'Mean Absolute % Error',
      trend: data.mape < 10 ? 'good' : data.mape < 20 ? 'warning' : 'error',
      color: '#dc2626',
      icon: '📉'
    },
    { 
      title: 'Weighted MAPE', 
      value: `${data.weightedMape}%`,
      description: 'Volume-weighted MAPE',
      trend: data.weightedMape < 10 ? 'good' : data.weightedMape < 20 ? 'warning' : 'error',
      color: '#059669',
      icon: '⚖️'
    },
    { 
      title: 'MAE', 
      value: data.mae.toLocaleString(),
      description: 'Mean Absolute Error',
      trend: 'neutral',
      color: '#7c3aed',
      icon: '📏'
    },
    { 
      title: 'Forecast Bias', 
      value: `${data.forecastBias}%`,
      description: 'Average forecast bias',
      trend: Math.abs(data.forecastBias) < 5 ? 'good' : 'warning',
      color: '#d97706',
      icon: '🎯'
    },
  ];

  const getTrendColor = (trend) => {
    return '#1e40af'
  };

  return (
    <Grid container spacing={2}>
      {kpis.map((kpi, index) => (
        <Grid item xs={12} sm={6} md={2.4} key={index}>
          <KPIItem 
            elevation={0}
            sx={{
              borderLeft: `4px solid #1e40af`,
              backgroundColor: 'white',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 8px 16px 0 rgba(0,0,0,0.1)',
              },
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ color: '#4b5563', fontWeight: 600 }}>
                {kpi.title}
              </Typography>
              <span style={{ fontSize: '1.2rem' }}>{kpi.icon}</span>
            </Box>
            <KPIValue variant="h4" sx={{ color: '#1e40af', fontSize: '2rem' }}>
              {kpi.value}
            </KPIValue>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 1 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: getTrendColor(kpi.trend), mr: 1 }} />
              <Typography variant="caption" sx={{ color: '#6b7280', fontSize: '0.7rem' }}>
                {kpi.description}
              </Typography>
            </Box>
          </KPIItem>
        </Grid>
      ))}
    </Grid>
  );
}

// Main App
const theme = createTheme({
  palette: {
    primary: { main: '#1e40af', light: '#3b82f6', dark: '#1e3a8a', contrastText: '#ffffff' },
    secondary: { main: '#3b82f6', light: '#60a5fa', dark: '#2563eb', contrastText: '#ffffff' },
    background: { default: '#f8fafc', paper: '#ffffff' },
    text: { primary: '#1f2937', secondary: '#4b5563' },
  },
  typography: { fontFamily: ['Inter', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', '"Helvetica Neue"', 'Arial', 'sans-serif'].join(','), h1: { fontWeight: 800 }, h2: { fontWeight: 800 }, h3: { fontWeight: 800 }, h4: { fontWeight: 800 }, h5: { fontWeight: 700 }, h6: { fontWeight: 700 }, subtitle1: { fontWeight: 600 }, subtitle2: { fontWeight: 600 }, body1: { fontWeight: 500 }, body2: { fontWeight: 500 }, button: { fontWeight: 600, textTransform: 'none' }, caption: { fontWeight: 500 }, overline: { fontWeight: 600 } },
  shape: { borderRadius: 8 },
});

function Dashboard1() {
  const [filters, setFilters] = useState({
    year: 2025,
    region: 'All',
    productFamily: 'All',
    startDate: '2025-05-01',
    endDate: '2025-06-01'
  });


  const kpiData = {
    totalDemand: 125000,
    mape: 15.2,
    weightedMape: 12.8,
    mae: 8500,
    forecastBias: 3.5
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Dashboard>
        <FilterBar filters={filters} onFilterChange={setFilters} />
        <Box sx={{ mt: 3, px: 2 }}>
          {/* KPI Section */}
          <Box sx={{ mb: 3 }}>
            <KPISection data={kpiData} />
          </Box>
          {/* Time Series Chart */}
          <Box sx={{ mt: 4, mb: 2 }}>
            <ChartCard title="Time Series">
              <ForecastActualTimeSeries />
            </ChartCard>
          </Box>
          {/* Dashboard Charts in Grid */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <ChartCard title="Feature Correlation Heatmap">
                <CorrelationHeatmap />
              </ChartCard>
            </Grid>
            <Grid item xs={12} md={6}>
              <ChartCard title="Forecast Error Distribution by Region">
                <ErrorDistributionChart />
              </ChartCard>
            </Grid>
            <Grid item xs={12} md={6}>
              <ChartCard title="Demand Volatility Index">
                <DemandVolatilityGauge />
              </ChartCard>
            </Grid>
            <Grid item xs={12} md={6}>
              <ChartCard title="Product Sales vs Forecast">
                <ProductSalesBreakdown />
              </ChartCard>
            </Grid>
          </Grid>
          {/* Full Width Chart */}
          <Box sx={{ backgroundColor: 'white', borderRadius: 2, boxShadow: '0 4px 12px 0 rgba(0,0,0,0.05)', p: 2, mb: 3, '&:hover': { boxShadow: '0 8px 16px 0 rgba(0,0,0,0.1)' } }}>
            <ForecastAccuracyWaterfall />
          </Box>
        </Box>
      </Dashboard>
    </ThemeProvider>
  );
}

export default Dashboard1;