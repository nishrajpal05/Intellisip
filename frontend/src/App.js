import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Droplets, Activity, AlertTriangle, TrendingUp, 
  Calendar, Clock, Thermometer, Target 
} from 'lucide-react';
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter
} from 'recharts';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [dailyData, setDailyData] = useState([]);
  const [hourlyData, setHourlyData] = useState([]);
  const [anomalyAlerts, setAnomalyAlerts] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    todayTotal: 0,
    avgDaily: 0,
    totalSips: 0,
    avgTemp: 0
  });

  // Fetch all data
  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      
      // Fetch daily intake
      const dailyRes = await axios.get(`${API_BASE}/daily-intake?days=7`);
      setDailyData(dailyRes.data.reverse());
      
      // Fetch hourly breakdown
      const hourlyRes = await axios.get(`${API_BASE}/hourly-breakdown`);
      setHourlyData(hourlyRes.data);
      
      // Fetch anomaly alerts
      const anomalyRes = await axios.get(`${API_BASE}/anomaly-alerts?days=7`);
      setAnomalyAlerts(anomalyRes.data);
      
      // Calculate stats
      if (dailyRes.data.length > 0) {
        const today = dailyRes.data[dailyRes.data.length - 1];
        const avgDaily = dailyRes.data.reduce((sum, d) => sum + d.total_ml, 0) / dailyRes.data.length;
        const totalSips = dailyRes.data.reduce((sum, d) => sum + d.sip_count, 0);
        const avgTemp = dailyRes.data.reduce((sum, d) => sum + d.avg_temp, 0) / dailyRes.data.length;
        
        setStats({
          todayTotal: today.total_ml,
          avgDaily: avgDaily,
          totalSips: totalSips,
          avgTemp: avgTemp
        });
      }
      
      // Mock prediction (in real scenario, calculate from latest data)
      const now = new Date();
      setPrediction({
        predicted_ml: Math.random() * 150 + 100,
        hour: now.getHours() + 1,
        confidence: 'high'
      });
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <Droplets className="loading-icon" size={64} />
        <p>Loading your hydration insights...</p>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Droplets size={32} />
            <h1>IntelliSip</h1>
          </div>
          <div className="header-stats">
            <div className="header-stat">
              <Calendar size={20} />
              <span>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
            </div>
            <div className="header-stat">
              <Clock size={20} />
              <span>{new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Cards */}
      <div className="stats-grid">
        <StatCard
          icon={<Droplets size={28} />}
          title="Today's Intake"
          value={`${Math.round(stats.todayTotal)} ml`}
          subtitle={`Goal: 2000ml (${Math.round((stats.todayTotal / 2000) * 100)}%)`}
          color="blue"
        />
        <StatCard
          icon={<TrendingUp size={28} />}
          title="7-Day Average"
          value={`${Math.round(stats.avgDaily)} ml`}
          subtitle={`${stats.avgDaily > 2000 ? '+' : ''}${Math.round(stats.avgDaily - 2000)} ml from goal`}
          color="green"
        />
        <StatCard
          icon={<Activity size={28} />}
          title="Total Sips"
          value={stats.totalSips}
          subtitle="Past 7 days"
          color="purple"
        />
        <StatCard
          icon={<Thermometer size={28} />}
          title="Avg Temperature"
          value={`${Math.round(stats.avgTemp)}°C`}
          subtitle="Environmental average"
          color="orange"
        />
      </div>

      {/* Main Dashboard */}
      <div className="dashboard-grid">
        {/* Daily Intake Chart */}
        <div className="card large">
          <div className="card-header">
            <h2>Weekly Hydration Trend</h2>
            <p>Your water intake over the past 7 days</p>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={dailyData}>
              <defs>
                <linearGradient id="colorIntake" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="date" 
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8' }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="total_ml" 
                stroke="#3b82f6" 
                fillOpacity={1} 
                fill="url(#colorIntake)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Breakdown */}
        <div className="card large">
          <div className="card-header">
            <h2> Hourly Breakdown</h2>
            <p>Water consumption by hour today</p>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="hour" 
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8' }}
              />
              <YAxis 
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Bar 
                dataKey="total_ml" 
                fill="#10b981" 
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Prediction Card */}
        <div className="card">
          <div className="card-header">
            <h2> AI Prediction</h2>
            <p>Next hour recommendation</p>
          </div>
          <div className="prediction-content">
            <div className="prediction-value">
              {prediction && Math.round(prediction.predicted_ml)}
              <span className="prediction-unit">ml</span>
            </div>
            <p className="prediction-text">
              Based on your patterns, you should drink around{' '}
              <strong>{prediction && Math.round(prediction.predicted_ml)}ml</strong>{' '}
              in the next hour.
            </p>
            <div className="prediction-confidence">
              <Target size={16} />
              <span>High Confidence</span>
            </div>
          </div>
        </div>

        {/* Anomaly Alerts */}
        <div className="card">
          <div className="card-header">
            <h2> Health Alerts</h2>
            <p>Unusual patterns detected</p>
          </div>
          <div className="alerts-list">
            {anomalyAlerts.length === 0 ? (
              <div className="no-alerts">
                <Activity size={48} />
                <p>No anomalies detected!</p>
                <span>Your hydration is on track 🎉</span>
              </div>
            ) : (
              anomalyAlerts.slice(0, 5).map((alert, idx) => (
                <div key={idx} className={`alert-item severity-${alert.severity}`}>
                  <AlertTriangle size={20} />
                  <div className="alert-content">
                    <strong>{alert.anomaly_type}</strong>
                    <p>{alert.message}</p>
                    <span className="alert-time">
                      {new Date(alert.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Temperature vs Intake */}
        <div className="card large">
          <div className="card-header">
            <h2> Temperature Impact</h2>
            <p>Correlation between temperature and intake</p>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="avg_temp" 
                name="Temperature" 
                unit="°C"
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8' }}
              />
              <YAxis 
                dataKey="total_ml" 
                name="Intake" 
                unit="ml"
                stroke="#94a3b8"
                tick={{ fill: '#94a3b8' }}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Scatter 
                name="Daily Data" 
                data={dailyData} 
                fill="#f59e0b"
                shape="circle"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>Powered By • IntelliSip v1.0 ©️NSR </p>
      </footer>
    </div>
  );
}

// Reusable Stat Card Component
function StatCard({ icon, title, value, subtitle, color }) {
  return (
    <div className={`stat-card stat-${color}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-info">
        <h3>{title}</h3>
        <div className="stat-value">{value}</div>
        <p className="stat-subtitle">{subtitle}</p>
      </div>
    </div>
  );
}

export default App;