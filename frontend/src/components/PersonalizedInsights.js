import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TrendingUp, Award, Calendar, Target } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function PersonalizedInsights() {
  const [weeklyReport, setWeeklyReport] = useState(null);
  const [personality, setPersonality] = useState(null);
  const [todayPrediction, setTodayPrediction] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchInsights();
  }, []);

  const fetchInsights = async () => {
    try {
      const [reportRes, personalityRes, predictionRes] = await Promise.all([
        axios.get(`${API_BASE}/insights/weekly-report`),
        axios.get(`${API_BASE}/insights/personality`),
        axios.get(`${API_BASE}/insights/today-prediction`)
      ]);

      setWeeklyReport(reportRes.data);
      setPersonality(personalityRes.data);
      setTodayPrediction(predictionRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching insights:', error);
      setLoading(false);
    }
  };

  if (loading) return <div>Loading insights...</div>;

  return (
    <div className="insights-container">
      {/* Today's Prediction */}
      {todayPrediction && (
        <div className="card insight-card">
          <h2>🔮 Today's Prediction</h2>
          <div className="prediction-box">
            <div className="prediction-status">{todayPrediction.message}</div>
            <div className="prediction-stats">
              <span>Current: {todayPrediction.current_ml}ml</span>
              <span>Predicted: {todayPrediction.predicted_final_ml}ml</span>
            </div>
          </div>
        </div>
      )}

      {/* Drinking Personality */}
      {personality && (
        <div className="card personality-card">
          <h2>{personality.icon} Your Hydration Personality</h2>
          <h3>{personality.type}</h3>
          <p>{personality.description}</p>
          <div className="personality-traits">
            <div className="trait">
              <strong>Strength:</strong> {personality.strength}
            </div>
            <div className="trait">
              <strong>Improvement:</strong> {personality.improvement}
            </div>
          </div>
        </div>
      )}

      {/* Weekly Report */}
      {weeklyReport && weeklyReport.summary && (
        <div className="card weekly-report">
          <h2>📊 Weekly Report</h2>
          
          {/* Summary Stats */}
          <div className="stats-grid">
            <div className="stat-box">
              <TrendingUp size={24} />
              <span>{weeklyReport.summary.avg_daily_ml}ml</span>
              <label>Daily Average</label>
            </div>
            <div className="stat-box">
              <Target size={24} />
              <span>{weeklyReport.summary.days_hit_goal}/7</span>
              <label>Days Hit Goal</label>
            </div>
            <div className="stat-box">
              <Award size={24} />
              <span>{weeklyReport.summary.success_rate}%</span>
              <label>Success Rate</label>
            </div>
          </div>

          {/* Recommendations */}
          <div className="recommendations">
            <h3>💡 Personalized Recommendations</h3>
            {weeklyReport.recommendations?.map((rec, idx) => (
              <div key={idx} className={`recommendation ${rec.type}`}>
                <span className="rec-icon">{rec.icon}</span>
                <div>
                  <strong>{rec.title}</strong>
                  <p>{rec.message}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default PersonalizedInsights;