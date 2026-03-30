import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class MLInsightsEngine:
    
    def __init__(self, db_collection):
        self.collection = db_collection
    
    def get_user_data(self, user_id: str = "user1", days: int = 30) -> pd.DataFrame:
        """Fetch user data for analysis"""
        data = list(self.collection.find({"user_id": user_id}))
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    def generate_weekly_report(self, user_id: str = "user1") -> Dict:
        """
        Generate comprehensive weekly hydration report
        """
        # Get last 7 days of data
        df = self.get_user_data(user_id, days=7)
        
        if df.empty:
            return {"error": "No data available"}
        
        # Filter last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        df_week = df[df['timestamp'] >= start_date]
        
        # Calculate daily totals
        daily_totals = df_week.groupby(df_week['timestamp'].dt.date)['daily_total'].max()
        
        # Metrics
        avg_daily = daily_totals.mean()
        goal = 2000
        days_hit_goal = (daily_totals >= goal).sum()
        best_day = daily_totals.idxmax()
        worst_day = daily_totals.idxmin()
        total_sips = len(df_week)
        
        # Calculate improvement vs previous week
        prev_week_start = start_date - timedelta(days=7)
        df_prev = df[(df['timestamp'] >= prev_week_start) & (df['timestamp'] < start_date)]
        prev_avg = df_prev.groupby(df_prev['timestamp'].dt.date)['daily_total'].max().mean()
        
        if prev_avg > 0:
            improvement = ((avg_daily - prev_avg) / prev_avg) * 100
        else:
            improvement = 0
        
        # Find patterns
        hourly_avg = df_week.groupby('hour')['water_ml'].mean()
        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()
        
        # Day of week analysis
        df_week['day_of_week'] = df_week['timestamp'].dt.dayofweek
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = df_week.groupby('day_of_week')['daily_total'].max()
        best_weekday = day_names[weekly_pattern.idxmax()]
        worst_weekday = day_names[weekly_pattern.idxmin()]
        
        return {
            "period": f"{start_date.date()} to {end_date.date()}",
            "summary": {
                "avg_daily_ml": round(avg_daily, 0),
                "days_hit_goal": int(days_hit_goal),
                "total_days": 7,
                "success_rate": round((days_hit_goal / 7) * 100, 1),
                "total_sips": int(total_sips),
                "avg_sips_per_day": round(total_sips / 7, 1)
            },
            "performance": {
                "best_day": {
                    "date": str(best_day),
                    "amount": round(daily_totals[best_day], 0)
                },
                "worst_day": {
                    "date": str(worst_day),
                    "amount": round(daily_totals[worst_day], 0)
                },
                "improvement_vs_last_week": round(improvement, 1)
            },
            "patterns": {
                "peak_drinking_hour": int(peak_hour),
                "lowest_drinking_hour": int(low_hour),
                "best_weekday": best_weekday,
                "worst_weekday": worst_weekday
            },
            "recommendations": self._generate_recommendations(df_week, avg_daily, peak_hour, worst_weekday)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, avg_daily: float, peak_hour: int, worst_day: str) -> List[Dict]:
        """
        Generate personalized recommendations based on patterns
        """
        recommendations = []
        goal = 2000
        
        # Recommendation 1: Overall performance
        if avg_daily >= goal:
            recommendations.append({
                "type": "praise",
                "icon": "🎉",
                "title": "Excellent Hydration!",
                "message": f"You're averaging {int(avg_daily)}ml/day. Keep it up!"
            })
        elif avg_daily >= goal * 0.8:
            recommendations.append({
                "type": "encouragement",
                "icon": "🎯",
                "title": "Almost There!",
                "message": f"You're at {int(avg_daily)}ml/day. Just {int(goal - avg_daily)}ml more to hit your goal consistently."
            })
        else:
            recommendations.append({
                "type": "improvement",
                "icon": "⚠️",
                "title": "Needs Improvement",
                "message": f"Current average: {int(avg_daily)}ml/day. Target: {goal}ml. Let's work on this!"
            })
        
        # Recommendation 2: Time distribution
        morning_intake = df[df['hour'] < 12]['water_ml'].sum()
        afternoon_intake = df[(df['hour'] >= 12) & (df['hour'] < 18)]['water_ml'].sum()
        evening_intake = df[df['hour'] >= 18]['water_ml'].sum()
        
        total_intake = morning_intake + afternoon_intake + evening_intake
        
        if total_intake > 0:
            morning_pct = (morning_intake / total_intake) * 100
            
            if morning_pct < 20:
                recommendations.append({
                    "type": "habit",
                    "icon": "☀️",
                    "title": "Boost Morning Hydration",
                    "message": f"Only {int(morning_pct)}% of your water is before noon. Try drinking 500ml before 12 PM for better energy!"
                })
        
        # Recommendation 3: Consistency
        daily_totals = df.groupby(df['timestamp'].dt.date)['daily_total'].max()
        std_dev = daily_totals.std()
        
        if std_dev > 500:
            recommendations.append({
                "type": "consistency",
                "icon": "📊",
                "title": "Work on Consistency",
                "message": "Your daily intake varies a lot. Try setting hourly reminders for more stable hydration."
            })
        
        # Recommendation 4: Specific day improvement
        recommendations.append({
            "type": "weekday",
            "icon": "📅",
            "title": f"Focus on {worst_day}s",
            "message": f"{worst_day} is your lowest hydration day. Set extra reminders for that day!"
        })
        
        # Recommendation 5: Temperature awareness
        if 'temperature' in df.columns:
            high_temp_days = df[df['temperature'] > 30]
            if len(high_temp_days) > 0:
                avg_hot = high_temp_days.groupby(high_temp_days['timestamp'].dt.date)['daily_total'].max().mean()
                if avg_hot < goal * 1.2:
                    recommendations.append({
                        "type": "weather",
                        "icon": "🌡️",
                        "title": "Hot Weather Adjustment",
                        "message": f"On hot days (>30°C), you should drink 20-40% more. Currently averaging {int(avg_hot)}ml on hot days."
                    })
        
        return recommendations
    
    def get_drinking_personality(self, user_id: str = "user1") -> Dict:
        """
        Analyze user's drinking personality/style
        """
        df = self.get_user_data(user_id, days=30)
        
        if df.empty:
            return {"error": "Insufficient data"}
        
        # Calculate patterns
        avg_sip_size = df['water_ml'].mean()
        sips_per_day = len(df) / 30
        hourly_pattern = df.groupby('hour')['water_ml'].sum()
        peak_hours = hourly_pattern.nlargest(3).index.tolist()
        
        # Determine personality
        if avg_sip_size > 250 and sips_per_day < 8:
            personality = {
                "type": "Gulper",
                "icon": "🥤",
                "description": "You prefer large drinks, fewer times a day",
                "strength": "Efficient - you don't forget when you do drink",
                "improvement": "Try smaller, more frequent sips for better absorption"
            }
        elif avg_sip_size < 150 and sips_per_day > 12:
            personality = {
                "type": "Sipper",
                "icon": "☕",
                "description": "You take small, frequent sips throughout the day",
                "strength": "Great for consistent hydration",
                "improvement": "You're doing great! Keep it up!"
            }
        elif len(peak_hours) > 0 and all(h >= 12 and h < 18 for h in peak_hours):
            personality = {
                "type": "Afternoon Warrior",
                "icon": "🌞",
                "description": "Most of your hydration happens in afternoon",
                "strength": "You stay hydrated during peak activity hours",
                "improvement": "Spread intake to morning and evening too"
            }
        else:
            personality = {
                "type": "Balanced Hydrator",
                "icon": "⚖️",
                "description": "You have a well-distributed drinking pattern",
                "strength": "Excellent balance throughout the day!",
                "improvement": "Maintain this great habit!"
            }
        
        personality["stats"] = {
            "avg_sip_ml": round(avg_sip_size, 0),
            "sips_per_day": round(sips_per_day, 1),
            "peak_hours": [f"{h}:00" for h in peak_hours]
        }
        
        return personality
    
    def predict_today_completion(self, user_id: str = "user1") -> Dict:
        """
        Predict if user will hit goal today based on current progress
        """
        now = datetime.now()
        today = now.date().isoformat()
        
        # Get today's data
        today_data = list(self.collection.find({
            "user_id": user_id,
            "day": today
        }))
        
        if not today_data:
            return {
                "prediction": "uncertain",
                "message": "No data yet today. Start drinking!"
            }
        
        current_total = max(d.get("daily_total", 0) for d in today_data)
        hours_left = 24 - now.hour
        goal = 2000
        remaining = goal - current_total
        
        # Get historical hourly average for remaining hours
        df = self.get_user_data(user_id, days=14)
        future_hours = range(now.hour + 1, 24)
        historical_future = df[df['hour'].isin(future_hours)].groupby('hour')['water_ml'].mean().sum()
        
        predicted_final = current_total + historical_future
        
        if predicted_final >= goal:
            confidence = min(95, int((predicted_final / goal) * 100))
            return {
                "prediction": "likely",
                "confidence": confidence,
                "current_ml": int(current_total),
                "predicted_final_ml": int(predicted_final),
                "message": f"{confidence}% chance you'll hit your goal today! On track! 🎯"
            }
        else:
            shortfall = goal - predicted_final
            return {
                "prediction": "unlikely",
                "confidence": 70,
                "current_ml": int(current_total),
                "predicted_final_ml": int(predicted_final),
                "shortfall_ml": int(shortfall),
                "message": f"You might fall short by {int(shortfall)}ml. Drink {int(remaining/hours_left)}ml per hour to stay on track!"
            }