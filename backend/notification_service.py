"""
Smart Notification Service
Generates intelligent hydration reminders based on user behavior
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

class NotificationService:
    
    def __init__(self, db_collection):
        self.collection = db_collection
        
    def check_hydration_status(self, user_id: str = "user1") -> Optional[Dict]:
        """
        Check if user needs a reminder based on last sip time
        Returns notification data or None
        """
        # Get last sip
        last_sip = self.collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )
        
        if not last_sip:
            return None
        
        # Calculate time since last sip
        now = datetime.now()
        last_time = last_sip["timestamp"]
        time_diff = now - last_time
        minutes_passed = time_diff.total_seconds() / 60
        
        # Get today's total
        today = now.date().isoformat()
        today_data = list(self.collection.find({
            "user_id": user_id,
            "day": today
        }))
        
        if today_data:
            today_total = max(d.get("daily_total", 0) for d in today_data)
        else:
            today_total = 0
        
        # CRITICAL: 3+ hours without water
        if minutes_passed >= 180:
            return {
                "level": "critical",
                "title": "🚨 URGENT: Dehydration Risk",
                "message": f"You haven't drunk water in {int(minutes_passed/60)} hours! Drink 300ml immediately.",
                "action": "drink_now",
                "recommended_ml": 300,
                "minutes_since_last": int(minutes_passed)
            }
        
        # WARNING: 2+ hours
        elif minutes_passed >= 120:
            return {
                "level": "warning",
                "title": "⚠️ Hydration Reminder",
                "message": f"It's been {int(minutes_passed/60)} hours since your last sip. Time to hydrate!",
                "action": "drink_soon",
                "recommended_ml": 200,
                "minutes_since_last": int(minutes_passed)
            }
        
        # INFO: 1+ hour
        elif minutes_passed >= 60:
            return {
                "level": "info",
                "title": "💧 Time for Water",
                "message": "Stay consistent! Have a sip now.",
                "action": "gentle_reminder",
                "recommended_ml": 150,
                "minutes_since_last": int(minutes_passed)
            }
        
        # Goal-based reminder (after 6 PM, check if behind)
        elif now.hour >= 18 and today_total < 1500:
            remaining = 2000 - today_total
            return {
                "level": "goal",
                "title": "🎯 Goal Reminder",
                "message": f"You need {remaining}ml more to hit today's goal. Keep going!",
                "action": "goal_push",
                "recommended_ml": min(remaining, 300),
                "today_total": today_total
            }
        
        return None
    
    def get_morning_reminder(self, user_id: str = "user1") -> Optional[Dict]:
        """
        Morning hydration boost reminder
        """
        now = datetime.now()
        
        # Only between 6 AM - 9 AM
        if 6 <= now.hour < 9:
            # Check if already drank this morning
            morning_sips = self.collection.count_documents({
                "user_id": user_id,
                "timestamp": {
                    "$gte": now.replace(hour=6, minute=0, second=0),
                    "$lte": now
                }
            })
            
            if morning_sips == 0:
                return {
                    "level": "routine",
                    "title": "☀️ Good Morning!",
                    "message": "Start your day right! Drink 250ml within 30 minutes of waking.",
                    "action": "morning_boost",
                    "recommended_ml": 250,
                    "tip": "Morning hydration boosts metabolism by 24%!"
                }
        
        return None
    
    def get_weather_based_reminder(self, temperature: float) -> Optional[Dict]:
        """
        Weather-based hydration recommendations
        """
        if temperature >= 35:
            return {
                "level": "weather",
                "title": "🌡️ Extreme Heat Alert!",
                "message": f"It's {temperature}°C! Increase intake by 40% today. Target: 2800ml",
                "action": "weather_adjust",
                "recommended_goal": 2800,
                "tip": "Drink BEFORE you feel thirsty in extreme heat."
            }
        elif temperature >= 30:
            return {
                "level": "weather",
                "title": "☀️ Hot Day",
                "message": f"Temperature: {temperature}°C. Increase intake by 25%. Target: 2500ml",
                "action": "weather_adjust",
                "recommended_goal": 2500,
                "tip": "Hot weather increases water loss through sweating."
            }
        elif temperature <= 15:
            return {
                "level": "weather",
                "title": "❄️ Cold Weather Reminder",
                "message": "Don't forget to hydrate even in cold weather!",
                "action": "cold_reminder",
                "recommended_goal": 1900,
                "tip": "People often forget to drink in cold weather."
            }
        
        return None
    
    def get_bedtime_reminder(self, user_id: str = "user1") -> Optional[Dict]:
        """
        Before-bed hydration check
        """
        now = datetime.now()
        
        # Between 9 PM - 11 PM
        if 21 <= now.hour < 23:
            # Check last sip
            last_sip = self.collection.find_one(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            )
            
            if last_sip:
                time_diff = now - last_sip["timestamp"]
                minutes = time_diff.total_seconds() / 60
                
                if minutes >= 120:  # 2+ hours
                    return {
                        "level": "routine",
                        "title": "🌙 Bedtime Hydration",
                        "message": "Have 150ml before sleep. Not too much to avoid bathroom trips!",
                        "action": "bedtime",
                        "recommended_ml": 150,
                        "tip": "Optimal for overnight hydration without disrupting sleep."
                    }
        
        return None
    
    def get_all_pending_notifications(self, user_id: str = "user1", temperature: float = 25) -> List[Dict]:
        """
        Get all applicable notifications
        """
        notifications = []
        
        # Check main hydration status
        status_notif = self.check_hydration_status(user_id)
        if status_notif:
            notifications.append(status_notif)
        
        # Morning reminder
        morning_notif = self.get_morning_reminder(user_id)
        if morning_notif:
            notifications.append(morning_notif)
        
        # Weather-based
        weather_notif = self.get_weather_based_reminder(temperature)
        if weather_notif:
            notifications.append(weather_notif)
        
        # Bedtime reminder
        bedtime_notif = self.get_bedtime_reminder(user_id)
        if bedtime_notif:
            notifications.append(bedtime_notif)
        
        return notifications