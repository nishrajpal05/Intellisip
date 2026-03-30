from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Smart Water Bottle API",
    description="AI-powered hydration monitoring system",
    version="1.0.0"
)

# Enable CORS (so frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MONGODB CONNECTION
# ============================================
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME", "water_bottle_db")

try:
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE_NAME]
    hydration_collection = db["hydration_data"]
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# ============================================
# LOAD ML MODELS
# ============================================
try:
    # Load models from the models folder
    prediction_model = joblib.load("../models/prediction_model.pkl")
    anomaly_model = joblib.load("../models/anomaly_model.pkl")
    print("✅ ML models loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("Make sure prediction_model.pkl and anomaly_model.pkl are in ../models/")
    prediction_model = None
    anomaly_model = None

# ============================================
# INITIALIZE SMART SERVICES
# ============================================
from notification_service import NotificationService
from ml_insights import MLInsightsEngine

notification_service = None
ml_insights = None

if db is not None:
    notification_service = NotificationService(hydration_collection)
    ml_insights = MLInsightsEngine(hydration_collection)
    print("✅ Smart services initialized!")

# ============================================
# PYDANTIC MODELS (DATA SCHEMAS)
# ============================================
class SipData(BaseModel):
    """Data structure for a single sip from ESP32"""
    user_id: str = "user1"
    timestamp: str  # ISO format: "2025-01-15T10:30:00"
    water_ml: float
    temperature: float

class DailyIntake(BaseModel):
    """Daily hydration summary"""
    date: str
    total_ml: float
    sip_count: int
    avg_temp: float

class PredictionRequest(BaseModel):
    """Request for next intake prediction"""
    user_id: str = "user1"
    hour: int
    temperature: float
    past_hour_intake: float
    avg_intake_so_far: float
    day_of_week: int
    is_weekend: int
    time_diff: float

class AnomalyAlert(BaseModel):
    """Anomaly detection result"""
    timestamp: str
    water_ml: float
    anomaly_type: str
    severity: str
    message: str

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    db_status = "disconnected"
    db_error = None
    
    if db is not None:
        try:
            # Test connection by pinging
            client.admin.command('ping')
            db_status = "connected"
        except Exception as e:
            db_status = "error"
            db_error = str(e)
    
    return {
        "status": "running",
        "service": "Smart Water Bottle API",
        "models_loaded": prediction_model is not None and anomaly_model is not None,
        "database_status": db_status,
        "database_error": db_error
    }

@app.post("/api/ingest-sip")
async def ingest_sip(sip: SipData):
    """
    Receive sip data from ESP32 and store in MongoDB
    This is what ESP32 will call when water is consumed
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # Parse timestamp
        dt = datetime.fromisoformat(sip.timestamp)
        
        # Calculate additional fields
        day = dt.date().isoformat()
        hour = dt.hour
        
        # Get daily total so far
        today_data = list(hydration_collection.find({
            "user_id": sip.user_id,
            "day": day
        }))
        daily_total = sum(d["water_ml"] for d in today_data) + sip.water_ml
        
        # Create document
        document = {
            "user_id": sip.user_id,
            "timestamp": dt,
            "hour": hour,
            "water_ml": sip.water_ml,
            "temperature": sip.temperature,
            "day": day,
            "daily_total": daily_total,
            "created_at": datetime.now()
        }
        
        # Insert into MongoDB
        result = hydration_collection.insert_one(document)
        
        return {
            "status": "success",
            "message": "Sip data recorded",
            "id": str(result.inserted_id),
            "daily_total": daily_total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing data: {str(e)}")

@app.get("/api/daily-intake", response_model=List[DailyIntake])
async def get_daily_intake(user_id: str = "user1", days: int = 7):
    """
    Get daily hydration summary for the past N days
    Used by frontend for daily intake chart
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # Aggregate daily totals
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$day",
                "total_ml": {"$max": "$daily_total"},
                "sip_count": {"$sum": 1},
                "avg_temp": {"$avg": "$temperature"}
            }},
            {"$sort": {"_id": -1}},
            {"$limit": days}
        ]
        
        results = list(hydration_collection.aggregate(pipeline))
        
        # Format response - convert day number to string
        daily_data = [
            DailyIntake(
                date=f"Day {r['_id']}" if isinstance(r['_id'], int) else str(r['_id']),
                total_ml=round(r["total_ml"], 2),
                sip_count=r["sip_count"],
                avg_temp=round(r["avg_temp"], 2)
            )
            for r in results
        ]
        
        return daily_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.post("/api/predict-next-intake")
async def predict_next_intake(request: PredictionRequest):
    """
    Predict next hour's water intake using ML model
    Frontend calls this to show predictions
    """
    if prediction_model is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded")
    
    try:
        # Create feature vector
        features = pd.DataFrame([{
            'hour': request.hour,
            'temperature': request.temperature,
            'past_hour_intake': request.past_hour_intake,
            'avg_intake_so_far': request.avg_intake_so_far,
            'day_of_week': request.day_of_week,
            'is_weekend': request.is_weekend,
            'time_diff': request.time_diff
        }])
        
        # Make prediction
        prediction = prediction_model.predict(features)[0]
        
        return {
            "predicted_ml": round(prediction, 2),
            "confidence": "high" if prediction > 100 else "medium",
            "message": f"You should drink approximately {int(prediction)}ml in the next hour"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/anomaly-alerts", response_model=List[AnomalyAlert])
async def get_anomaly_alerts(user_id: str = "user1", days: int = 7):
    """
    Detect anomalies in recent hydration data
    Frontend displays these as health alerts
    """
    if anomaly_model is None or db is None:
        raise HTTPException(status_code=500, detail="Anomaly model or database not available")
    
    try:
        # Get recent data
        recent_data = list(hydration_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(days * 15))
        
        if not recent_data:
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_data)
        
        # Calculate time differences
        df = df.sort_values('timestamp')
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60
        df['time_diff'] = df['time_diff'].fillna(0)
        
        # Prepare features for anomaly detection
        anomaly_features = df[['water_ml', 'time_diff', 'temperature', 'hour', 'daily_total']]
        
        # Detect anomalies
        predictions = anomaly_model.predict(anomaly_features)
        df['is_anomaly'] = predictions
        
        # Generate alerts for anomalies
        alerts = []
        anomalies = df[df['is_anomaly'] == -1]
        
        for _, row in anomalies.iterrows():
            # Determine anomaly type and severity
            if row['water_ml'] < 100:
                anomaly_type = "Low intake"
                severity = "warning"
                message = f"Small sip detected ({int(row['water_ml'])}ml). Consider drinking more."
            elif row['time_diff'] > 180:  # >3 hours
                anomaly_type = "Long gap"
                severity = "critical"
                message = f"No water for {int(row['time_diff']/60)} hours. Stay hydrated!"
            elif row['water_ml'] > 400:
                anomaly_type = "Unusual spike"
                severity = "info"
                message = f"Large intake detected ({int(row['water_ml'])}ml). Everything okay?"
            else:
                anomaly_type = "Unusual pattern"
                severity = "info"
                message = "Drinking pattern differs from your normal behavior."
            
            alerts.append(AnomalyAlert(
                timestamp=row['timestamp'].isoformat(),
                water_ml=row['water_ml'],
                anomaly_type=anomaly_type,
                severity=severity,
                message=message
            ))
        
        return alerts[:10]  # Return top 10 recent alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")

@app.get("/api/hourly-breakdown")
async def get_hourly_breakdown(user_id: str = "user1", day: Optional[int] = None):
    """
    Get hourly breakdown of water intake for a specific day
    Used for time-series visualization
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # Use most recent day if no day specified
        if day is None:
            # Get the latest day number from database
            latest = hydration_collection.find_one(
                {"user_id": user_id},
                sort=[("day", -1)]
            )
            day = latest["day"] if latest else 59
        
        # Aggregate by hour
        pipeline = [
            {"$match": {"user_id": user_id, "day": day}},
            {"$group": {
                "_id": "$hour",
                "total_ml": {"$sum": "$water_ml"},
                "sip_count": {"$sum": 1},
                "avg_temp": {"$avg": "$temperature"}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        results = list(hydration_collection.aggregate(pipeline))
        
        # Format response
        hourly_data = [
            {
                "hour": r["_id"],
                "total_ml": round(r["total_ml"], 2),
                "sip_count": r["sip_count"],
                "avg_temp": round(r["avg_temp"], 2)
            }
            for r in results
        ]
        
        return hourly_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching hourly data: {str(e)}")

@app.post("/api/load-synthetic-data")
async def load_synthetic_data():
    """
    Load synthetic data from CSV into MongoDB
    Run this once to populate database with test data
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    try:
        # Load CSV
        df = pd.read_csv("../models/synthetic_hydration_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert to documents
        documents = df.to_dict('records')
        
        # Insert into MongoDB
        hydration_collection.delete_many({})  # Clear existing data
        result = hydration_collection.insert_many(documents)
        
        return {
            "status": "success",
            "message": f"Loaded {len(result.inserted_ids)} records into database",
            "records": len(result.inserted_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

# ============================================
# NEW ENDPOINTS - SMART REMINDERS
# ============================================

@app.get("/api/notifications/check")
async def check_notifications(user_id: str = "user1", temperature: float = 25):
    """
    Check for pending notifications/reminders
    Frontend calls this periodically
    """
    if notification_service is None:
        raise HTTPException(status_code=500, detail="Notification service not available")
    
    try:
        notifications = notification_service.get_all_pending_notifications(user_id, temperature)
        return {
            "count": len(notifications),
            "notifications": notifications
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking notifications: {str(e)}")

@app.get("/api/notifications/hydration-status")
async def get_hydration_status(user_id: str = "user1"):
    """
    Get current hydration status and any warnings
    """
    if notification_service is None:
        raise HTTPException(status_code=500, detail="Notification service not available")
    
    try:
        status = notification_service.check_hydration_status(user_id)
        return status if status else {"status": "good", "message": "Hydration on track!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================
# NEW ENDPOINTS - ML PERSONALIZATION
# ============================================

@app.get("/api/insights/weekly-report")
async def get_weekly_report(user_id: str = "user1"):
    """
    Generate comprehensive weekly hydration report
    """
    if ml_insights is None:
        raise HTTPException(status_code=500, detail="ML insights service not available")
    
    try:
        report = ml_insights.generate_weekly_report(user_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/api/insights/personality")
async def get_drinking_personality(user_id: str = "user1"):
    """
    Analyze user's drinking personality/habits
    """
    if ml_insights is None:
        raise HTTPException(status_code=500, detail="ML insights service not available")
    
    try:
        personality = ml_insights.get_drinking_personality(user_id)
        return personality
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/insights/today-prediction")
async def predict_today(user_id: str = "user1"):
    """
    Predict if user will hit goal today
    """
    if ml_insights is None:
        raise HTTPException(status_code=500, detail="ML insights service not available")
    
    try:
        prediction = ml_insights.predict_today_completion(user_id)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n Starting server on http://localhost:{port}")
    print(f"API docs available at http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)