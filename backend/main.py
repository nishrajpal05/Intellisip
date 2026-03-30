"""
Smart Water Bottle - FastAPI Backend
Complete API with ML inference, data serving, and ESP32 integration.
"""

from datetime import datetime
from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI(
    title="HydroIQ API",
    description="Smart Water Bottle AI Backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def resolve_existing_dir(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find required directory. Checked: {searched}")


MODELS_DIR = resolve_existing_dir(BASE_DIR / "models", PROJECT_ROOT / "ml" / "models")
DATA_DIR = resolve_existing_dir(BASE_DIR / "data", PROJECT_ROOT / "data")

rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
iso_forest = joblib.load(MODELS_DIR / "isolation_forest.pkl")
feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

if hasattr(rf_model, "n_jobs"):
    rf_model.n_jobs = 1
if hasattr(iso_forest, "n_jobs"):
    iso_forest.n_jobs = 1

with open(MODELS_DIR / "metadata.json", encoding="utf-8") as file:
    metadata = json.load(file)

df = pd.read_csv(DATA_DIR / "hydration_logs.csv")
daily = pd.read_csv(DATA_DIR / "daily_summary.csv")
anomalies = pd.read_csv(DATA_DIR / "anomaly_results.csv")
comparison = pd.read_csv(DATA_DIR / "prediction_comparison.csv")
with open(DATA_DIR / "user_profile.json", encoding="utf-8") as file:
    user_profile = json.load(file)


class SipLog(BaseModel):
    user_id: str
    timestamp: str
    water_ml: float
    temperature: float


class UserProfile(BaseModel):
    user_id: str
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str
    goal: str


def build_features(
    hour,
    temperature,
    daily_total,
    time_since_last_sip,
    rolling_avg,
    bmi=22.86,
    is_weekend=0,
):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    temp_bucket = 0 if temperature < 22 else (1 if temperature < 27 else (2 if temperature < 32 else 3))
    baseline = 70 * 35 + 250
    deficit = baseline - daily_total
    intake_ratio = daily_total / baseline if baseline > 0 else 0

    return pd.DataFrame(
        [[
            hour,
            hour_sin,
            hour_cos,
            temperature,
            temp_bucket,
            time_since_last_sip,
            daily_total,
            rolling_avg,
            deficit,
            intake_ratio,
            bmi,
            is_weekend,
        ]],
        columns=feature_names,
    )


@app.get("/")
def root():
    return {"message": "HydroIQ API v2.0", "status": "active"}


@app.get("/api/user/profile")
def get_user_profile():
    return {**user_profile, "bmi": round(user_profile["weight"] / (user_profile["height"] / 100) ** 2, 2)}


@app.get("/api/overview")
def get_overview():
    total_records = len(df)
    avg_daily = round(daily["total_intake"].mean(), 1)
    max_daily = int(daily["total_intake"].max())
    min_daily = int(daily["total_intake"].min())
    baseline = 70 * 35 + 250
    personalized_target = metadata["personalized_target"]
    achievement_rate = round((daily["total_intake"] >= personalized_target * 0.85).mean() * 100, 1)

    return {
        "total_records": total_records,
        "total_days": len(daily),
        "avg_daily_intake": avg_daily,
        "max_daily_intake": max_daily,
        "min_daily_intake": min_daily,
        "baseline_intake": baseline,
        "personalized_target": personalized_target,
        "achievement_rate": achievement_rate,
        "rf_mae": metadata["rf_mae"],
        "rf_r2": metadata["rf_r2"],
        "total_anomaly_days": metadata["total_anomaly_days"],
    }


@app.get("/api/daily-intake")
def get_daily_intake(limit: int = 60):
    records = daily.tail(limit).to_dict(orient="records")
    return {"data": records, "count": len(records)}


@app.get("/api/weekly-intake")
def get_weekly_intake():
    daily_copy = daily.copy()
    daily_copy["date"] = pd.to_datetime(daily_copy["date"])
    daily_copy["week"] = daily_copy["date"].dt.isocalendar().week
    weekly = daily_copy.groupby("week").agg(
        total_intake=("total_intake", "mean"),
        avg_sips=("n_sips", "mean"),
    ).reset_index()
    return {"data": weekly.to_dict(orient="records")}


@app.get("/api/hourly-pattern")
def get_hourly_pattern():
    hourly = df.groupby("hour").agg(
        avg_intake=("water_ml", "mean"),
        total_intake=("water_ml", "sum"),
        count=("water_ml", "count"),
    ).reset_index()
    hourly["avg_intake"] = hourly["avg_intake"].round(1)
    return {"data": hourly.to_dict(orient="records")}


@app.get("/api/temp-correlation")
def get_temp_correlation():
    sample = df[["temperature", "water_ml", "daily_total", "hour"]].copy()
    sample["temp_rounded"] = sample["temperature"].round(0)
    corr = df[["temperature", "water_ml"]].corr().iloc[0, 1]
    return {
        "data": sample.sample(min(200, len(sample)), random_state=42).to_dict(orient="records"),
        "correlation": round(corr, 3),
    }


@app.get("/api/predict-next-hour")
def predict_next_hour(temperature: float = 25.0):
    current_hour = datetime.now().hour
    today_records = df[df["date"] == df["date"].max()]
    daily_total = today_records["daily_total"].max() if len(today_records) > 0 else 800
    rolling_avg = today_records["water_ml"].mean() if len(today_records) > 0 else 150
    time_gap = 1.5

    feat = build_features(
        hour=(current_hour + 1) % 24,
        temperature=temperature,
        daily_total=daily_total,
        time_since_last_sip=time_gap,
        rolling_avg=rolling_avg,
    )
    prediction = rf_model.predict(feat)[0]
    prediction = round(float(np.clip(prediction, 50, 400)), 1)

    target = metadata["personalized_target"]
    remaining = max(0, target - daily_total)
    hours_left = max(1, 22 - current_hour)
    needed_per_hour = round(remaining / hours_left, 1)

    return {
        "next_hour": (current_hour + 1) % 24,
        "predicted_ml": prediction,
        "current_daily_total": round(float(daily_total), 1),
        "personalized_target": target,
        "remaining_ml": round(float(remaining), 1),
        "needed_per_hour": needed_per_hour,
        "on_track": bool(daily_total >= (target * (current_hour / 22))),
    }


@app.get("/api/prediction-comparison")
def get_prediction_comparison(limit: int = 100):
    sample = comparison.tail(limit).copy()
    sample["error"] = (sample["predicted_ml"] - sample["actual_ml"]).abs().round(1)
    mae = round(sample["error"].mean(), 2)
    return {
        "data": sample.to_dict(orient="records"),
        "mae": mae,
        "count": len(sample),
    }


@app.get("/api/anomaly-alerts")
def get_anomaly_alerts():
    alerts = []
    anomaly_days = anomalies[anomalies["anomaly_label"] == -1].copy()

    avg_intake = metadata["user_avg_intake"]
    p25 = metadata["p25_intake"]
    target = metadata["personalized_target"]

    for _, row in anomaly_days.iterrows():
        severity = "warning"
        reasons = []
        deficit_pct = round((1 - row["total_intake"] / avg_intake) * 100, 1)

        if row["total_intake"] < p25:
            reasons.append(f"Intake was {abs(deficit_pct):.0f}% below your average")
            severity = "critical"
        elif row["max_gap"] > 4:
            reasons.append(f"Longest gap between sips: {row['max_gap']:.1f} hours")
        elif row["total_intake"] > avg_intake * 1.4:
            reasons.append(f"Unusually high intake: {row['total_intake']:.0f}ml (spike detected)")
            severity = "info"
        elif row["n_sips"] < 5:
            reasons.append(f"Only {row['n_sips']} sips recorded (very low activity)")
        else:
            reasons.append("Unusual hydration pattern detected")

        alerts.append(
            {
                "date": row["date"],
                "severity": severity,
                "total_intake": round(row["total_intake"], 1),
                "avg_sips": round(float(row["n_sips"]), 1),
                "max_gap_hours": round(row["max_gap"], 1),
                "anomaly_score": round(float(row["anomaly_score"]), 3),
                "reasons": reasons,
                "deficit_pct": deficit_pct,
            }
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda item: severity_order.get(item["severity"], 3))
    gap_count = len(df[df["time_since_last_sip"] > 3])

    return {
        "alerts": alerts,
        "total_alerts": len(alerts),
        "critical_count": sum(1 for alert in alerts if alert["severity"] == "critical"),
        "warning_count": sum(1 for alert in alerts if alert["severity"] == "warning"),
        "long_gap_incidents": gap_count,
        "avg_intake": avg_intake,
        "personalized_target": target,
    }


@app.get("/api/recommendation")
def get_recommendation():
    baseline = 70 * 35 + 250
    user_avg = metadata["user_avg_intake"]
    target = metadata["personalized_target"]

    tips = []
    if user_avg < baseline * 0.8:
        tips.append("You're consistently drinking below baseline. Start with small increases - add 200ml per day.")
    if metadata["total_anomaly_days"] > 5:
        tips.append("Multiple anomalous days detected. Try setting hourly reminders.")
    tips.append("Drink 200ml immediately after waking up to jumpstart hydration.")
    tips.append("Link drinking water to existing habits - coffee, meals, bathroom breaks.")

    hourly = df.groupby("hour")["water_ml"].mean()
    weakest_hour = hourly.idxmin()
    tips.append(f"Your hydration dips most at {weakest_hour}:00 - set a reminder for that time.")

    return {
        "baseline_intake": baseline,
        "user_avg_intake": round(user_avg, 1),
        "personalized_target": target,
        "gap": round(target - user_avg, 1),
        "tips": tips,
        "model_accuracy": f"{round(metadata['rf_r2'] * 100, 1)}%",
        "ml_mae": f"+/-{metadata['rf_mae']}ml",
    }


@app.post("/api/ingest-sip")
def ingest_sip(sip: SipLog):
    return {
        "status": "received",
        "user_id": sip.user_id,
        "water_ml": sip.water_ml,
        "temperature": sip.temperature,
        "timestamp": sip.timestamp,
        "message": "Sip logged. In production, this would write to MongoDB.",
    }


@app.get("/api/ml-stats")
def get_ml_stats():
    return {
        "linear_regression": {
            "mae": metadata["lr_mae"],
            "r2": metadata["lr_r2"],
        },
        "random_forest": {
            "mae": metadata["rf_mae"],
            "r2": metadata["rf_r2"],
        },
        "feature_importance": metadata["feature_importance"],
        "training_samples": metadata["training_samples"],
        "test_samples": metadata["test_samples"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
