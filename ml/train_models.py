"""
Smart Water Bottle - ML Training Pipeline
- Regression: Predict next sip volume
- Anomaly Detection: Isolation Forest
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
df = pd.read_csv("hydration_logs.csv")
daily = pd.read_csv("daily_summary.csv")

print(f"📦 Loaded {len(df)} records")

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["temp_bucket"] = pd.cut(df["temperature"], bins=[0, 22, 27, 32, 40], labels=[0, 1, 2, 3]).astype(int)
df["hydration_deficit"] = df["baseline_intake"] - df["daily_total"]
df["intake_ratio"] = df["daily_total"] / df["baseline_intake"]

# ─── REGRESSION FEATURES ─────────────────────────────────────────────────────
FEATURES = [
    "hour", "hour_sin", "hour_cos",
    "temperature", "temp_bucket",
    "time_since_last_sip",
    "daily_total", "rolling_avg_intake",
    "hydration_deficit", "intake_ratio",
    "bmi", "is_weekend"
]
TARGET = "water_ml"

df_clean = df.dropna(subset=FEATURES + [TARGET])
X = df_clean[FEATURES]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── LINEAR REGRESSION (Baseline) ────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"\n📈 Linear Regression:")
print(f"   MAE: {lr_mae:.2f} ml | R²: {lr_r2:.3f}")

# ─── RANDOM FOREST (Improved) ─────────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"\n🌲 Random Forest:")
print(f"   MAE: {rf_mae:.2f} ml | R²: {rf_r2:.3f}")

# Feature importance
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n🔍 Top Features:")
for feat, imp in fi.head(5).items():
    print(f"   {feat}: {imp:.3f}")

# ─── ANOMALY DETECTION ────────────────────────────────────────────────────────
# Daily-level anomaly
daily_features = ["total_intake", "avg_sip_size", "n_sips", "avg_temp", "max_gap"]
daily_clean = daily.dropna(subset=daily_features)
X_daily = daily_clean[daily_features].values

iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_daily)
anomaly_labels = iso_forest.predict(X_daily)
anomaly_scores = iso_forest.score_samples(X_daily)

n_anomalies = (anomaly_labels == -1).sum()
print(f"\n🚨 Anomaly Detection:")
print(f"   Total days: {len(daily_clean)}")
print(f"   Anomalous days detected: {n_anomalies}")

# ─── COMPUTE STATS FOR PERSONALIZATION ENGINE ─────────────────────────────────
user_avg_intake = daily["total_intake"].mean()
user_std_intake = daily["total_intake"].std()
baseline = 70 * 35 + 250  # moderate activity

# Personalized target
if user_avg_intake < baseline:
    gradual_increment = (baseline - user_avg_intake) * 0.15
    personalized_target = user_avg_intake + gradual_increment
else:
    personalized_target = baseline

# Alert thresholds
p25 = daily["total_intake"].quantile(0.25)
p75 = daily["total_intake"].quantile(0.75)

# Compute predictions vs actual for test set
test_df = df_clean.iloc[X_test.index] if hasattr(X_test, 'index') else df_clean.tail(len(X_test))
comparison_records = []
for i, (idx, row) in enumerate(df_clean.iterrows()):
    feat = row[FEATURES].values.reshape(1, -1)
    pred_rf = rf.predict(feat)[0]
    comparison_records.append({
        "timestamp": row["timestamp"],
        "date": row["date"],
        "hour": row["hour"],
        "actual_ml": row["water_ml"],
        "predicted_ml": round(pred_rf, 1),
        "temperature": row["temperature"],
        "daily_total": row["daily_total"]
    })

comparison_df = pd.DataFrame(comparison_records)
comparison_df.to_csv("prediction_comparison.csv", index=False)

# ─── SAVE MODELS ─────────────────────────────────────────────────────────────
joblib.dump(lr, "models/linear_regression.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(iso_forest, "models/isolation_forest.pkl")
joblib.dump(FEATURES, "models/feature_names.pkl")

# Save anomaly results with dates
daily_clean = daily_clean.copy()
daily_clean["anomaly_label"] = anomaly_labels
daily_clean["anomaly_score"] = anomaly_scores
daily_clean.to_csv("anomaly_results.csv", index=False)

# ─── SAVE METADATA ───────────────────────────────────────────────────────────
metadata = {
    "lr_mae": round(lr_mae, 2),
    "lr_r2": round(lr_r2, 3),
    "rf_mae": round(rf_mae, 2),
    "rf_r2": round(rf_r2, 3),
    "user_avg_intake": round(user_avg_intake, 1),
    "user_std_intake": round(user_std_intake, 1),
    "baseline_intake": baseline,
    "personalized_target": round(personalized_target, 1),
    "p25_intake": round(p25, 1),
    "p75_intake": round(p75, 1),
    "total_anomaly_days": int(n_anomalies),
    "feature_importance": {k: round(v, 4) for k, v in fi.items()},
    "training_samples": len(X_train),
    "test_samples": len(X_test)
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Models saved!")
print(f"📊 User avg intake: {user_avg_intake:.0f} ml/day")
print(f"🎯 Personalized target: {personalized_target:.0f} ml/day")
print(f"📦 Files: models/random_forest.pkl, models/scaler.pkl, models/isolation_forest.pkl")
