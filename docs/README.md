# HydroIQ — AI Smart Water Bottle Platform
## Phase 2: Complete ML + Backend + Frontend + IoT

```
┌─────────────────────────────────────────────────────────┐
│  ESP32 Sensor  →  FastAPI Backend  →  ML Models         │
│       ↓                ↓                  ↓             │
│  Water Level    MongoDB/CSV DB    Prediction + Anomaly   │
│  Temperature    REST API          Personalization        │
│       ↓                ↓                  ↓             │
│  POST /ingest   GET /predict      React Dashboard        │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
HydroIQ/
├── data/
│   ├── generate_data.py        ← Synthetic data generator (60 days)
│   ├── hydration_logs.csv      ← 672 sip records
│   ├── daily_summary.csv       ← 60 daily aggregates
│   ├── anomaly_results.csv     ← Isolation Forest labels
│   ├── prediction_comparison.csv
│   └── user_profile.json       ← User U01 profile
│
├── ml/
│   ├── train_models.py         ← Full ML training pipeline
│   └── models/
│       ├── random_forest.pkl   ← Regression model (MAE ±23.7ml)
│       ├── linear_regression.pkl
│       ├── isolation_forest.pkl ← Anomaly detection
│       ├── scaler.pkl          ← Feature scaler
│       ├── feature_names.pkl
│       └── metadata.json       ← Model metrics + user stats
│
├── backend/
│   ├── main.py                 ← FastAPI application (12 endpoints)
│   └── requirements.txt
│
├── frontend/
│   └── index.html              ← Complete 6-page dashboard
│
└── docs/
    ├── README.md               ← This file
    └── esp32_firmware.ino      ← Complete Arduino sketch
```

---

## 🚀 Quick Start

### 1. Install Python dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Generate data (skip if using provided CSVs)
```bash
cd data
python generate_data.py
```

### 3. Train ML models (skip if using provided .pkl files)
```bash
cd ml
python train_models.py
```

### 4. Run backend
```bash
# Copy ML models and CSV files to backend folder first
cp -r ml/models backend/
cp data/*.csv backend/
cp data/*.json backend/

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open dashboard
```bash
# Just open frontend/index.html in any browser
# No build step needed — it's a single HTML file with all data embedded
open frontend/index.html
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/user/profile` | User profile + BMI |
| GET | `/api/overview` | Dashboard summary stats |
| GET | `/api/daily-intake` | 60-day daily intake data |
| GET | `/api/weekly-intake` | Weekly averages |
| GET | `/api/hourly-pattern` | Hour-by-hour drinking pattern |
| GET | `/api/temp-correlation` | Temperature vs intake scatter |
| GET | `/api/predict-next-hour` | ML prediction for next sip |
| GET | `/api/prediction-comparison` | Predicted vs actual records |
| GET | `/api/anomaly-alerts` | Health alerts from Isolation Forest |
| GET | `/api/recommendation` | Personalized hydration tips |
| GET | `/api/ml-stats` | Model performance metrics |
| POST | `/api/ingest-sip` | ESP32 sip data ingestion |

### Example POST payload (ESP32):
```json
{
  "user_id": "U01",
  "timestamp": "2025-03-01 14:30:00",
  "water_ml": 150,
  "temperature": 27.5
}
```

---

## 🤖 ML Models

### Regression (Predict Next Sip Volume)
| Model | MAE | R² |
|-------|-----|-----|
| Linear Regression | ±24.4 ml | 0.353 |
| **Random Forest** | **±23.7 ml** | **0.400** |

**12 Features used:**
- `hour`, `hour_sin`, `hour_cos` — time of day (circular encoding)
- `temperature`, `temp_bucket` — ambient temperature
- `time_since_last_sip` — hydration gap
- `daily_total`, `rolling_avg_intake` — cumulative behavior
- `hydration_deficit`, `intake_ratio` — progress vs baseline
- `bmi`, `is_weekend` — user profile features

**Top Feature:** `rolling_avg_intake` dominates at 58.5% importance — your personal drinking rhythm is the strongest predictor.

### Anomaly Detection (Isolation Forest)
- **Model:** `IsolationForest(contamination=0.10)`
- **Result:** 6 anomalous days out of 60 (10%)
- **Anomaly types detected:**
  - Low intake days (< 25th percentile) → Critical
  - Long sip gaps (> 4 hours) → Warning
  - Sudden intake spikes → Info

---

## 💡 Personalization Engine

```
Baseline = weight × 35 + activity_bonus
         = 70 × 35 + 250 = 2,700 ml

User avg = 2,022 ml (actual behavior)
Gap      = 678 ml below baseline

Personalized target = user_avg + 15% of gap
                    = 2,022 + 101.6 ≈ 2,124 ml
```

The system gradually bridges the gap rather than immediately demanding 2,700ml — this improves adherence.

---

## 🔌 ESP32 Hardware Setup

```
HC-SR04 Ultrasonic (Water Level):
  VCC  → 3.3V
  GND  → GND
  TRIG → GPIO 5
  ECHO → GPIO 18

DS18B20 Temperature:
  VCC  → 3.3V
  GND  → GND
  DATA → GPIO 4  (+ 4.7kΩ to 3.3V)
```

**Calibration:** Edit `distanceToMl()` in the firmware to match your bottle's dimensions.

---

## 📊 Dataset Stats
- **Duration:** 60 days (Jan 1 – Mar 1, 2025)
- **Records:** 672 sip events
- **Avg sips/day:** 11.2
- **Avg daily intake:** 2,022 ml
- **Temperature range:** 18°C – 29.8°C
- **Anomaly rate:** 10% (6 days)

---

## 🔮 Future Extensions (Phase 3)
- [ ] MongoDB integration (replace CSV)
- [ ] LSTM time-series model for hourly forecasting
- [ ] Autoencoder-based anomaly detection
- [ ] Weather API integration (OpenWeather)
- [ ] React Native mobile app
- [ ] Push notifications via Firebase FCM
- [ ] Multi-user support with JWT auth
- [ ] Wearable heart rate → exertion-aware targets
