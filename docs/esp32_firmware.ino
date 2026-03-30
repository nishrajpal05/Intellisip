/*
  HydroIQ Smart Water Bottle — ESP32 Firmware
  ============================================
  Hardware:
    - ESP32 Dev Board
    - HC-SR04 Ultrasonic sensor (water level)
    - DS18B20 Temperature sensor (ambient)
    - OLED 128x64 I2C display (optional)

  Wiring:
    HC-SR04  → TRIG=GPIO5, ECHO=GPIO18
    DS18B20  → DATA=GPIO4 (with 4.7kΩ pull-up to 3.3V)
    OLED     → SDA=GPIO21, SCL=GPIO22

  Sends POST /api/ingest-sip every sip detection
  Fetches GET /api/predict-next-hour every 10 min
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <time.h>

// ── CONFIG ──────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://YOUR_SERVER_IP:8000";
const char* USER_ID       = "U01";
const char* NTP_SERVER    = "pool.ntp.org";

// ── PINS ────────────────────────────────────────────────────────
#define TRIG_PIN      5
#define ECHO_PIN      18
#define TEMP_PIN      4
#define LED_ALERT     2   // onboard LED for alerts

// ── SENSORS ─────────────────────────────────────────────────────
OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

// ── STATE ───────────────────────────────────────────────────────
float lastWaterLevel   = -1;
float dailyTotal       = 0;
unsigned long lastSipMs = 0;
unsigned long lastPredMs = 0;
bool  alertActive      = false;

// ── HELPERS ─────────────────────────────────────────────────────
String getTimestamp() {
  time_t now = time(nullptr);
  struct tm* t = localtime(&now);
  char buf[20];
  sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d",
    t->tm_year+1900, t->tm_mon+1, t->tm_mday,
    t->tm_hour, t->tm_min, t->tm_sec);
  return String(buf);
}

float readWaterLevel() {
  // HC-SR04: measure distance to water surface
  // Full bottle = ~3cm, Empty = ~20cm
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  float distance = duration * 0.034 / 2.0;  // cm
  return (distance < 2 || distance > 25) ? -1 : distance;
}

float distanceToMl(float distanceCm) {
  // Calibrate for your bottle:
  // 3cm  → 750ml (full)
  // 20cm → 0ml   (empty)
  float ml = map(distanceCm * 10, 30, 200, 750, 0);
  return constrain(ml, 0, 750);
}

float readTemperature() {
  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  return (t == -127.0) ? 25.0 : t;  // fallback 25°C
}

// ── API CALLS ───────────────────────────────────────────────────
bool postSip(float water_ml, float temperature) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(SERVER_URL) + "/api/ingest-sip";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(8000);

  StaticJsonDocument<256> doc;
  doc["user_id"]     = USER_ID;
  doc["timestamp"]   = getTimestamp();
  doc["water_ml"]    = (int)round(water_ml);
  doc["temperature"] = temperature;

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  bool ok = (code == 200);

  if (ok) {
    Serial.printf("✓ Sip logged: %.0fml @ %.1f°C\n", water_ml, temperature);
  } else {
    Serial.printf("✗ POST failed: %d\n", code);
  }

  http.end();
  return ok;
}

void fetchPrediction() {
  if (WiFi.status() != WL_CONNECTED) return;

  float temp = readTemperature();
  String url = String(SERVER_URL) + "/api/predict-next-hour?temperature=" + String(temp, 1);

  HTTPClient http;
  http.begin(url);
  http.setTimeout(8000);
  int code = http.GET();

  if (code == 200) {
    String payload = http.getString();
    StaticJsonDocument<512> doc;
    deserializeJson(doc, payload);

    float predicted = doc["predicted_ml"];
    float remaining = doc["remaining_ml"];
    bool  onTrack   = doc["on_track"];

    Serial.printf("📊 Prediction: %.0fml next sip | %.0fml remaining | %s\n",
      predicted, remaining, onTrack ? "On track ✓" : "Behind ⚠");

    // Alert if off-track: blink LED
    if (!onTrack) {
      alertActive = true;
      Serial.println("⚠ ALERT: Hydration falling behind target!");
    } else {
      alertActive = false;
    }
  }

  http.end();
}

void fetchAnomalyAlerts() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(String(SERVER_URL) + "/api/anomaly-alerts");
  http.setTimeout(8000);
  int code = http.GET();

  if (code == 200) {
    String payload = http.getString();
    StaticJsonDocument<1024> doc;
    deserializeJson(doc, payload);

    int critical = doc["critical_count"];
    if (critical > 0) {
      Serial.printf("🚨 %d critical anomaly alert(s) active!\n", critical);
    }
  }
  http.end();
}

// ── SETUP ───────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n🚰 HydroIQ ESP32 Starting...");

  // Pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_ALERT, OUTPUT);
  tempSensor.begin();

  // WiFi
  Serial.printf("Connecting to %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries++ < 30) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n✓ WiFi connected: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n✗ WiFi failed — offline mode");
  }

  // NTP time sync
  configTime(19800, 0, NTP_SERVER);  // UTC+5:30 for India; adjust offset
  Serial.println("⏰ Time synced via NTP");

  // Initial read
  lastWaterLevel = distanceToMl(readWaterLevel());
  Serial.printf("🍶 Initial water level: %.0f ml\n", lastWaterLevel);

  Serial.println("✅ HydroIQ ready!\n");
}

// ── MAIN LOOP ───────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  // ── SIP DETECTION (every 2 seconds) ──
  float currentLevel = distanceToMl(readWaterLevel());

  if (currentLevel >= 0 && lastWaterLevel >= 0) {
    float sipVolume = lastWaterLevel - currentLevel;

    // Valid sip: 30ml–400ml, at least 15s since last sip
    if (sipVolume >= 30 && sipVolume <= 400 && (now - lastSipMs) > 15000) {
      float temperature = readTemperature();
      dailyTotal += sipVolume;
      lastSipMs = now;

      Serial.printf("\n💧 Sip detected! %.0f ml (daily: %.0f ml)\n", sipVolume, dailyTotal);

      postSip(sipVolume, temperature);
    }
  }

  if (currentLevel >= 0) lastWaterLevel = currentLevel;

  // ── PREDICTION FETCH (every 10 minutes) ──
  if (now - lastPredMs > 600000UL || lastPredMs == 0) {
    lastPredMs = now;
    fetchPrediction();
    // Check anomalies every 30 min
    if (now % 1800000UL < 600000UL) fetchAnomalyAlerts();
  }

  // ── ALERT LED ──
  if (alertActive) {
    digitalWrite(LED_ALERT, (now / 500) % 2);  // blink 1Hz
  } else {
    digitalWrite(LED_ALERT, LOW);
  }

  // ── DAILY RESET at midnight ──
  time_t t = time(nullptr);
  struct tm* tm = localtime(&t);
  if (tm->tm_hour == 0 && tm->tm_min == 0 && tm->tm_sec < 10) {
    Serial.printf("🌅 Day reset. Yesterday total: %.0f ml\n", dailyTotal);
    dailyTotal = 0;
  }

  delay(2000);
}
