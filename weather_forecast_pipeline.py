
# weather_forecast_pipeline.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Daily
import requests
import json
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ==========================
# CONFIG
# ==========================
CITY = "New Delhi"
LAT, LON = 28.6139, 77.2090
DAYS_HISTORY = 30

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_TABLE = "forecast"

MODEL_PATH = "weather_model.h5"
SCALER_PATH = "scaler.pkl"

# ==========================
# STEP 1: Fetch Weather Data
# ==========================
today = datetime.now().date()
start = today - timedelta(days=DAYS_HISTORY)
end = today

location = Point(LAT, LON)
data = Daily(location, start, end)
data = data.fetch().reset_index()
data.rename(columns={
    "tavg": "temperature",
    "wspd": "wind_speed",
    "pres": "pressure",
    "prcp": "precipitation"
}, inplace=True)

# Keep only required columns
data = data[["time", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]]
data.rename(columns={"time": "date"}, inplace=True)
data.dropna(inplace=True)

# ==========================
# STEP 2: Preprocess
# ==========================
scaler = joblib.load(SCALER_PATH)
X = scaler.transform(data.drop(columns=["date"]))

# ==========================
# STEP 3: Predict
# ==========================
model = load_model(MODEL_PATH)
predicted_temperature = model.predict(X).flatten()
data["predicted_temperature"] = predicted_temperature

# Optional: add actual_temperature for training/eval (mock value here)
data["actual_temperature"] = data["temperature"] + np.random.normal(0, 1, size=len(data))

# ==========================
# STEP 4: Upload to Supabase
# ==========================
headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

for _, row in data.iterrows():
    row_dict = row.to_dict()
    row_dict["date"] = row_dict["date"].strftime("%Y-%m-%d")  # Convert Timestamp
    try:
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers=headers,
            data=json.dumps(row_dict)
        )
        print(f"Uploaded {row_dict['date']} → {res.status_code}")
    except Exception as e:
        print(f"Error uploading {row_dict['date']}: {e}")
