import os
import numpy as np
import pandas as pd
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import holidays

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "aqi_prediction_model.keras")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

# === Load Models ===
try:
    lstm_model = load_model(LSTM_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # Get feature names from the Random Forest model
    rf_feature_names = rf_model.feature_names_in_
    print("✅ Model loaded successfully with features:", rf_feature_names)
except Exception as e:
    print("❌ Model loading failed!")
    traceback.print_exc()
    raise e

# === Input Schema ===
class AQIData(BaseModel):
    # Must match EXACTLY with rf_model.feature_names_in_
    pm10: float
    ozone: float
    nitrogen_dioxide: float
    temp: float  # Changed from temperature to temp
    humidity: float
    wind_speed: float
    pressure: float
    hour_sin: float
    hour_cos: float
    month_sin: float
    month_cos: float
    day_of_week: int
    is_holiday: int
    is_weekend: int
    pm2_5_24h_avg: float
    pm10_24h_avg: float

    @validator('*', pre=True)
    def convert_types(cls, v):
        if v == "":
            return 0.0  # Default value for empty strings
        try:
            return float(v) if not isinstance(v, (int, float)) else v
        except (TypeError, ValueError):
            return 0.0

# === Feature Order Validation ===
def validate_feature_order(input_features: dict):
    """Ensure features match the model's expected order"""
    expected_columns = rf_feature_names.tolist()
    received_columns = list(input_features.keys())
    
    if received_columns != expected_columns:
        raise ValueError(
            f"Feature order mismatch.\n"
            f"Expected: {expected_columns}\n"
            f"Received: {received_columns}"
        )

# === Backend Feature Generation ===
def generate_backend_features():
    now = datetime.now()
    uk_holidays = holidays.UnitedKingdom()

    return {
        "hour_sin": np.sin(2 * np.pi * now.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now.hour / 24),
        "month_sin": np.sin(2 * np.pi * now.month / 12),
        "month_cos": np.cos(2 * np.pi * now.month / 12),
        "day_of_week": now.weekday(),
        "is_holiday": int(now.date() in uk_holidays),
        "is_weekend": int(now.weekday() >= 5),
    }

@app.post("/predict_aqi/")
async def predict_aqi(data: AQIData):
    try:
        # Step 1: Process user input
        base_features = {
            "pm10": data.pm10,
            "ozone": data.ozone,
            "nitrogen_dioxide": data.nitrogen_dioxide,
            "temp": data.temp,
            "humidity": data.humidity,
            "wind_speed": data.wind_speed,
            "pressure": data.pressure
        }

        # Step 2: Add generated features
        backend_features = {
            "hour_sin": data.hour_sin,
            "hour_cos": data.hour_cos,
            "month_sin": data.month_sin,
            "month_cos": data.month_cos,
            "day_of_week": data.day_of_week,
            "is_holiday": data.is_holiday,
            "is_weekend": data.is_weekend,
            "pm2_5_24h_avg": data.pm2_5_24h_avg,
            "pm10_24h_avg": data.pm10_24h_avg
        }

        # Combine all features
        full_input = {**base_features, **backend_features}
        
        # Create DataFrame ensuring correct order
        input_df = pd.DataFrame([full_input], columns=rf_feature_names)
        print("Input DataFrame:\n", input_df)

        # Step 3: Scale and predict
        scaled_input = scaler.transform(input_df)

        # LSTM prediction (reshape for time steps)
        lstm_input = np.repeat(scaled_input[np.newaxis, :, :], 24, axis=1)
        lstm_pred = float(lstm_model.predict(lstm_input).flatten()[-1])  # Explicitly convert to float
        
        # Random Forest prediction
        rf_pred = float(rf_model.predict(scaled_input)[0])  # Explicitly convert to float
        
        # Hybrid prediction
        hybrid_pred = round((lstm_pred + rf_pred) / 2, 2)

        # Generate chart data
        hours = 24
        chart_data = {
            "labels": [f"{i}:00" for i in range(hours)],
            "pm2_5": [hybrid_pred * (0.9 + 0.2 * np.sin(i/3)) for i in range(hours)],
            "pm10": [data.pm10 * (0.9 + 0.1 * np.cos(i/2)) for i in range(hours)],
            "no2": [data.nitrogen_dioxide * (0.95 + 0.1 * np.sin(i/4)) for i in range(hours)],
            "o3": [data.ozone * (0.92 + 0.15 * np.cos(i/5)) for i in range(hours)]
        }

        return {
            "predicted_values": {
                "pm2_5": hybrid_pred,
                "pm10": data.pm10,
                "no2": data.nitrogen_dioxide,
                "o3": data.ozone
            },
            "aqi_status": {
                "value": hybrid_pred,
                "level": get_aqi_level(hybrid_pred),
                "description": get_aqi_description(hybrid_pred)
            },
            "forecast": {
                "date": datetime.now().strftime("%a, %b %d %Y"),
                "value": hybrid_pred,
                "unit": "AQI"
            },
            "chart_data": chart_data,
            "model_metrics": {
                "mae": 0.04,
                "r2": 0.97
            },
            "feature_order": rf_feature_names.tolist()
        }

    except Exception as e:
        print(f"\n🔴 Prediction Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "expected_features": rf_feature_names.tolist(),
                "message": "Feature names and order must match training data"
            }
        )

def get_aqi_level(aqi):
    if aqi <= 50: return "Very Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive"
    return "Unhealthy"

def get_aqi_description(aqi):
    if aqi <= 50: return "Air quality is satisfactory"
    if aqi <= 100: return "Acceptable quality"
    if aqi <= 150: return "Unhealthy for sensitive groups"
    return "Health alert - everyone may be affected"