# aqi-monitor
Air Quality Monitoring 
Hybrid Random Forest + LSTM model for real-time AQI forecasting
Project Overview
An AI system that combines Random Forest (RF) and Long Short-Term Memory (LSTM) models to forecast the Air Quality Index (AQI) with high accuracy. Designed for environmental agencies and public health organizations.

Core Objectives:
 Real-time AQI prediction
 Hybrid AI model (RF + LSTM) for robustness
 User-friendly dashboard with visualizations

 Key Features
Hybrid AI Model:

RF for interpretability + LSTM for temporal patterns.

Optimized with GridSearchCV (RF) and Keras Tuner (LSTM).

Data Pipeline:

Integrates OpenAQ (pollution) + OpenWeatherMap (weather).

Cyclic feature engineering (hour_sin, month_cos).

Web App:

Backend: FastAPI

Frontend: React.js + Chart.js

Installation
Prerequisites
Python 3.9+
Node.js (for frontend)

#Run the backend (FastAPI):uvicorn backend.main:app --reload  
#Run the frontend (React):npm run dev
