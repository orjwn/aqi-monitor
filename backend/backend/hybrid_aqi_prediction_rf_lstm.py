try:
    import keras_tuner as kt
except ImportError:
    print("Installing keras-tuner...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner"])
    import keras_tuner as kt

from tensorflow.compat.v1 import reset_default_graph
reset_default_graph()
import os
import requests
import pandas as pd
import numpy as np
import holidays
import joblib
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import keras_tuner as kt

# CONSTANTS
TIMESTEPS = 24 * 7  # 1 week of hourly data
FORECAST_HORIZON = 24  # Predict 24h ahead
LAT = 51.5074  # London coordinates
LON = -0.1278
PAST_DAYS = 90  # 3 months historical data
MODEL_DIR = "app/models"

# UTILITY CLASSES
class EpochProgress(Callback):
    """Custom callback to display training progress"""
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        print(f"Epoch {epoch+1:03d} - loss: {logs['loss']:.4f} - mae: {logs['mae']:.4f} - "
              f"val_loss: {logs['val_loss']:.4f} - val_mae: {logs['val_mae']:.4f} - {elapsed:.1f}s")

# DATA FUNCTIONS
def fetch_air_quality_data(lat=LAT, lon=LON, days=PAST_DAYS):
    """Fetches air quality data from OpenAQ API with fallback to cached data"""
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,ozone,nitrogen_dioxide&past_days={days}"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "pm10": data["hourly"]["pm10"],
            "pm2_5": data["hourly"]["pm2_5"],
            "ozone": data["hourly"]["ozone"],
            "nitrogen_dioxide": data["hourly"]["nitrogen_dioxide"]
        })
        df.to_csv("air_quality_enhanced.csv", index=False)
        return df
    except Exception as e:
        print(f"Air quality API fetch failed: {e}. Using cached data.")
        return pd.read_csv("air_quality_enhanced.csv")

def fetch_weather_data(lat=LAT, lon=LON, api_key='d8d8574b09dd165fd64c98c47070d233'):
    """Fetches weather data from OpenWeatherMap with robust error handling"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if 'list' not in data:
            raise ValueError("Unexpected API response format")
            
        records = []
        for entry in data['list']:
            records.append({
                'time': pd.to_datetime(entry['dt_txt']),
                'temp': entry['main']['temp'],
                'humidity': entry['main']['humidity'],
                'pressure': entry['main']['pressure'],
                'wind_speed': entry['wind']['speed']
            })
            
        df = pd.DataFrame(records)
        df.to_csv("weather_enhanced.csv", index=False)
        return df
        
    except Exception as e:
        print(f"Weather API error: {str(e)}")
        try:
            return pd.read_csv("weather_enhanced.csv")
        except FileNotFoundError:
            print("No cached data found. Generating dummy weather data.")
            dates = pd.date_range(end=datetime.now(), periods=24*7, freq='h')
            return pd.DataFrame({
                'time': dates,
                'temp': np.random.uniform(10, 30, len(dates)),
                'humidity': np.random.uniform(30, 80, len(dates)),
                'pressure': np.random.uniform(980, 1040, len(dates)),
                'wind_speed': np.random.uniform(0, 10, len(dates))
            })

# FEATURE ENGINEERING
def add_time_features(df):
    """Adds temporal features including trigonometric encoding"""
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    
    # Trigonometric encoding for cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Special days
    uk_holidays = holidays.UnitedKingdom()
    df['is_holiday'] = df['time'].apply(lambda x: x in uk_holidays).astype(int)
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Rolling statistics
    df['pm2_5_24h_avg'] = df['pm2_5'].rolling(window=24).mean()
    df['pm10_24h_avg'] = df['pm10'].rolling(window=24).mean()
    
    return df.dropna()

def preprocess_data(aq_df, weather_df):
    """Merges and cleans air quality and weather data"""
    df = pd.merge(aq_df, weather_df, on='time', how='outer').sort_values('time')
    df = df.interpolate(method='linear').bfill()  # Handle missing values
    df = add_time_features(df)
    df = df.dropna().drop_duplicates()
    df.to_csv("processed_data.csv", index=False)
    return df

# MODEL FUNCTIONS
def prepare_lstm_data(X, y, timesteps=TIMESTEPS, horizon=FORECAST_HORIZON):
    """Creates time-series sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X) - horizon):
        X_seq.append(X.iloc[i-timesteps:i].values)
        y_seq.append(y.iloc[i + horizon])  # Predict future values
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(hp, input_shape):
    """Keras Tuner compatible LSTM builder"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(
            units=hp.Int('units_1', 64, 256, step=64),
            return_sequences=True,
            kernel_regularizer=l1_l2(
                l1=hp.Float('l1_reg', 1e-4, 1e-1, sampling='log'),
                l2=hp.Float('l2_reg', 1e-4, 1e-1, sampling='log')
            )
        ),
        BatchNormalization(),
        Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)),
        LSTM(
            units=hp.Int('units_2', 32, 128, step=32),
            kernel_regularizer=l1_l2(
                l1=hp.Float('l1_reg', 1e-4, 1e-1, sampling='log'),
                l2=hp.Float('l2_reg', 1e-4, 1e-1, sampling='log')
            )
        ),
        Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='mse',
        metrics=['mae']
    )
    return model

def tune_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best RF params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_lstm_model(X_train, y_train, X_val, y_val, input_shape):
    """Train LSTM with proper progress display"""
    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, input_shape),
        objective='val_mae',
        max_trials=10,
        directory='tuner_results',
        project_name='aqi_lstm',
        overwrite=True
    )
    
    print("\nStarting LSTM hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5),
            EpochProgress()
        ],
        verbose=0
    )
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"\nBest LSTM params: {best_hps.values}")
    return best_model

def train_and_save_models(X, y):
    """Main training pipeline with hyperparameter tuning"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("presentation", exist_ok=True)
    
    # Feature scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.save")
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    results = {'rf': {'mae': [], 'r2': []}, 'lstm': {'mae': [], 'r2': []}}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled_df)):
        print(f"\n=== Fold {fold + 1} ===")
        X_train, X_test = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 1. Train Random Forest
        rf_model = tune_random_forest(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        results['rf']['mae'].append(mean_absolute_error(y_test, rf_pred))
        results['rf']['r2'].append(r2_score(y_test, rf_pred))

        # 2. Prepare LSTM data
        X_seq, y_seq = prepare_lstm_data(X_scaled_df, y)
        split = int(0.8 * len(X_seq))
        X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
        y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

        # 3. Train LSTM
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        lstm_model = train_lstm_model(
            X_train_seq, y_train_seq,
            X_test_seq, y_test_seq,
            input_shape
        )
        lstm_pred = lstm_model.predict(X_test_seq).flatten()
        results['lstm']['mae'].append(mean_absolute_error(y_test_seq, lstm_pred))
        results['lstm']['r2'].append(r2_score(y_test_seq, lstm_pred))

    # Save best models
    joblib.dump(rf_model, f"{MODEL_DIR}/rf_model.pkl")
    lstm_model.save(f"{MODEL_DIR}/aqi_prediction_model.keras", include_optimizer=False)

    # Print results
    print("\n=== Final Results ===")
    for model in results:
        avg_mae = np.mean(results[model]['mae'])
        avg_r2 = np.mean(results[model]['r2'])
        print(f"{model.upper():<6} - MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}")

# VISUALIZATION
def plot_results(y_true, y_pred, title):
    """Plots actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"{title} (MAE: {mean_absolute_error(y_true, y_pred):.2f})")
    plt.xlabel("Time Steps")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"presentation/{title.lower().replace(' ', '_')}.png")
    plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    # 1. Data collection
    print("Fetching air quality data...")
    aq_data = fetch_air_quality_data()
    print("Fetching weather data...")
    weather_data = fetch_weather_data()
    
    # 2. Preprocessing
    print("\nPreprocessing data...")
    processed_df = preprocess_data(aq_data, weather_data)
    features = processed_df[[
        'pm10', 'ozone', 'nitrogen_dioxide', 'temp', 'humidity',
        'wind_speed', 'pressure', 'hour_sin', 'hour_cos',
        'month_sin', 'month_cos', 'day_of_week', 'is_holiday',
        'is_weekend', 'pm2_5_24h_avg', 'pm10_24h_avg'
    ]]
    target = processed_df['pm2_5']
    
    # 3. Model training
    print("\nTraining models...")
    train_and_save_models(features, target)
    
    # 4. Load and demo best models
    print("\nGenerating demo predictions...")
    rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    lstm_model = load_model(f"{MODEL_DIR}/aqi_prediction_model.keras")
    
    # Generate example predictions
    example_idx = -100  # Last 100 hours
    X_sample = features.iloc[example_idx:]
    y_true = target.iloc[example_idx + FORECAST_HORIZON:]  # Shift for 24h prediction
    
    # RF prediction
    scaler = joblib.load(f"{MODEL_DIR}/scaler.save")  # Load the saved scaler
    X_sample_scaled = pd.DataFrame(
        scaler.transform(X_sample),
        columns=X_sample.columns,
        index=X_sample.index
    )
    rf_pred = rf_model.predict(X_sample_scaled)
    plot_results(y_true, rf_pred[:-FORECAST_HORIZON], "Random Forest 24h Forecast")
    
    # LSTM prediction
    X_seq, _ = prepare_lstm_data(features, target)
    lstm_pred = lstm_model.predict(X_seq[example_idx:]).flatten()
    plot_results(y_true, lstm_pred[:-FORECAST_HORIZON], "LSTM 24h Forecast")
    
    print("\nTraining complete! Models saved to app/models/")