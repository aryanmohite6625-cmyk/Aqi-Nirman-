import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# 1. Load dataset (city_hour.csv)
df = pd.read_csv("city_hour.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(by=["City", "Datetime"]).reset_index(drop=True)

# Focus on Mumbai / Navi Mumbai or multi-city representative temporal data
pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
df = df.dropna(subset=["AQI"] + pollutants).copy()

# 2. Engineer Temporal & Lag Features
for col in pollutants + ["AQI"]:
    df[f"{col}_lag1"] = df.groupby("City")[col].shift(1)
    df[f"{col}_lag3"] = df.groupby("City")[col].shift(3)
    df[f"{col}_rolling_mean_3h"] = (
        df.groupby("City")[col].transform(lambda x: x.rolling(3).mean())
    )

df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek

# 3. Define 4-Hour Future Target: Will AQI spike by >50 points or exceed 250 in 4 hours?
HORIZON = 4
df["AQI_future_4h"] = df.groupby("City")["AQI"].shift(-HORIZON)
df["AQI_delta_4h"] = df["AQI_future_4h"] - df["AQI"]

# Target: 1 if severe spike occurs within 4 hrs, else 0
df["Spike_Alert_4h"] = (
    (df["AQI_delta_4h"] >= 40) | (df["AQI_future_4h"] >= 200)
).astype(int)

# Identify Spike Type based on pollutant dominance in the spike
def classify_spike_type(row):
    if row["Spike_Alert_4h"] == 0:
        return "Normal"
    # Ratio checks
    pm_ratio = row["PM10"] / (row["PM2.5"] + 1e-4)
    if pm_ratio > 1.8 and row["PM10"] > 150:
        return "Dust / Construction Spike"
    elif row["NO2"] > 60 or row["CO"] > 1.5:
        return "Vehicular / Congestion Spike"
    elif row["SO2"] > 30 or row["O3"] > 60:
        return "Industrial / Smog Spike"
    return "General Ambient Spike"

df["Spike_Type"] = df.apply(classify_spike_type, axis=1)

df = df.dropna()

feature_cols = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "hour", "dayofweek",
    "PM2.5_lag1", "PM2.5_lag3", "AQI_lag1", "AQI_rolling_mean_3h"
]

X = df[feature_cols]
y_alert = df["Spike_Alert_4h"]
y_type = df["Spike_Type"]

# 4. Train Models
X_train, X_test, y_train, y_test = train_test_split(
    X, y_alert, test_size=0.2, shuffle=False
)

alert_model = GradientBoostingClassifier(n_estimators=120, max_depth=5, random_state=42)
alert_model.fit(X_train, y_train)

type_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
type_model.fit(X_train, y_type.iloc[X_train.index])

# Save models
joblib.dump(alert_model, "aqi_alert_model.pkl")
joblib.dump(type_model, "aqi_type_model.pkl")
print("ML Models successfully trained and serialized.")
