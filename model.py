import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import time

print("Loading city_hour.csv...")
t0 = time.time()
df = pd.read_csv("city_hour.csv")

# Filter to Mumbai/Navi Mumbai or take latest representative rows if multi-city
if "City" in df.columns and "Mumbai" in df["City"].values:
    df_mumbai = df[df["City"].isin(["Mumbai", "Navi Mumbai"])].copy()
    if len(df_mumbai) > 5000:
        df = df_mumbai
    else:
        df = df.tail(60000).copy()
else:
    df = df.tail(60000).copy()

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(by=["City", "Datetime"]).reset_index(drop=True)

pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
df = df.dropna(subset=["AQI", "PM2.5", "PM10"]).copy()

# Forward fill and median fill minor pollutant gaps so rows aren't wasted
for p in pollutants:
    if p in df.columns:
        df[p] = df.groupby("City")[p].transform(lambda x: x.ffill().bfill())
        df[p] = df[p].fillna(df[p].median())

print(f"Dataset preprocessed ({len(df):,} records). Engineering lag features...")

# Temporal & Lag Features
for col in pollutants + ["AQI"]:
    df[f"{col}_lag1"] = df.groupby("City")[col].shift(1)
    df[f"{col}_lag3"] = df.groupby("City")[col].shift(3)
    df[f"{col}_rolling_mean_3h"] = df.groupby("City")[col].transform(lambda x: x.rolling(3).mean())

df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek

# 4-Hour Future Horizon Prediction
HORIZON = 4
df["AQI_future_4h"] = df.groupby("City")["AQI"].shift(-HORIZON)
df["AQI_delta_4h"] = df["AQI_future_4h"] - df["AQI"]

# Spike Alert Target (+30 pts climb or AQI >= 200)
df["Spike_Alert_4h"] = (
    (df["AQI_delta_4h"] >= 30) | (df["AQI_future_4h"] >= 200)
).astype(int)

# Spike Type Labeling
def classify_spike_type(row):
    if row["Spike_Alert_4h"] == 0:
        return "Nominal Air Quality"
    pm_ratio = row["PM10"] / (row["PM2.5"] + 1e-4)
    if pm_ratio > 1.8 and row["PM10"] > 110:
        return "Particulate Dust Surge"
    elif row["SO2"] >= 35:
        return "Chemical / Industrial Venting"
    elif row["NO2"] > 45 or row["CO"] > 1.8:
        return "Vehicular Exhaust Plume"
    return "Nominal Air Quality"

df["Spike_Type"] = df.apply(classify_spike_type, axis=1)

feature_cols = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "hour", "dayofweek",
    "PM2.5_lag1", "PM2.5_lag3", "AQI_lag1", "AQI_rolling_mean_3h"
]

# TARGETED DROPNA: Only drop rows where our specific features/targets are missing
df = df.dropna(subset=feature_cols + ["Spike_Alert_4h", "Spike_Type"]).reset_index(drop=True)

X = df[feature_cols]
y_alert = df["Spike_Alert_4h"]
y_type = df["Spike_Type"]

print(f"Clean samples ready for training: {len(X):,} rows.")

X_train, X_test, y_alert_train, y_alert_test, y_type_train, y_type_test = train_test_split(
    X, y_alert, y_type, test_size=0.2, shuffle=False
)

print(f"Training fast histogram gradient boosting models on {len(X_train):,} samples...")

alert_model = HistGradientBoostingClassifier(max_iter=80, max_depth=5, random_state=42)
alert_model.fit(X_train, y_alert_train)

type_model = HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=42)
type_model.fit(X_train, y_type_train)

# Save the trained weights
joblib.dump(alert_model, "aqi_alert_model.pkl")
joblib.dump(type_model, "aqi_type_model.pkl")

print(f"Done in {round(time.time() - t0, 1)}s! Serialized to 'aqi_alert_model.pkl' & 'aqi_type_model.pkl'.")
