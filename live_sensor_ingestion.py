import pandas as pd
import numpy as np
from datetime import datetime

# 1. Ward Geolocation Registry
WARD_DATA_REGISTRY = [
    {"Ward_ID": "A", "Ward_Name": "Airoli", "lat": 19.1557, "lon": 72.9986, "Zone": "Residential/Industrial"},
    {"Ward_ID": "B", "Ward_Name": "Mahape", "lat": 19.1082, "lon": 73.0163, "Zone": "TTC Industrial MIDC"},
    {"Ward_ID": "C", "Ward_Name": "Kopar Khairane", "lat": 19.1026, "lon": 73.0031, "Zone": "Residential/Commercial"},
    {"Ward_ID": "D", "Ward_Name": "Vashi", "lat": 19.0770, "lon": 72.9986, "Zone": "High Traffic/Toll"},
    {"Ward_ID": "E", "Ward_Name": "Nerul", "lat": 19.0330, "lon": 73.0297, "Zone": "Educational/Residential"},
    {"Ward_ID": "F", "Ward_Name": "CBD Belapur", "lat": 19.0144, "lon": 73.0380, "Zone": "Administrative/Coastal"},
    {"Ward_ID": "G", "Ward_Name": "Panvel/Kharghar", "lat": 19.0473, "lon": 73.0699, "Zone": "Highway/Construction"}
]

# 2. Advanced Spike Classification Logic (shared by Admin/Citizen)
def process_air_hazard_telemetry(row):
    # Forecast AQI after 3-4 hours based on pollutant momentum proxy
    rate = 1.25 if row["Current_AQI"] > 200 else 1.08
    future_aqi = int(row["Current_AQI"] * rate)
    
    # Classify Spike Nature and required Municipal Action
    if row["SO2"] >= 45:
        spike_type = "Chemical/Industrial Smog"
        action = "Dispatch MPCB Fly Squad; Activate TTC Fog Towers; Alert health centers."
        public_advisory = "INDUSTRIAL ALERT: Limit outdoor activity. Wear N95 masks. Report smells to NMMC."
        color = "purple" # High toxicity
        spread_radius_km = 3.2
    elif (row["PM10"] / (row["PM2.5"] + 1e-4)) > 1.9 and row["PM10"] > 200:
        spike_type = "Particulate/Dust Surge"
        action = "Deploy 2x Misting cannons; Halt un-sheeted earthworks."
        public_advisory = "DUST ALERT: Close windows. Traffic police: ensure fog lights."
        color = "orange" # High PM
        spread_radius_km = 2.0
    elif row["NO2"] > 70 or row["CO"] > 2.0:
        spike_type = "Vehicular Exhaust Peak"
        action = "Alert Traffic Police; Optimize signals at arterial junctions."
        public_advisory = "TRAFFIC POLLUTION: Drivers: Idle off at signals. Vulnerable: stay indoors."
        color = "red" # High Combustibles
        spread_radius_km = 1.5
    else:
        spike_type = "Stable Ambient Conditions"
        action = "Routine monitoring."
        public_advisory = "Enjoy your day. Air quality within limits."
        color = "green" # Normal
        spread_radius_km = 0.5
        
    return pd.Series([future_aqi, spike_type, action, public_advisory, color, spread_radius_km])

def get_calibrated_ward_data():
    """
    Simulates fetching and processing live telemetry from Ward IoT edge nodes.
    """
    df = pd.DataFrame(WARD_DATA_REGISTRY)
    
    # Add varying pollutant values across Navi Mumbai wards for visualization
    df["PM2.5"] = [88, 165, 110, 140, 72, 65, 155]
    df["PM10"] = [165, 240, 190, 210, 130, 115, 310]
    df["NO2"] = [45, 55, 48, 82, 32, 28, 52]
    df["CO"] = [1.2, 1.5, 1.1, 2.6, 0.8, 0.7, 1.8]
    df["SO2"] = [22, 62, 28, 18, 14, 12, 20]
    df["O3"] = [35, 40, 38, 30, 25, 20, 32]
    
    # Proxy AQI Calculation
    df["Current_AQI"] = df[["PM2.5", "PM10", "NO2"]].max(axis=1) * 1.5
    df["Current_AQI"] = df["Current_AQI"].astype(int)
    
    # Process Advanced Logic
    df[["Predicted_AQI_4h", "Spike_Type", "Recommended_Action", "Public_Advisory", "Color", "Impact_Radius_KM"]] = df.apply(
        process_air_hazard_telemetry, axis=1
    )
    
    return df
