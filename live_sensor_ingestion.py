import pandas as pd
import numpy as np
from datetime import datetime

# 1. Ward Geolocation Registry
WARD_DATA_REGISTRY = [
    {"Ward_ID": "A", "Ward_Name": "Airoli", "lat": 19.1557, "lon": 72.9986, "Zone": "Mixed Light Industrial"},
    {"Ward_ID": "B", "Ward_Name": "Mahape", "lat": 19.1082, "lon": 73.0163, "Zone": "TTC Industrial MIDC Corridor"},
    {"Ward_ID": "C", "Ward_Name": "Kopar Khairane", "lat": 19.1026, "lon": 73.0031, "Zone": "Commercial Density Hub"},
    {"Ward_ID": "D", "Ward_Name": "Vashi", "lat": 19.0770, "lon": 72.9986, "Zone": "Arterial Highway & Transit Hub"},
    {"Ward_ID": "E", "Ward_Name": "Nerul", "lat": 19.0330, "lon": 73.0297, "Zone": "Educational & Residential Belt"},
    {"Ward_ID": "F", "Ward_Name": "CBD Belapur", "lat": 19.0144, "lon": 73.0380, "Zone": "Administrative & Coastal Wind"},
    {"Ward_ID": "G", "Ward_Name": "Panvel/Kharghar", "lat": 19.0473, "lon": 73.0699, "Zone": "Expressway & Construction Hub"}
]

# 2. Authentic Indian CPCB NAQI Calculation
def calculate_cpcb_aqi(pm25, pm10, no2=25.0, so2=15.0, co=0.8):
    def calc_sub(val, bp):
        if val is None or np.isnan(val) or val < 0: return 0
        for (b_lo, b_hi, i_lo, i_hi) in bp:
            if b_lo <= val <= b_hi:
                return ((i_hi - i_lo) / (b_hi - b_lo)) * (val - b_lo) + i_lo
        return 500 if val > bp[-1][1] else 0

    pm25_bp = [(0, 30, 0, 50), (30.1, 60, 51, 100), (60.1, 90, 101, 200), (90.1, 120, 201, 300), (120.1, 250, 301, 400), (250.1, 380, 401, 500)]
    pm10_bp = [(0, 50, 0, 50), (50.1, 100, 51, 100), (100.1, 250, 101, 200), (250.1, 350, 201, 300), (350.1, 430, 301, 400), (430.1, 510, 401, 500)]
    no2_bp  = [(0, 40, 0, 50), (40.1, 80, 51, 100), (80.1, 180, 101, 200), (180.1, 280, 201, 300), (280.1, 400, 301, 400), (400.1, 500, 401, 500)]
    so2_bp  = [(0, 40, 0, 50), (40.1, 80, 51, 100), (80.1, 380, 101, 200), (380.1, 800, 201, 300), (800.1, 1600, 301, 400), (1600.1, 2100, 401, 500)]
    co_bp   = [(0, 1.0, 0, 50), (1.01, 2.0, 51, 100), (2.01, 10.0, 101, 200), (10.01, 17.0, 201, 300), (17.01, 34.0, 301, 400), (34.01, 50.0, 401, 500)]

    subs = [
        calc_sub(pm25, pm25_bp), calc_sub(pm10, pm10_bp),
        calc_sub(no2, no2_bp), calc_sub(so2, so2_bp), calc_sub(co, co_bp)
    ]
    return int(round(max(subs)))

# 3. Synchronized Spike Classification
def process_air_hazard_telemetry(row):
    rate = 1.25 if row["Current_AQI"] > 200 else 1.05
    future_aqi = int(row["Current_AQI"] * rate)

    if row["SO2"] >= 35:
        spike_type = "Chemical / Industrial Venting"
        action = "Dispatch MPCB Fly Squad; Activate static TTC industrial fog towers; Alert clinics."
        public_advisory = "INDUSTRIAL ALERT: Chemical odor detected. Keep windows closed and wear N95 masks."
        color = "#8A2BE2"
        spread_radius_km = 3.2
    elif (row["PM10"] / (row["PM2.5"] + 1e-4)) > 1.8 and row["PM10"] > 110:
        spike_type = "Particulate Dust Surge"
        action = "Deploy 2x Water Misting Cannons; Enforce immediate halt on un-sheeted earthworks."
        public_advisory = "DUST ALERT: High ambient dust levels. Children and runners avoid outdoor exposure."
        color = "#FF8C00"
        spread_radius_km = 2.0
    elif row["NO2"] > 45 or row["CO"] > 1.8:
        spike_type = "Vehicular Exhaust Plume"
        action = "Alert Traffic Police; Optimize green corridor signal cycles; Reroute heavy freight."
        public_advisory = "TRAFFIC POLLUTION: Exhaust buildup near transit corridors. Avoid roadside jogging."
        color = "#DC143C"
        spread_radius_km = 1.6
    else:
        spike_type = "Nominal Air Quality"
        action = "Standard ambient monitoring. No emergency deployment needed."
        public_advisory = "Air quality is within acceptable limits. Safe for regular outdoor routines."
        color = "#00FF7F" if row["Current_AQI"] <= 50 else "#92D050" if row["Current_AQI"] <= 100 else "#FFD700"
        spread_radius_km = 0.8

    return pd.Series([future_aqi, spike_type, action, public_advisory, color, spread_radius_km])

def get_calibrated_ward_data():
    df = pd.DataFrame(WARD_DATA_REGISTRY)

    # Realistic physical concentrations in ug/m3
    df["PM2.5"] = [52, 68, 55, 62, 45, 42, 65]
    df["PM10"]  = [85, 115, 90, 105, 75, 70, 145]
    df["NO2"]   = [32, 38, 35, 58, 26, 24, 36]
    df["CO"]    = [0.6, 0.8, 0.7, 1.9, 0.5, 0.4, 0.9]
    df["SO2"]   = [14, 42, 16, 12, 10, 8, 15]
    df["O3"]    = [28, 32, 30, 25, 22, 20, 26]

    # Calculate real Indian CPCB NAQI for each ward
    df["Current_AQI"] = df.apply(
        lambda r: calculate_cpcb_aqi(r["PM2.5"], r["PM10"], r["NO2"], r["SO2"], r["CO"]), axis=1
    )

    df[["Predicted_AQI_4h", "Spike_Type", "Recommended_Action", "Public_Advisory", "Color", "Impact_Radius_KM"]] = df.apply(
        process_air_hazard_telemetry, axis=1
    )

    return df

if __name__ == "__main__":
    print(get_calibrated_ward_data()[["Ward_Name", "Current_AQI", "Spike_Type", "Impact_Radius_KM"]])
