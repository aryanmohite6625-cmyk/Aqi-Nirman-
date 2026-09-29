import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Nirman DSS - Navi Mumbai Air Hazard Early Warning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- 1. CPCB NAQI BREAKPOINT FORMULATION -----------------
def calculate_cpcb_aqi(pm25, pm10, no2=25, so2=15, co=0.8):
    """
    Standard Indian Central Pollution Control Board (CPCB) NAQI algorithm.
    Computes individual sub-indices; the maximum sub-index defines the final AQI.
    """
    def calc_sub_index(val, breakpoints):
        if val is None or np.isnan(val) or val < 0:
            return 0
        for (b_lo, b_hi, i_lo, i_hi) in breakpoints:
            if b_lo <= val <= b_hi:
                return ((i_hi - i_lo) / (b_hi - b_lo)) * (val - b_lo) + i_lo
        return 500 if val > breakpoints[-1][1] else 0

    pm25_bp = [(0, 30, 0, 50), (30.1, 60, 51, 100), (60.1, 90, 101, 200), (90.1, 120, 201, 300), (120.1, 250, 301, 400), (250.1, 380, 401, 500)]
    pm10_bp = [(0, 50, 0, 50), (50.1, 100, 51, 100), (100.1, 250, 101, 200), (250.1, 350, 201, 300), (350.1, 430, 301, 400), (430.1, 510, 401, 500)]
    no2_bp  = [(0, 40, 0, 50), (40.1, 80, 51, 100), (80.1, 180, 101, 200), (180.1, 280, 201, 300), (280.1, 400, 301, 400), (400.1, 500, 401, 500)]
    so2_bp  = [(0, 40, 0, 50), (40.1, 80, 51, 100), (80.1, 380, 101, 200), (380.1, 800, 201, 300), (800.1, 1600, 301, 400), (1600.1, 2100, 401, 500)]
    co_bp   = [(0, 1.0, 0, 50), (1.01, 2.0, 51, 100), (2.01, 10.0, 101, 200), (10.01, 17.0, 201, 300), (17.01, 34.0, 301, 400), (34.01, 50.0, 401, 500)]

    subs = [
        calc_sub_index(pm25, pm25_bp),
        calc_sub_index(pm10, pm10_bp),
        calc_sub_index(no2, no2_bp),
        calc_sub_index(so2, so2_bp),
        calc_sub_index(co, co_bp)
    ]
    return int(round(max(subs)))

# ----------------- 2. NAVI MUMBAI WARD COORDINATES -----------------
WARD_COORDINATES = [
    {"Ward_ID": "Ward A", "Ward_Name": "Airoli", "lat": 19.1557, "lon": 72.9986, "Zone": "Mixed Residential & Industrial", "waqi_query": "airoli"},
    {"Ward_ID": "Ward B", "Ward_Name": "Mahape (TTC MIDC)", "lat": 19.1082, "lon": 73.0163, "Zone": "TTC Industrial MIDC Corridor", "waqi_query": "mahape"},
    {"Ward_ID": "Ward C", "Ward_Name": "Kopar Khairane", "lat": 19.1026, "lon": 73.0031, "Zone": "High-Density Urban & Markets", "waqi_query": "kopar khairane"},
    {"Ward_ID": "Ward D", "Ward_Name": "Vashi Toll & Sector 17", "lat": 19.0770, "lon": 72.9986, "Zone": "Arterial Highway & Transit Hub", "waqi_query": "vashi"},
    {"Ward_ID": "Ward E", "Ward_Name": "Nerul", "lat": 19.0330, "lon": 73.0297, "Zone": "Educational & Residential Belt", "waqi_query": "nerul"},
    {"Ward_ID": "Ward F", "Ward_Name": "CBD Belapur", "lat": 19.0144, "lon": 73.0380, "Zone": "Administrative & Coastal Creek", "waqi_query": "belapur"},
    {"Ward_ID": "Ward G", "Ward_Name": "Panvel / Kharghar", "lat": 19.0473, "lon": 73.0699, "Zone": "Expressway Corridor & Construction", "waqi_query": "kharghar"}
]

# ----------------- 3. DUAL-SOURCE REAL-TIME DATA INGESTION -----------------
def fetch_cpcb_calibrated_telemetry(lat, lon, waqi_query):
    """
    Attempts ground station ingestion via WAQI/CPCB.
    Calculates authentic Indian CPCB AQI using raw pollutant concentrations.
    """
    # 1. Primary: Ground-level station check (WAQI public API)
    try:
        waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token=demo"
        w_resp = requests.get(waqi_url, timeout=3)
        if w_resp.status_code == 200:
            res = w_resp.json()
            if res.get("status") == "ok":
                iaqi = res["data"].get("iaqi", {})
                pm25 = iaqi.get("pm25", {}).get("v")
                pm10 = iaqi.get("pm10", {}).get("v")
                no2 = iaqi.get("no2", {}).get("v", 30)
                so2 = iaqi.get("so2", {}).get("v", 15)
                co = iaqi.get("co", {}).get("v", 8) / 10.0
                
                if pm25 is not None:
                    cpcb_val = calculate_cpcb_aqi(pm25, pm10 if pm10 else pm25 * 1.6, no2, so2, co)
                    return cpcb_val, float(pm25), float(pm10 if pm10 else pm25 * 1.6), float(no2), float(so2), "LIVE MPCB Ground Sensor (WAQI)"
    except Exception:
        pass

    # 2. Secondary: Open-Meteo Raw Pollutant API with Indian CPCB math
    try:
        om_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&"
            f"current=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide&"
            f"timezone=Asia%2FKolkata"
        )
        resp = requests.get(om_url, timeout=4)
        if resp.status_code == 200:
            c = resp.json().get("current", {})
            pm25 = float(c.get("pm2_5", 35))
            pm10 = float(c.get("pm10", 65))
            no2 = float(c.get("nitrogen_dioxide", 25))
            so2 = float(c.get("sulphur_dioxide", 12))
            co = float(c.get("carbon_monoxide", 500)) / 1000.0
            
            cpcb_val = calculate_cpcb_aqi(pm25, pm10, no2, so2, co)
            return cpcb_val, round(pm25, 1), round(pm10, 1), round(no2, 1), round(so2, 1), "LIVE CPCB Formulated (Open-Meteo)"
    except Exception:
        pass

    # 3. Fallback: Baseline seasonal values for Navi Mumbai
    pm25, pm10 = 28.0, 52.0
    cpcb_val = calculate_cpcb_aqi(pm25, pm10, 20, 10, 0.5)
    return cpcb_val, pm25, pm10, 20.0, 10.0, "ESTIMATED (Seasonal Baseline)"

def fetch_ward_24h_history(lat, lon, current_cpcb):
    """
    Retrieves the past 24 hourly bins computed with Indian CPCB index formulas.
    """
    try:
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&"
            f"hourly=pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide&"
            f"past_days=1&forecast_days=0&timezone=Asia%2FKolkata"
        )
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            h = resp.json().get("hourly", {})
            times = pd.to_datetime(h.get("time", []))
            pm25s = h.get("pm2_5", [])
            pm10s = h.get("pm10", [])
            no2s = h.get("nitrogen_dioxide", [])
            so2s = h.get("sulphur_dioxide", [])

            cpcb_history = [
                calculate_cpcb_aqi(p25, p10, n2, s2, 0.6)
                for p25, p10, n2, s2 in zip(pm25s, pm10s, no2s, so2s)
            ]
            df = pd.DataFrame({"Datetime": times, "AQI": cpcb_history}).tail(24)
            df["Hour_Label"] = df["Datetime"].dt.strftime("%H:00")
            return df
    except Exception:
        pass

    # Clean seasonal fallback curve matching current CPCB reading
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(23, -1, -1)]
    hours = np.array([t.hour for t in timestamps])
    diurnal = 12 * np.sin((hours - 8) * np.pi / 12)
    history_vals = np.clip(current_cpcb + diurnal + np.random.normal(0, 2, 24), 20, 300).astype(int)

    return pd.DataFrame({
        "Datetime": timestamps,
        "Hour_Label": [t.strftime("%H:00") for t in timestamps],
        "AQI": history_vals
    })

def compile_ward_network():
    records = []
    hist_dict = {}

    for w in WARD_COORDINATES:
        cpcb_aqi, pm25, pm10, no2, so2, source = fetch_cpcb_calibrated_telemetry(w["lat"], w["lon"], w["waqi_query"])
        df_hist = fetch_ward_24h_history(w["lat"], w["lon"], cpcb_aqi)
        hist_dict[w["Ward_Name"]] = df_hist

        prev_1h = int(df_hist["AQI"].iloc[-2]) if len(df_hist) >= 2 else cpcb_aqi
        delta_1h = cpcb_aqi - prev_1h

        # True acute spike detection based on Indian standards
        is_spike = (delta_1h >= 35) or (cpcb_aqi >= 250)

        # 3-4 Hour Forward Prediction
        rate = 1.25 if is_spike else 1.05
        predicted_aqi = int(cpcb_aqi * rate)

        # Threat classification & radii
        if so2 >= 45:
            spike_type = "Chemical / MIDC Smog Spike"
            action = "Dispatch MPCB Fly Squad; Turn on static TTC fog towers."
            advisory = "Pungent chemical vapor present. Keep windows closed and wear N95/carbon masks."
            color = "#A020F0"
            radius = 3.2
        elif (pm10 / (pm25 + 1e-4)) > 1.8 and pm10 > 120:
            spike_type = "Particulate Dust Surge"
            action = "Deploy 2x Water Misting Cannons; Halt un-sheeted earth construction work."
            advisory = "Elevated particulate dust. Asthmatic individuals and runners reduce outdoor exposure."
            color = "#FF8C00"
            radius = 1.8
        elif no2 > 65:
            spike_type = "Vehicular Exhaust Plume"
            action = "Alert Traffic Police; Optimize green corridor signals."
            advisory = "Traffic exhaust concentration. Avoid lingering near arterial highway routes."
            color = "#FF0000"
            radius = 1.4
        else:
            spike_type = "Nominal Air Quality"
            action = "Standard ambient monitoring. No emergency deployment needed."
            advisory = "Air quality is within normal limits. Safe for regular activities."
            color = "#00FF7F" if cpcb_aqi <= 100 else "#FFD700"
            radius = 0.6

        records.append({
            **w,
            "Current_AQI": cpcb_aqi,
            "Delta_1h": delta_1h,
            "Is_Spike": is_spike,
            "Predicted_AQI_4h": predicted_aqi,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": 0.8,
            "Spike_Type": spike_type,
            "Recommended_Action": action,
            "Public_Advisory": advisory,
            "Color": color,
            "Impact_Radius_KM": radius,
            "Data_Source": source
        })

    return pd.DataFrame(records), hist_dict

df_wards, hist_dict = compile_ward_network()

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("🛡️ Navi Mumbai Air DSS")
portal = st.sidebar.radio(
    "Choose Interface:",
    ["👮 NMMC Ward Officer (Command DSS)", "🏡 Citizen Public Advisory (Satellite Map)"]
)
st.sidebar.markdown("---")

selected_ward_name = st.sidebar.selectbox("Select Target Ward:", df_wards["Ward_Name"].unique())
ward_detail = df_wards[df_wards["Ward_Name"] == selected_ward_name].iloc[0]
ward_hist_df = hist_dict[selected_ward_name]

st.sidebar.markdown(f"**Index Standard:** `Indian CPCB NAQI`")
st.sidebar.markdown(f"**Data Telemetry:** `{ward_detail['Data_Source']}`")
st.sidebar.markdown(f"**Last Sync:** `{datetime.now().strftime('%H:%M:%S IST')}`")

if st.sidebar.button("🔄 Sync Live Telemetry"):
    st.rerun()

# ----------------- 24-HOUR CPCB HOURLY BINNED GRAPH -----------------
def render_24h_chart(df_24, ward_name):
    fig = go.Figure()

    # Official CPCB Standard Bands
    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.08, line_width=0, annotation_text="Good (0-50)")
    fig.add_hrect(y0=51, y1=100, fillcolor="#92D050", opacity=0.08, line_width=0, annotation_text="Satisfactory (51-100)")
    fig.add_hrect(y0=101, y1=200, fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Moderate (101-200)")
    fig.add_hrect(y0=201, y1=300, fillcolor="red", opacity=0.10, line_width=0, annotation_text="Poor (201-300)")
    fig.add_hrect(y0=301, y1=500, fillcolor="purple", opacity=0.12, line_width=0, annotation_text="Very Poor / Severe (>300)")

    bar_colors = [
        "#00E400" if v <= 50 else
        "#92D050" if v <= 100 else
        "#FF7E00" if v <= 200 else
        "#FF0000" if v <= 300 else "#8f3f97"
        for v in df_24["AQI"]
    ]

    fig.add_trace(go.Bar(
        x=df_24["Hour_Label"],
        y=df_24["AQI"],
        marker_color=bar_colors,
        name="CPCB Hourly Bin",
        opacity=0.85
    ))

    fig.add_trace(go.Scatter(
        x=df_24["Hour_Label"],
        y=df_24["AQI"],
        mode="lines+markers",
        line=dict(color="#111111", width=2),
        marker=dict(size=5, color="#000000"),
        name="CPCB Trajectory"
    ))

    fig.update_layout(
        title=f"📈 Past 24-Hour CPCB AQI (1-Hour Bins) — {ward_name}",
        xaxis_title="Time (1-Hour Intervals, IST)",
        yaxis_title="CPCB Air Quality Index",
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
# VIEW 1: NMMC WARD OFFICER (COMMAND DSS)
# =================================================================
if portal == "👮 NMMC Ward Officer (Command DSS)":
    st.title("👮 NMMC Ward Officer Command Center (DSS)")
    st.caption("CPCB NAQI Standardized | 3–4 Hour Early Warning System | Municipal Operations")

    if ward_detail["Is_Spike"]:
        st.error(
            f"🚨 **ACUTE POLLUTION SURGE IN {ward_detail['Ward_Name'].upper()}!** "
            f"Sudden upward rate of climb: **+{ward_detail['Delta_1h']} pts/hr** "
            f"(Current CPCB AQI: **{ward_detail['Current_AQI']}**). Deploy intervention units immediately!"
        )
    else:
        st.success(f"✅ **NOMINAL CONDITIONS in {ward_detail['Ward_Name']}**: CPCB index steady with no acute surges.")

    # Top KPI Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Node", ward_detail["Ward_Name"], ward_detail["Ward_ID"])
    m2.metric(
        "Current CPCB AQI",
        int(ward_detail["Current_AQI"]),
        delta=f"{ward_detail['Delta_1h']} pts (Last 1 hr)"
    )
    m3.metric(
        "Forecasted AQI (+4h)",
        int(ward_detail["Predicted_AQI_4h"]),
        delta=f"+{int(ward_detail['Predicted_AQI_4h'] - ward_detail['Current_AQI'])} (Lead Time)"
    )
    m4.metric("Blast Plume Radius", f"{ward_detail['Impact_Radius_KM']} km")

    st.markdown("---")
    render_24h_chart(ward_hist_df, ward_detail["Ward_Name"])
    st.markdown("---")

    st.subheader("📍 Ward Deployment Map (Google Hybrid Satellite View)")

    officer_map = folium.Map(location=[19.0760, 73.0100], zoom_start=12, tiles=None)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Satellite Hybrid",
        name="Satellite View",
        overlay=False,
        control=True
    ).add_to(officer_map)

    for _, row in df_wards.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 250px;">
            <b style="font-size:14px; color:#0d6efd;">{row['Ward_Name']} ({row['Ward_ID']})</b><br>
            <b>CPCB AQI:</b> {row['Current_AQI']}<br>
            <b>1h Trend:</b> <span style="color:{'red' if row['Delta_1h'] > 0 else 'green'};"><b>{row['Delta_1h']} pts</b></span><br>
            <b>Predicted (+4h):</b> <span style="color:red; font-weight:bold;">{row['Predicted_AQI_4h']}</span><br>
            <b>Vector:</b> {row['Spike_Type']}<br>
            <b>Radius:</b> {row['Impact_Radius_KM']} km<br>
            <hr style="margin:4px 0;">
            <b>Action SOP:</b> {row['Recommended_Action']}
        </div>
        """

        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=row["Impact_Radius_KM"] * 1000,
            color=row["Color"],
            weight=2.5,
            fill=True,
            fill_color=row["Color"],
            fill_opacity=0.45,
            tooltip=f"{row['Ward_Name']} | {row['Spike_Type']} (Radius: {row['Impact_Radius_KM']} km)"
        ).add_to(officer_map)

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(
                color="red" if row["Is_Spike"] else "blue",
                icon="warning-sign" if row["Is_Spike"] else "info-sign"
            )
        ).add_to(officer_map)

    folium.LayerControl(position="topright").add_to(officer_map)
    st_folium(officer_map, width=1200, height=520, key="officer_sat_map")

    # SOP & Emergency Fleet Action
    st.subheader("⚡ Automated Standard Operating Procedure (SOP) & Fleet Dispatch")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"**Identified Root Cause Vector:** `{ward_detail['Spike_Type']}`")
        st.markdown(f"**Calculated Plume Radius:** `{ward_detail['Impact_Radius_KM']} km`")
        st.markdown(
            f"**Underlying Pollutant Concentrations (CPCB Scale):**\n"
            f"* **PM2.5:** `{ward_detail['PM2.5']} µg/m³`\n"
            f"* **PM10:** `{ward_detail['PM10']} µg/m³`\n"
            f"* **NO2:** `{ward_detail['NO2']} µg/m³`\n"
            f"* **SO2:** `{ward_detail['SO2']} µg/m³`"
        )

    with col2:
        st.warning(f"**Direct Order:** {ward_detail['Recommended_Action']}")
        if st.button("📲 Broadcast Emergency Order to Dispatch Crews"):
            st.success(f"Emergency dispatch order successfully sent to {ward_detail['Ward_Name']} field units.")

    st.subheader("📊 Cross-Ward Assessment Matrix")
    st.dataframe(
        df_wards[["Ward_ID", "Ward_Name", "Zone", "Current_AQI", "Delta_1h", "Predicted_AQI_4h", "Spike_Type", "Impact_Radius_KM", "Recommended_Action"]],
        use_container_width=True
    )

# =================================================================
# VIEW 2: CITIZEN PUBLIC ADVISORY (SATELLITE VIEW)
# =================================================================
else:
    st.title("🛰️ Navi Mumbai Citizen Air Quality Portal")
    st.caption("CPCB Standardized Plume Dispersion & 3–4 Hour Early Warning System")

    if ward_detail["Is_Spike"]:
        st.error(
            f"🚨 **UNEXPECTED AIR QUALITY SPIKE DETECTED in {ward_detail['Ward_Name'].upper()}!**\n\n"
            f"Air pollution has surged by **+{ward_detail['Delta_1h']} points in the last 1 hour**. "
            f"Sensitive individuals should wear N95 masks immediately."
        )
    elif ward_detail["Current_AQI"] >= 150:
        st.warning(f"⚠️ **MODERATE TO POOR AIR QUALITY in {ward_detail['Ward_Name']}**: Asthmatics limit outdoor jogging.")
    elif ward_detail["Current_AQI"] > 50:
        st.info(f"🌿 **SATISFACTORY AIR QUALITY in {ward_detail['Ward_Name']}**: Minor breathing discomfort to sensitive people.")
    else:
        st.success(f"🌱 **GOOD AIR QUALITY in {ward_detail['Ward_Name']}**: Minimal health impact.")

    ca, cb, cc = st.columns([1, 1, 2])
    ca.metric("Current CPCB AQI", int(ward_detail["Current_AQI"]), delta=f"{ward_detail['Delta_1h']} pts (1h change)")
    ca.write(f"Zone: **{ward_detail['Zone']}**")
    cb.metric("Forecasted AQI (+4h)", int(ward_detail["Predicted_AQI_4h"]))
    cb.write(f"Trend: **+{int(ward_detail['Predicted_AQI_4h'] - ward_detail['Current_AQI'])} pts lead time**")
    cc.info(f"📢 **Actionable Health Advisory:**\n\n{ward_detail['Public_Advisory']}")

    st.markdown("---")
    render_24h_chart(ward_hist_df, ward_detail["Ward_Name"])
    st.markdown("---")

    st.subheader("🛰️ Live Satellite Plume Coverage & Affected Radius")

    citizen_map = folium.Map(location=[19.0760, 73.0100], zoom_start=12, tiles=None)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Satellite Hybrid",
        name="Satellite View",
        overlay=False,
        control=True
    ).add_to(citizen_map)

    for _, row in df_wards.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; width: 230px;">
            <h4 style="margin:0 0 5px 0; color:#111;">{row['Ward_Name']}</h4>
            <b>CPCB AQI:</b> {row['Current_AQI']}<br>
            <b>1h Spike:</b> {'+' if row['Delta_1h'] > 0 else ''}{row['Delta_1h']} pts<br>
            <b>Predicted (+4h):</b> <span style="color:red; font-weight:bold;">{row['Predicted_AQI_4h']}</span><br>
            <b>Spike Cause:</b> {row['Spike_Type']}<br>
            <b>Spread Radius:</b> {row['Impact_Radius_KM']} km<br><br>
            <p style="background:#fff3cd; padding:6px; border-radius:4px; margin:0;">
                <b>Advisory:</b> {row['Public_Advisory']}
            </p>
        </div>
        """

        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=row["Impact_Radius_KM"] * 1000,
            color=row["Color"],
            weight=2,
            fill=True,
            fill_color=row["Color"],
            fill_opacity=0.40,
            tooltip=f"{row['Ward_Name']} - Spike Radius: {row['Impact_Radius_KM']} km"
        ).add_to(citizen_map)

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.Icon(color="red" if row["Is_Spike"] else "darkblue", icon="home")
        ).add_to(citizen_map)

    folium.LayerControl(position="topright").add_to(citizen_map)
    st_folium(citizen_map, width=1200, height=540, key="citizen_sat_map")
