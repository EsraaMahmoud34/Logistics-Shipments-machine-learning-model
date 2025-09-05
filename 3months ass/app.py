import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ==============================
# Page Config & Theme
# ==============================
st.set_page_config(
    page_title="Shipment Status Prediction",
    page_icon="📦",
    layout="centered"
)

# Custom Green Theme CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #2e7d32;  /* Green */
        --secondary-color: #66bb6a;
        --text-color-light: #ffffff;
        --text-color-dark: #000000;
    }

    /* Main Background */
    .stApp {
        background: var(--bg-color, #f0fdf4);
    }

    /* Titles */
    h1, h2, h3 {
        color: var(--primary-color) !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: var(--secondary-color);
        color: black;
    }

    /* Result Card */
    .result-box {
        background: linear-gradient(135deg, #a5d6a7, #c8e6c9);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #1b5e20;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load trained pipeline
# ==============================
model = joblib.load("xgb_model.pkl")

st.title("📦 Shipment Status Prediction")
st.markdown("🚚 *Fill in the shipment details to predict its status.*")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("🔧 Input Shipment Details")

origin = st.sidebar.text_input("🏭 Origin Warehouse")
destination = st.sidebar.text_input("🏬 Destination")
carrier = st.sidebar.text_input("🚛 Carrier")
ship_date = st.sidebar.date_input("📅 Shipment Date", date.today())
delivery_date = st.sidebar.date_input("📦 Delivery Date (optional)", date.today())
weight = st.sidebar.number_input("⚖️ Weight (kg)", min_value=0.0)
cost = st.sidebar.number_input("💲 Cost", min_value=0.0)
distance = st.sidebar.number_input("🛣️ Distance (miles)", min_value=0)
transit_days = st.sidebar.number_input("⏳ Transit Days", min_value=0)

# ==============================
# Prediction Button
# ==============================
if st.sidebar.button("✨ Predict Status"):
    # Convert to dataframe
    input_data = pd.DataFrame([{
        "Origin_Warehouse": origin,
        "Destination": destination,
        "Carrier": carrier,
        "Shipment_Date": pd.to_datetime(ship_date),
        "Delivery_Date": pd.to_datetime(delivery_date),
        "Weight_kg": weight,
        "Cost": cost,
        "Distance_miles": distance,
        "Transit_Days": transit_days
    }])

    # Feature Engineering (same as training)
    input_data["Planned_Days"] = (input_data["Delivery_Date"] - input_data["Shipment_Date"]).dt.days
    input_data["Ship_Day"] = input_data["Shipment_Date"].dt.day
    input_data["Ship_Month"] = input_data["Shipment_Date"].dt.month
    input_data["Ship_Year"] = input_data["Shipment_Date"].dt.year
    input_data = input_data.drop(columns=["Shipment_Date", "Delivery_Date"])

    # Predict
    pred = model.predict(input_data)[0]

    # Map prediction
    status_map = {0: "✅ Delivered", 1: "⚠️ Problematic"}
    pred_label = status_map.get(pred, f"Unknown ({pred})")

    # Show result in styled box
    st.markdown(f"""
        <div class="result-box">
            📊 Predicted Shipment Status: <br> <span style="font-size:28px;">{pred_label}</span>
        </div>
    """, unsafe_allow_html=True)
