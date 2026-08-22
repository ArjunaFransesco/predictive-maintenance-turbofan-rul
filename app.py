import streamlit as st
import pandas as pd
import numpy as np
import joblib, os

st.set_page_config(page_title="Turbofan RUL Estimator", page_icon="✈️", layout="wide")
st.title("✈️ Industrial Turbofan Remaining Useful Life (RUL) Prognostics")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **Portfolio**: [GitHub](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📡 Jet Engine Telemetry Sensors")
    t24 = st.slider("T24 LPC Outlet Temp (°R)", 635.0, 665.0, 645.0)
    t30 = st.slider("T30 HPC Outlet Temp (°R)", 1570.0, 1630.0, 1595.0)
    t50 = st.slider("T50 LPT Outlet Temp (°R)", 1385.0, 1435.0, 1408.0)
    p30 = st.slider("P30 HPC Total Pressure (psia)", 540.0, 565.0, 550.0)
    nf = st.slider("Nf Fan Speed (rpm)", 2380.0, 2395.0, 2387.0)
    nc = st.slider("Nc Core Speed (rpm)", 9030.0, 9090.0, 9060.0)
    vib = st.slider("Vibration RMS (g)", 0.02, 0.15, 0.06)

with col2:
    st.subheader("🎯 Prognostics Prediction")
    model_path = os.path.join(os.path.dirname(__file__), "models/turbofan_rul_lgbm.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        raw_feats = [t24, t30, t50, p30, nf, nc, vib]
        all_feats = raw_feats + raw_feats + [0.0]*len(raw_feats)
        pred_rul = float(model.predict([all_feats])[0])
        st.metric("Predicted Remaining Useful Life (RUL)", f"{max(0, pred_rul):.1f} Cycles")
        
        if pred_rul < 30:
            st.error("🚨 CRITICAL: Immediate scheduled engine overhaul required!")
        elif pred_rul < 80:
            st.warning("⚠️ ATTENTION: Maintenance inspection recommended within next 30 flight cycles.")
        else:
            st.success("✅ OPTIMAL: Engine degradation within safe operational envelope.")
