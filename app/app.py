import os
import sys
import joblib
import pandas as pd
import streamlit as st

# allow importing from app folder
sys.path.append(os.path.dirname(__file__))

from feature_builder import build_demo_feature_row

st.set_page_config(page_title="AKI Risk Predictor", layout="centered")

st.title("AKI Risk Prediction Demo")
st.write("Predict the risk of Acute Kidney Injury within the next 24 hours using ICU patient measurements.")

# load model
MODEL_PATH = os.path.join("models", "hgb_aki.joblib")
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
features = model_data["features"]

st.subheader("Patient Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=65)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.2)
    bun = st.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=20.0)
    map_val = st.number_input("MAP (mmHg)", min_value=20.0, max_value=200.0, value=75.0)
    sbp = st.number_input("SBP (mmHg)", min_value=40.0, max_value=250.0, value=110.0)

with col2:
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=43.0, value=37.0)
    lactate = st.number_input("Lactate", min_value=0.1, max_value=20.0, value=1.5)
    wbc = st.number_input("WBC", min_value=0.1, max_value=100.0, value=8.0)
    hr = st.number_input("Heart Rate", min_value=20.0, max_value=250.0, value=90.0)
    iculos = st.number_input("ICU Length of Stay (hours)", min_value=1.0, max_value=1000.0, value=24.0)
    hosp_adm_time = st.number_input("Hours from Hospital Admission to ICU", min_value=-500.0, max_value=500.0, value=-12.0)

predict_btn = st.button("Predict AKI Risk")

if predict_btn:
    user_input = {
        "Age": age,
        "Gender": gender,
        "Creatinine": creatinine,
        "BUN": bun,
        "MAP": map_val,
        "SBP": sbp,
        "Temp": temp,
        "Lactate": lactate,
        "WBC": wbc,
        "HR": hr,
        "ICULOS": iculos,
        "HospAdmTime": hosp_adm_time,

        # optional defaults for missing fields used in training
        "O2Sat": 98.0,
        "DBP": 70.0,
        "Resp": 18.0,
        "Glucose": 110.0,
        "Platelets": 250.0,
        "Hgb": 13.0,
        "Hct": 39.0,
        "Potassium": 4.0,
        "Chloride": 102.0,
        "Calcium": 9.0,
        "Magnesium": 2.0,
        "Phosphate": 3.5,
        "Bilirubin_total": 0.8,
        "AST": 25.0,
        "EtCO2": None,
    }

    X_demo = build_demo_feature_row(user_input, features)
    risk = model.predict_proba(X_demo)[0, 1]

    if risk < 0.05:
        band = "LOW"
        color = "green"
    elif risk < 0.20:
        band = "MEDIUM"
        color = "orange"
    else:
        band = "HIGH"
        color = "red"

    st.subheader("Prediction Result")
    st.metric("AKI Risk (next 24h)", f"{risk:.1%}")
    st.markdown(f"**Risk Band:** :{color}[{band}]")

    st.subheader("Clinical Summary")
    explanations = []

    if creatinine > 1.5:
        explanations.append("- Elevated creatinine may indicate impaired kidney function.")
    if bun > 25:
        explanations.append("- High BUN may reflect renal stress or dehydration.")
    if map_val < 65:
        explanations.append("- Low MAP may reduce kidney perfusion and increase AKI risk.")
    if lactate > 2:
        explanations.append("- Elevated lactate may suggest systemic illness or poor perfusion.")
    if temp > 38 or temp < 36:
        explanations.append("- Abnormal temperature may indicate infection or systemic instability.")
    if age > 70:
        explanations.append("- Older age is associated with increased AKI vulnerability.")

    if not explanations:
        explanations.append("- Current values do not show strong high-risk warning patterns.")

    for item in explanations:
        st.write(item)

    with st.expander("Show feature row sent to model"):
        st.dataframe(X_demo)