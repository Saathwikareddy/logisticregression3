import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Telecom Customer Churn Prediction")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=900.0)

if st.button("Predict"):
    data = np.array([[tenure, monthly_charges, total_charges]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.write("Churn Probability:", round(probability, 2))

    if prediction == 1:
        st.error("Customer is LIKELY TO CHURN")
    else:
        st.success("Customer is LIKELY TO STAY")
