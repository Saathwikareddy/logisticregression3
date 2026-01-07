import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- CSS ----------------
st.markdown("""
<style>
/* Gradient background */
.stApp {
    background: linear-gradient(135deg, #1e1e23, #79737f);
    padding: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #e0e0e0;
    margin-bottom: 40px;
}

/* Card */
.card {
    background-color: #ffffff;
    padding: 35px;
    border-radius: 18px;
    max-width: 480px;
    margin: auto;
    box-shadow: 0px 12px 25px rgba(0,0,0,0.15);
}

/* Button */
.stButton > button {
    background-color: #667eea;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    height: 48px;
    width: 100%;
    border: none;
}

/* Result boxes */
.good {
    background-color: #e6f4ea;
    color: #137333;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    margin-top: 20px;
}

.bad {
    background-color: #fdecea;
    color: #a50e0e;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    margin-top: 20px;
}
</style>

""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown('<div class="title">📊 Telecom Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a customer is likely to churn</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges (₹)", min_value=0.0, value=900.0)

if st.button("Predict Churn"):
    input_data = np.array([[tenure, monthly_charges, total_charges]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.write(f"### 🔍 Churn Probability: **{probability:.2f}**")

    if prediction == 1:
        st.markdown('<div class="bad">⚠️ Customer is LIKELY TO CHURN</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good">✅ Customer is LIKELY TO STAY</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
