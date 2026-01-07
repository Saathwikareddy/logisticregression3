import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
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

# ---------------- UI HEADER ----------------
st.markdown('<div class="title">📊 Telco Customer Churn Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning powered churn analytics</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(""C:\Users\saath\Downloads\WA_Fn-UseC_-Telco-Customer-Churn (2).csv"")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = (df['Churn'].str.strip() == 'Yes').astype(int)

# ---------------- KPI METRICS ----------------
total = len(df)
churned = df['Churn'].sum()
stayed = total - churned
churn_rate = churned / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", total)
c2.metric("Churned", churned)
c3.metric("Stayed", stayed)
c4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.markdown("---")

# ---------------- CUSTOMER DISTRIBUTION ----------------
st.subheader("Customer Distribution")

fig1, ax1 = plt.subplots()
ax1.bar(["Staying", "Leaving"], [stayed, churned])
ax1.set_ylabel("Customers")
st.pyplot(fig1)

# ---------------- MODEL PERFORMANCE ----------------
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

accuracy = (y_pred == y).mean() * 100
st.subheader("Model Performance")
st.write(f"Accuracy: **{accuracy:.2f}%**")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("Model Evaluation")

cm = confusion_matrix(y, y_pred)
fig2, ax2 = plt.subplots()
im = ax2.imshow(cm)

for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm[i, j], ha="center", va="center")

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_xticks([0,1])
ax2.set_yticks([0,1])
ax2.set_xticklabels(["Stay", "Leave"])
ax2.set_yticklabels(["Stay", "Leave"])
st.pyplot(fig2)

# ---------------- ROC CURVE ----------------
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax3.plot([0,1], [0,1], linestyle="--")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()
st.pyplot(fig3)

st.markdown("---")

# ---------------- PREDICT CUSTOMER CHURN ----------------
st.subheader("Predict Customer Churn")

tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 900.0)

if st.button("Predict Churn"):
    data = np.array([[tenure, monthly_charges, total_charges]])
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    st.write(f"Churn Probability: **{prob:.2f}**")

    if pred == 1:
        st.markdown('<div class="bad">⚠️ Customer is LIKELY TO CHURN</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good">✅ Customer is LIKELY TO STAY</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
