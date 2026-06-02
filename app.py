import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🔒")

st.title("🔒 Credit Card Fraud Detection")
st.markdown("**AI-powered fraud detection model trained on real credit card transactions**")
st.markdown("---")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount (€)", min_value=0.0, value=100.0)

values = []

col1, col2 = st.columns(2)

with col1:
    for i in range(1, 15):
        values.append(st.slider(f"V{i}", -5.0, 5.0, 0.0))

with col2:
    for i in range(15, 29):
        values.append(st.slider(f"V{i}", -5.0, 5.0, 0.0))

st.markdown("---")

if st.button("🔍 Detect Fraud", use_container_width=True):
    amount_scaled = amount

    features = np.array([values + [amount_scaled]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — {probability * 100:.1f}% confidence")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION — {(1 - probability) * 100:.1f}% confidence")

    st.info("Model AUC Score: 95.28% | Random Forest Classifier")

st.markdown("---")
st.caption("Built by Ashitha Praisy Aluri | Python | Streamlit | Scikit-learn")
