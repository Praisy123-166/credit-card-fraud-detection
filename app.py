import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🔒")

st.title("🔒 Credit Card Fraud Detection")
st.markdown("**AI-powered fraud detection model trained on 284,807 real transactions**")
st.markdown("---")

st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (€)", min_value=0.0, value=100.0)
    v1 = st.slider("V1", -5.0, 5.0, 0.0)
    v2 = st.slider("V2", -5.0, 5.0, 0.0)
    v3 = st.slider("V3", -5.0, 5.0, 0.0)
    v4 = st.slider("V4", -5.0, 5.0, 0.0)
    v5 = st.slider("V5", -5.0, 5.0, 0.0)
    v6 = st.slider("V6", -5.0, 5.0, 0.0)
    v7 = st.slider("V7", -5.0, 5.0, 0.0)
    v8 = st.slider("V8", -5.0, 5.0, 0.0)
    v9 = st.slider("V9", -5.0, 5.0, 0.0)
    v10 = st.slider("V10", -5.0, 5.0, 0.0)
    v11 = st.slider("V11", -5.0, 5.0, 0.0)
    v12 = st.slider("V12", -5.0, 5.0, 0.0)
    v13 = st.slider("V13", -5.0, 5.0, 0.0)
    v14 = st.slider("V14", -5.0, 5.0, 0.0)

with col2:
    v15 = st.slider("V15", -5.0, 5.0, 0.0)
    v16 = st.slider("V16", -5.0, 5.0, 0.0)
    v17 = st.slider("V17", -5.0, 5.0, 0.0)
    v18 = st.slider("V18", -5.0, 5.0, 0.0)
    v19 = st.slider("V19", -5.0, 5.0, 0.0)
    v20 = st.slider("V20", -5.0, 5.0, 0.0)
    v21 = st.slider("V21", -5.0, 5.0, 0.0)
    v22 = st.slider("V22", -5.0, 5.0, 0.0)
    v23 = st.slider("V23", -5.0, 5.0, 0.0)
    v24 = st.slider("V24", -5.0, 5.0, 0.0)
    v25 = st.slider("V25", -5.0, 5.0, 0.0)
    v26 = st.slider("V26", -5.0, 5.0, 0.0)
    v27 = st.slider("V27", -5.0, 5.0, 0.0)
    v28 = st.slider("V28", -5.0, 5.0, 0.0)

st.markdown("---")

if st.button("🔍 Detect Fraud", use_container_width=True):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    amount_scaled = scaler.fit_transform([[amount]])[0][0]
    
    features = np.array([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
                          v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
                          v21,v22,v23,v24,v25,v26,v27,v28,amount_scaled]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — {probability*100:.1f}% confidence")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION — {(1-probability)*100:.1f}% confidence")
    
    st.info(f"Model AUC Score: 95.28% | Trained on 284,807 real transactions")

st.markdown("---")
st.caption("Built by Ashitha Praisy Aluri | Random Forest Classifier | Kaggle Credit Card Fraud Dataset")
