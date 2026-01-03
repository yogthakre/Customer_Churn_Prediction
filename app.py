import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn")

# Input fields (example basic ones)
age = st.number_input("Age", min_value=18, max_value=100)
tenure = st.number_input("Tenure in Months", min_value=0)
monthly_charge = st.number_input("Monthly Charge")
total_charges = st.number_input("Total Charges")

# Predict button
if st.button("Predict Churn"):
    input_data = np.array([[age, tenure, monthly_charge, total_charges]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to CHURN ❌")
    else:
        st.success("Customer is NOT likely to churn ✅")
