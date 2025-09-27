# --- frontend/app.py ---
# This is our Streamlit application for the frontend.

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="⚡",
    layout="centered"
)

# --- API URL ---
# This is the URL where our FastAPI backend is running.
# For local testing, it's localhost. For deployment, it will be the Render URL.
API_URL = "http://127.0.0.1:8000/predict"

# --- Application Header ---
st.title("⚡ Customer Churn Predictor")
st.markdown("""
This app demonstrates a full-stack AI application. 
Enter the customer's details below, and the backend API will return a churn prediction.
""")

# --- Input Form ---
st.header("Enter Customer Data")

# We use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.7, format="%.2f")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=70.7, format="%.2f")
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

# A dictionary to hold all the other default values needed by the API
# In a real app, you would have form fields for all of these.
default_values = {
    "gender": "Female",
    "Partner": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": "Yes"
}

# --- Prediction Logic ---
if st.button("Predict Churn", type="primary"):
    # Create the payload dictionary for the API request
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "SeniorCitizen": senior_citizen,
        "TotalCharges": total_charges,
        "PaymentMethod": payment_method,
        "Dependents": dependents,
        **default_values # A cool Python trick to merge dictionaries
    }

    # Display a spinner while waiting for the API response
    with st.spinner("Sending data to the model... Please wait."):
        try:
            # Send the POST request to our FastAPI backend
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the JSON response
            result = response.json()
            
            # Display the result
            st.subheader("Prediction Result")
            prediction = result['prediction']
            probability = result['probability_of_churn']
            
            if prediction == "Yes":
                st.error(f"Prediction: This customer is likely to CHURN (Probability: {probability:.2%})")
            else:
                st.success(f"Prediction: This customer is likely to STAY (Probability of churn: {probability:.2%})")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")