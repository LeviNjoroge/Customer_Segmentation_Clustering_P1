import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer segmentation app")
st.write("Enter customer details")

Age = st.number_input("Age", min_value=0, max_value=100, value=25)
Income = st.number_input("Income", min_value=0, max_value=999999, value=5000)
NumWebPurchases = st.number_input(
    "Num of Web Purchases", min_value=0, max_value=999, value=10)
NumStorePurchases = st.number_input(
    "Num Store Purchases", min_value=0, max_value=999, value=10)
NumWebVisitsMonth = st.number_input(
    "Num Web Visits Month", min_value=0, max_value=999, value=10)
total_spending = st.number_input(
    "Total Spending (Total amount spent)", min_value=0, max_value=99999, value=100)
recency = st.number_input("Recency", min_value=0, max_value=365, value=10)

input_data = pd.DataFrame({
    "Age": [Age],
    "Income": [Income],
    "NumWebPurchases": [NumWebPurchases],
    "NumStorePurchases": [NumStorePurchases],
    "NumWebVisitsMonth": [NumWebVisitsMonth],
    "total_spending": [total_spending],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment : Cluster {cluster}")
