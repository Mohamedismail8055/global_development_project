# above line is to write in app.py file created by user

import pickle
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained KMeans model and scaler
try:
    kmeans_model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

st.title('KMeans Clustering Prediction')

# Input fields in the Streamlit app
features = [
    "Birth Rate", "CO2 Emissions", "Energy Usage", "GDP",
    "Health Exp/Capita", "Infant Mortality Rate", "Internet Usage",
    "Life Expectancy Female", "Life Expectancy Male", "Mobile Phone Usage",
    "Population 0-14", "Population 15-64", "Population 65+",
    "Tourism Inbound", "Tourism Outbound"
]

input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}", min_value=0.0, value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Validate inputs
if input_df.isnull().values.any():
    st.error("Please fill in all the required fields.")
else:
    try:
        # Scale the input data
        scaled_input_data = scaler.transform(input_df)

        # Make predictions
        prediction = kmeans_model.predict(scaled_input_data)

        # Display the prediction
        st.success(f"Predicted Cluster: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
