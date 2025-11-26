import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Streamlit app title
st.title("House Price Prediction App")

st.write("""
Enter the house features below to predict the price.
""")

# Input fields
MedInc = st.number_input("Median Income (10k $)", min_value=0.0, step=0.1)
HouseAge = st.number_input("House Age (years)", min_value=0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, step=0.1)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, step=0.1)
Population = st.number_input("Population in Block", min_value=0)
AveOccup = st.number_input("Average Occupants", min_value=0.0, step=0.1)
Latitude = st.number_input("Latitude", min_value=0.0, step=0.0001)
Longitude = st.number_input("Longitude", min_value=0.0, step=0.0001)

# Button to make prediction
if st.button("Predict Price"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        "MedInc": [MedInc],
        "HouseAge": [HouseAge],
        "AveRooms": [AveRooms],
        "AveBedrms": [AveBedrms],
        "Population": [Population],
        "AveOccup": [AveOccup],
        "Latitude": [Latitude],
        "Longitude": [Longitude]
    })

    # Optional: scale features if you scaled during training
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # input_data_scaled = scaler.fit_transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"The predicted median house value is: ${prediction*100000:.2f}")
