import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load trained model & scaler
# -------------------------
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# Streamlit App Title
# -------------------------
st.title("California House Price Prediction App")
st.write("Enter the house features below to predict the median house value.")

# -------------------------
# Input Fields
# -------------------------
MedInc = st.number_input("Median Income (10k $)", min_value=0.0, step=0.1)
HouseAge = st.number_input("House Age (years)", min_value=0, step=1)
AveRooms = st.number_input("Average Rooms", min_value=0.0, step=0.1)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, step=0.1)
Population = st.number_input("Population in Block", min_value=0, step=1)
AveOccup = st.number_input("Average Occupants", min_value=0.0, step=0.1)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.0001)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, step=0.0001)

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict Price"):
    # Create DataFrame from user input
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

    # Scale input data using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Ensure prediction is not negative
    prediction = max(prediction, 0)

    # Display result
    st.success(f"The predicted median house value is: ${prediction*100000:.2f}")
