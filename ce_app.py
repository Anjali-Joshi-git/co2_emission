import streamlit as st
import cloudpickle
import numpy as np

# Load model with cloudpickle
with open("car_emissions_model.pkl", "rb") as f:
    model = cloudpickle.load(f)

st.title("🌍 Car CO₂ Emissions Prediction")

st.write("Enter vehicle details below to estimate CO₂ emissions:")

# Example inputs (adapt based on your dataset features)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1)
fuel_city = st.number_input("Fuel Consumption City (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_comb = st.number_input("Fuel Consumption Combined (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)

# Create feature array
features = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Estimated CO₂ Emission: {prediction[0]:.2f} g/km")