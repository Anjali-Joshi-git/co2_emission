import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("car_emissions_model.pkl")

st.title("🚗App CarbonSense")

st.write("Provide vehicle details below to estimate CO₂ emissions:")

# --- Categorical Inputs ---
make = st.text_input("Make (e.g., Toyota, Honda, Ford)")
model_name = st.text_input("Model (e.g., Corolla, Civic, Focus)")
#vehicle_class = st.selectbox("Vehicle Class", ["SUV", "Sedan", "Truck", "Compact", "Other"])
vehicle_class = st.selectbox("Vehicle Class", [
        "SUV - Small", "SUV - Standard", "Compact", "Mid-size", "Full-size",
        "Two-seater", "Station wagon - Small", "Station wagon - Mid-size",
        "Pickup truck - Small", "Pickup truck - Standard", "Minivan",
        "Van - Passenger", "Van - Cargo", "Special purpose vehicle",
        "Luxury", "Sports"
    ])
#transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Other"])
transmission = st.selectbox("Transmission", ['AS', 'M', 'AV', 'AM', 'A'])
fuel_type = st.selectbox("Fuel Type", ['Z', 'D', 'X', 'E', 'N'])
#fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

# --- Numeric Inputs ---
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1)
fuel_city = st.number_input("Fuel Consumption City (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_comb_l = st.number_input("Fuel Consumption Combined (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_comb_mpg = st.number_input("Fuel Consumption Combined (mpg)", min_value=1.0, max_value=100.0, step=0.1)

# --- Build dataframe with ALL 11 features ---
input_df = pd.DataFrame([{
    "make": make,
    "model": model_name,
    "vehicle_class": vehicle_class,
    "engine_size": engine_size,
    "cylinders": cylinders,
    "transmission": transmission,
    "fuel_type": fuel_type,
    "fuel_consumption_city": fuel_city,
    "fuel_consumption_hwy": fuel_hwy,
    "fuel_consumption_comb(l/100km)": fuel_comb_l,
    "fuel_consumption_comb(mpg)": fuel_comb_mpg
}])

# --- Prediction ---
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Estimated CO₂ Emission: {prediction[0]:.2f} g/km")
    except Exception as e:
        st.error(f"❌ Error: {e}")