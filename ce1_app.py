import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model and features
# -----------------------------
model, categorical_features, numeric_features = joblib.load("co2_knn_model.pkl")

st.set_page_config(page_title="CO₂ Emissions Predictor", layout="centered")

# -----------------------------
# Title
# -----------------------------
st.title("🚗 App CarbonSense")
st.markdown("Enter vehicle details below and get an estimated **CO₂ Emission (g/km)**.")

# -----------------------------
# User Inputs
# -----------------------------
st.header("🔧 Vehicle Inputs")

col1, col2 = st.columns(2)

with col1:
    make = st.text_input("Make", "Toyota") or "Unknown"
    model_name = st.text_input("Model", "Corolla") or "Unknown"
    vehicle_class = st.selectbox("Vehicle Class", [
        "SUV - Small", "SUV - Standard", "Compact", "Mid-size", "Full-size",
        "Two-seater", "Station wagon - Small", "Station wagon - Mid-size",
        "Pickup truck - Small", "Pickup truck - Standard", "Minivan",
        "Van - Passenger", "Van - Cargo", "Special purpose vehicle",
        "Luxury", "Sports"
    ])
    transmission = st.selectbox("Transmission", ['AS', 'M', 'AV', 'AM', 'A'])
    fuel_type = st.selectbox("Fuel Type", ['Z', 'D', 'X', 'E', 'N'])

with col2:
    engine_size = float(st.number_input("Engine Size (L)", min_value=1.0, max_value=10.0, step=0.1))
    cylinders = int(st.number_input("Cylinders", min_value=2, max_value=16, step=1))
    fuel_city = float(st.number_input("Fuel Consumption City (L/100km)", min_value=1.0, max_value=30.0, step=0.1))
    fuel_hwy = float(st.number_input("Fuel Consumption Hwy (L/100km)", min_value=1.0, max_value=25.0, step=0.1))
    fuel_comb_l = float(st.number_input("Fuel Consumption Comb (L/100km)", min_value=1.0, max_value=25.0, step=0.1))
    fuel_comb_mpg = float(st.number_input("Fuel Consumption Comb (mpg)", min_value=1.0, max_value=100.0, step=0.1))

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "make": [str(make)],
    "model": [str(model_name)],
    "vehicle_class": [str(vehicle_class)],
    "transmission": [str(transmission)],
    "fuel_type": [str(fuel_type)],
    "engine_size": [engine_size],
    "cylinders": [cylinders],
    "fuel_consumption_city": [fuel_city],
    "fuel_consumption_hwy": [fuel_hwy],
    "fuel_consumption_comb(l/100km)": [fuel_comb_l],
    "fuel_consumption_comb(mpg)": [fuel_comb_mpg]
})

# Reorder columns to match training order
input_data = input_data[categorical_features + numeric_features]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict CO₂ Emissions"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated CO₂ Emissions: **{prediction:.2f} g/km**")

        # -----------------------------
        # Visualization Section
        # -----------------------------
        st.header("📊 Visualization")

# Select only numeric columns for plotting
        numeric_only = input_data.select_dtypes(include=['float64', 'int64'])

        fig, ax = plt.subplots()
        ax.bar(numeric_only.columns, numeric_only.values[0])
        ax.set_title("Entered Numeric Features")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("🔍 Input Data:", input_data)