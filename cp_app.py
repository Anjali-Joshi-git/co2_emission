import streamlit as st
import joblib
import pandas as pd

# ---------------------------
# Load the trained pipeline
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("car_emissions_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🚗 CO₂ Emission Predictor", layout="wide")

st.title("🚗 Car CO₂ Emission Prediction App")
st.write("Enter car details to predict **CO₂ emissions** using the trained ML model.")

if model is not None:
    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        engine_size = st.slider("Engine Size (L)", 1.0, 7.0, 3.0, 0.1)
        cylinders = st.slider("Cylinders", 3, 16, 4, 1)
        fuel_consumption_city = st.slider("Fuel Consumption City (L/100 km)", 2.0, 30.0, 10.0, 0.5)
    
    with col2:
        fuel_consumption_hwy = st.slider("Fuel Consumption Hwy (L/100 km)", 2.0, 20.0, 7.0, 0.5)
        fuel_consumption_comb = st.slider("Fuel Consumption Comb (L/100 km)", 2.0, 25.0, 8.5, 0.5)
        fuel_type = st.selectbox("Fuel Type", ["X", "Z", "E", "D"])  # change based on your dataset
    
    # Input DataFrame
    input_data = pd.DataFrame({
        "Engine Size(L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel Consumption City (L/100 km)": [fuel_consumption_city],
        "Fuel Consumption Hwy (L/100 km)": [fuel_consumption_hwy],
        "Fuel Consumption Comb (L/100 km)": [fuel_consumption_comb],
        "Fuel Type": [fuel_type]
    })

    # Prediction
    if st.button("🔮 Predict CO₂ Emission"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated CO₂ Emission: **{prediction:.2f} g/km**")
else:
    st.warning("⚠️ Model not loaded. Please ensure 'car_emissions_model.pkl' exists in the project folder.")