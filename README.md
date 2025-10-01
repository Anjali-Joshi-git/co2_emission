# co2_emission
This project is focused on predicting CO₂ emissions of vehicles using different machine learning models. The goal is to analyze how various factors such as engine size, fuel type, number of cylinders, and vehicle weight affect carbon emissions, and then build models to make accurate predictions.

**[CO2 Emission Predictor ] (https://co2emission-piddgpdmfyovzgpfdhmpc7.streamlit.app/)**

#  Technologies Used
- Python 🐍
- Pandas, NumPy → Data preprocessing & analysis-Matplotlib, Seaborn → Data visualization
- Scikit-learn → Machine learning models (Linear Regression, Random Forest, KNN, etc.)
- Streamlit (optional) → For interactive web app

### Features:
- Data preprocessing (handling categorical & numerical features)  
- Model training (KNN, Random Forest, Linear Regression)  
- Streamlit app for deployment  
- Visualization of results  

### Project Structure:
 ├── ce_app.py              # Streamlit app

 ├── car_emissions_model.pkl # Trained ML pipeline

 ├── requirements.txt       # Python dependencies

 ├── runtime.txt            # Python version for deployment

 ├── data/                  # Dataset (if included)

 └── README.md              # Project documentation



 ### Dataset
The dataset contains details about different vehicles:

Numerical features: Engine Size, Cylinders, Fuel Consumption (City, Hwy, Comb)
Categorical features: Make, Model, Vehicle Class, Transmission, Fuel Type
🎯 Target variable: CO₂ Emissions (g/km)  

### Deployment
The project is deployed using Streamlit Cloud.

**[Click here to try the live app ] (https://co2emission-piddgpdmfyovzgpfdhmpc7.streamlit.app/)**

