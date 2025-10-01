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
- `ce_train.py` → Training script  
- `ce1_app.py` → Streamlit app  
- `co2_emissions.csv` → Dataset  
- `requirements.txt` → Dependencies
- `CO2_ML_Emission_Project.ipynb`→ Jupyter Notebook(EDA + Model Training)

 ### Dataset
The dataset contains details about different vehicles:

Numerical features: Engine Size, Cylinders, Fuel Consumption (City, Hwy, Comb)
Categorical features: Make, Model, Vehicle Class, Transmission, Fuel Type
🎯 Target variable: CO₂ Emissions (g/km)  

### Deployment
The project is deployed using Streamlit Cloud.
**[Click here to try the live app ]**
**(https://co2emission-piddgpdmfyovzgpfdhmpc7.streamlit.app/)**

