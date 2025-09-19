import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_error
import joblib

data=pd.read_csv("C:\\Users\\cw\\Downloads\\co2_emissions.csv")

categorical_features=['make','model','vehicle_class','transmission','fuel_type']
numeric_features = [
    "engine_size", "cylinders",
    "fuel_consumption_city", "fuel_consumption_hwy",
    "fuel_consumption_comb(l/100km)", "fuel_consumption_comb(mpg)"
]
X=data[categorical_features + numeric_features]
y=data['co2_emissions']


preprocessor= ColumnTransformer(
     transformers= [('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features),
                    ('num',StandardScaler(),numeric_features)
    
])

pipeline= Pipeline([
    ('preprocessor',preprocessor),
    ('model',KNeighborsRegressor())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {"model__n_neighbors": [3, 5, 7, 9]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2")
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Params:", grid_search.best_params_)
print("Test R²:", r2_score(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)   # returns MSE
rmse = np.sqrt(mse)
print("RMSE:", rmse)
#print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Save trained pipeline
joblib.dump((best_model,categorical_features, numeric_features), "co2_knn_model.pkl")
print("✅ Model saved successfully!")