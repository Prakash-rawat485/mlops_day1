import numpy as np
import pandas as pd
import joblib


new_data = pd.DataFrame({
    "TV": [230.1, 44.5, 17.2],"Radio": [37.8, 39.3, 45.9],
    "Newspaper": [69.2, 45.1, 69.3]})

model=joblib.load(r"D:\MY Projects (github)\mlops_day1\models\linear_regression_model.pkl")

prediction=model.predict(new_data)
print("Predictions for new data:\n", prediction)