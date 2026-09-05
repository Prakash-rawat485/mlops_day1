import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_data_schema():
    data_path = os.path.join(BASE_DIR, "data", "data.csv")
    assert os.path.exists(data_path), "Data file data.csv not found!"
    df = pd.read_csv(data_path)
    expected_cols = {"TV", "Radio", "Newspaper", "Sales"}
    assert expected_cols.issubset(set(df.columns)), "Missing required columns in dataset"
    assert df[["TV", "Radio", "Newspaper"]].isnull().sum().sum() == 0, "Null values found in features"

def test_champion_model_loading_and_prediction():
    model_path = os.path.join(BASE_DIR, "models", "champion_model.pkl")
    assert os.path.exists(model_path), f"Champion model not found at {model_path}"
    
    model = joblib.load(model_path)
    
    sample_input = pd.DataFrame([[100.0, 25.0, 10.0]], columns=["TV", "Radio", "Newspaper"])
    prediction = model.predict(sample_input)
    
    assert len(prediction) == 1, "Prediction output shape invalid"
    assert float(prediction[0]) > 0, "Prediction value is invalid"