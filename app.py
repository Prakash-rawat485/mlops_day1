import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.title("Advertising Sales Predictor")

tv = st.number_input("TV Budget")
radio = st.number_input("Radio Budget")
newspaper = st.number_input("Newspaper Budget")

if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Sales: {prediction[0]:.2f}"
    )