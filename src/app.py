
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("best_crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load country list and feature names from training
df = pd.read_csv("./data/crop_data_cleaned.csv")
country_list = sorted(df["Country"].unique())
base_features = [
    "Rainfall_mm_per_year", "Avg_Temp_C", "Pesticides_tonnes",
    "Rainfall_dev", "Temp_dev",
    "Rainfall_x_Temp", "Rainfall_x_Pest", "Temp_x_Pest",
    "Pest_per_mmRain", "Rainfall_sq", "Temp_sq", "Pest_sq"
]
country_dummies = pd.get_dummies(df["Country"], drop_first=True)
all_feature_names = base_features + list(country_dummies.columns)

st.title("🌾 Crop Yield Prediction App")
st.write("Enter rainfall, temperature, pesticide usage, and select country to predict crop yield.")

# Input fields
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
pesticides = st.number_input("Pesticide Use (kg/ha)", min_value=0.0, step=0.1)
country = st.selectbox("Country", country_list)

if st.button("Predict Yield"):
    # Compute engineered features
    rainfall_dev = rainfall - df["Rainfall_mm_per_year"].mean()
    temp_dev = temperature - df["Avg_Temp_C"].mean()
    rainfall_x_temp = rainfall * temperature
    rainfall_x_pest = rainfall * pesticides
    temp_x_pest = temperature * pesticides
    pest_per_mmRain = pesticides / (rainfall + 1e-6)
    rainfall_sq = rainfall ** 2
    temp_sq = temperature ** 2
    pest_sq = pesticides ** 2

    # Build feature dict
    input_dict = {
        "Rainfall_mm_per_year": rainfall,
        "Avg_Temp_C": temperature,
        "Pesticides_tonnes": pesticides,
        "Rainfall_dev": rainfall_dev,
        "Temp_dev": temp_dev,
        "Rainfall_x_Temp": rainfall_x_temp,
        "Rainfall_x_Pest": rainfall_x_pest,
        "Temp_x_Pest": temp_x_pest,
        "Pest_per_mmRain": pest_per_mmRain,
        "Rainfall_sq": rainfall_sq,
        "Temp_sq": temp_sq,
        "Pest_sq": pest_sq
    }

    # One-hot encode country
    for c in country_dummies.columns:
        input_dict[c] = 1 if c == country else 0

    # Ensure all features are present
    input_features = [input_dict.get(f, 0) for f in all_feature_names]
    input_df = pd.DataFrame([input_features], columns=all_feature_names)

    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} kg/ha")
