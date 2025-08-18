

import pandas as pd
import joblib

def load_data(path):
	"""Load CSV data and handle missing values."""
	df = pd.read_csv(path)
	df = df.dropna()
	return df

def save_model(model, filename):
	"""Save model to disk."""
	joblib.dump(model, filename)

def load_model(filename):
	"""Load model from disk."""
	return joblib.load(filename)

def engineer_features(df):
	"""Add engineered features to DataFrame."""
	df = df.copy()
	df["Rainfall_dev"] = df["Rainfall_mm_per_year"] - df["Rainfall_mm_per_year"].mean()
	df["Temp_dev"] = df["Avg_Temp_C"] - df["Avg_Temp_C"].mean()
	df["Rainfall_x_Temp"] = df["Rainfall_mm_per_year"] * df["Avg_Temp_C"]
	df["Rainfall_x_Pest"] = df["Rainfall_mm_per_year"] * df["Pesticides_tonnes"]
	df["Temp_x_Pest"] = df["Avg_Temp_C"] * df["Pesticides_tonnes"]
	df["Pest_per_mmRain"] = df["Pesticides_tonnes"] / (df["Rainfall_mm_per_year"] + 1e-6)
	df["Rainfall_sq"] = df["Rainfall_mm_per_year"] ** 2
	df["Temp_sq"] = df["Avg_Temp_C"] ** 2
	df["Pest_sq"] = df["Pesticides_tonnes"] ** 2
	return df

def encode_country(df):
	"""One-hot encode the Country column."""
	if "Country" in df.columns:
		return pd.get_dummies(df, columns=["Country"], drop_first=True)
	return df
