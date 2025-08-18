# data_preprocessing.py
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.dropna()
    return df

def preprocess_data(df):
    df = clean_data(df)
    # Add more preprocessing steps as needed
    return df

