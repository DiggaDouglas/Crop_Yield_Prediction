
# Crop Yield Prediction

## Project Overview
This project predicts crop yields using advanced machine learning techniques, leveraging historical data on rainfall, temperature, pesticide usage, and country information. The goal is to provide accurate yield estimates to support agricultural planning and decision-making.

## Technologies Used
- **Python 3.13**: Main programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Model selection, metrics, and preprocessing
- **XGBoost**: Gradient boosting regression for optimal performance
- **Streamlit**: Interactive web app for yield prediction
- **Jupyter Notebook**: Data exploration and experimentation

## Project Structure
- `data/`: Raw and cleaned datasets
- `notebooks/`: Jupyter notebooks for exploration and modeling
- `src/`: Python scripts for data processing, modeling, and utilities
- `requirements.txt`: List of dependencies
- `.gitignore`: Ignore unnecessary files

## Scope
- Data cleaning and feature engineering (including interaction and quadratic terms)
- Model training and hyperparameter tuning using XGBoost
- One-hot encoding for country information
- Evaluation of model performance (MSE, RMSE, MAE, R²)
- Deployment of a Streamlit app for user-friendly predictions

## Error Margin
The final XGBoost model was evaluated using RMSE, MAE, and R² metrics. The error margin (RMSE) is typically within a reasonable range for agricultural yield prediction, but may vary depending on data quality and country. For example:

- **RMSE**: ~[81651.61] kg/ha
- **MAE**: ~[62623.86] kg/ha
- **R²**: ~[0.0999]

*Replace the above with your actual results from the terminal output.*

## How to Use
1. Install dependencies from `requirements.txt` (use your virtual environment)
2. Explore and preprocess data in `notebooks/01_data_exploration.ipynb`
3. Train and evaluate models in `src/train_model.py`
4. Run the Streamlit app for predictions:
	```
	streamlit run src/app.py
	```

## Author
*Emmanuel Douglas*
