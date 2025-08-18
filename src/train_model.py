import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("./data/crop_data_cleaned.csv")

# === Feature Engineering ===
df["Rainfall_dev"] = df["Rainfall_mm_per_year"] - df["Rainfall_mm_per_year"].mean()
df["Temp_dev"] = df["Avg_Temp_C"] - df["Avg_Temp_C"].mean()

df["Rainfall_x_Temp"] = df["Rainfall_mm_per_year"] * df["Avg_Temp_C"]
df["Rainfall_x_Pest"] = df["Rainfall_mm_per_year"] * df["Pesticides_tonnes"]
df["Temp_x_Pest"] = df["Avg_Temp_C"] * df["Pesticides_tonnes"]

df["Pest_per_mmRain"] = df["Pesticides_tonnes"] / (df["Rainfall_mm_per_year"] + 1e-6)  # avoid divide by zero

# Quadratic terms
df["Rainfall_sq"] = df["Rainfall_mm_per_year"] ** 2
df["Temp_sq"] = df["Avg_Temp_C"] ** 2
df["Pest_sq"] = df["Pesticides_tonnes"] ** 2


# Features and target (include engineered features and one-hot encoded Country)
base_features = [
    "Rainfall_mm_per_year", "Avg_Temp_C", "Pesticides_tonnes",
    "Rainfall_dev", "Temp_dev",
    "Rainfall_x_Temp", "Rainfall_x_Pest", "Temp_x_Pest",
    "Pest_per_mmRain", "Rainfall_sq", "Temp_sq", "Pest_sq"
]
X = df[base_features]
if "Country" in df.columns:
    X = pd.concat([X, pd.get_dummies(df["Country"], drop_first=True)], axis=1)
y = df["Yield"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost + tuning
xgb = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='r2', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Best Parameters: {grid_search.best_params_}")
print(f"📊 MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

# Scatter plot
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield (Engineered Features)")
plt.show()

# Feature importance
plt.bar(X.columns, best_model.feature_importances_, color="green")
plt.ylabel("Importance")
plt.title("Feature Importance - Tuned XGBoost (Engineered Features)")
plt.xticks(rotation=90)
plt.show()

# Save model
joblib.dump(best_model, "best_crop_yield_model.pkl")
print("💾 Model saved as best_crop_yield_model.pkl")
