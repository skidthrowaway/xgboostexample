import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Load the dataset
try:
    data = pd.read_csv('california_housing.csv')
except FileNotFoundError:
    print("Error: File not found. Please ensure the file is in the current working directory")
    exit()

data = data.dropna()
# Inspect the first few rows of data
print(data.head())

# Split data into features (X) and target (y)
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# XGBoost Regressor
# =========================

# Define the XGBoost model with hyperparameters, can be tuned
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=9,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0
)

# Train the XGBoost model
print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)

# Make predictions with XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost Mean Absolute Error: {mae_xgb:.2f}")

# =========================
# Linear Regression
# =========================

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the Linear Regression model
print("\nTraining Linear Regression model...")
lr_model.fit(X_train, y_train)

# Make predictions with Linear Regression
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f"Linear Regression Mean Absolute Error: {mae_lr:.2f}")

# =========================
# Comparison
# =========================

print("\nModel Comparison:")
print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"Linear Regression MAE: {mae_lr:.2f}")

if mae_xgb < mae_lr:
    print("XGBoost performs better than Linear Regression based on MAE.")
elif mae_xgb > mae_lr:
    print("Linear Regression performs better than XGBoost based on MAE.")
else:
    print("Both models have the same MAE.")

# =========================
# Feature Importance for XGBoost
# =========================

import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title('Top 10 Feature Importances from XGBoost')
plt.show()
