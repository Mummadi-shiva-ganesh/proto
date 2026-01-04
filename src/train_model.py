import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from data_loader import load_and_process_data

def train_models():
    print("Loading data...")
    df = load_and_process_data()
    
    # Define Target and Features
    # We want to predict Solar Generation Capability based on Weather
    # We can also predict 'Battery Voltage' given SOC, but that's usually a lookup.
    # Let's focus on the ML part: Predicting Solar Power.
    
    feature_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    target_col = 'AC_POWER'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Linear Regression (Baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
    
    # Model 2: Random Forest (Better for non-linear saturation)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred_rf)
    print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
    print(f"Random Forest R2 Score: {r2:.4f}")
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/solar_model.pkl")
    print("Model saved to models/solar_model.pkl")
    
    # Optional: We could also train a model to predict Battery Voltage from SOC
    # But usually a simple formula or lookup is fine. Let's stick to the high-value Solar ML model.

if __name__ == "__main__":
    train_models()
