"""
Comprehensive Model Training Pipeline
Implements multiple ML algorithms according to 2nd Review specifications
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Handles training and evaluation of multiple ML models"""
    
    def __init__(self, data_path='data/solar_data.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess dataset"""
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Display dataset info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering - extract time-based features if datetime exists
        if 'datetime' in df.columns or 'DATE_TIME' in df.columns:
            date_col = 'datetime' if 'datetime' in df.columns else 'DATE_TIME'
            df[date_col] = pd.to_datetime(df[date_col])
            df['hour'] = df[date_col].dt.hour
            df['day'] = df[date_col].dt.day
            df['month'] = df[date_col].dt.month
            df = df.drop(columns=[date_col])
        
        # Separate features and target
        # Assuming the target is power output (adjust column name as needed)
        target_columns = ['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']
        target_col = None
        
        for col in target_columns:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            # If no standard target found, use last column
            target_col = df.columns[-1]
            
        print(f"Target variable: {target_col}")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Data preprocessing complete")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)
    
    def train_baseline_model(self, X_train, X_val, y_train, y_val):
        """Train baseline Linear Regression model"""
        print("\n" + "="*60)
        print("🔵 Training Baseline Model: Linear Regression")
        print("="*60)
        
        start_time = time.time()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        results = {
            'model': model,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time
        }
        
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = results
        
        print(f"✅ Training R²: {results['train_r2']:.4f}")
        print(f"✅ Validation R²: {results['val_r2']:.4f}")
        print(f"✅ Validation RMSE: {results['val_rmse']:.4f}")
        print(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def train_random_forest(self, X_train, X_val, y_train, y_val):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n" + "="*60)
        print("🌲 Training Random Forest Regressor")
        print("="*60)
        
        start_time = time.time()
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        print("🔍 Performing Grid Search...")
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='r2', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_time = time.time() - start_time
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Metrics
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time,
            'feature_importance': dict(zip(self.feature_names, 
                                          best_model.feature_importances_))
        }
        
        self.models['Random Forest'] = best_model
        self.results['Random Forest'] = results
        
        print(f"✅ Training R²: {results['train_r2']:.4f}")
        print(f"✅ Validation R²: {results['val_r2']:.4f}")
        print(f"✅ Validation RMSE: {results['val_rmse']:.4f}")
        print(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost model with hyperparameter tuning"""
        print("\n" + "="*60)
        print("🚀 Training XGBoost Regressor")
        print("="*60)
        
        start_time = time.time()
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        print("🔍 Performing Grid Search...")
        xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_time = time.time() - start_time
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Metrics
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_time': train_time,
            'feature_importance': dict(zip(self.feature_names,
                                          best_model.feature_importances_))
        }
        
        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = results
        
        print(f"✅ Training R²: {results['train_r2']:.4f}")
        print(f"✅ Validation R²: {results['val_r2']:.4f}")
        print(f"✅ Validation RMSE: {results['val_rmse']:.4f}")
        print(f"⏱️  Training time: {train_time:.2f}s")
        
        return results
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train R²': [self.results[m]['train_r2'] for m in self.results],
            'Val R²': [self.results[m]['val_r2'] for m in self.results],
            'Val RMSE': [self.results[m]['val_rmse'] for m in self.results],
            'Val MAE': [self.results[m]['val_mae'] for m in self.results],
            'Train Time (s)': [self.results[m]['train_time'] for m in self.results]
        })
        
        comparison_df = comparison_df.sort_values('Val R²', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Validation R²: {self.results[best_model_name]['val_r2']:.4f}")
        
        return comparison_df
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Evaluate best model on test set"""
        print("\n" + "="*60)
        print("🎯 FINAL TEST SET EVALUATION")
        print("="*60)
        
        y_pred = self.best_model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"Model: {self.best_model_name}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test MAPE: {test_mape:.2f}%")
        
        return {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape
        }
    
    def save_models(self):
        """Save best model and preprocessor"""
        print("\n💾 Saving models...")
        
        # Save best model
        joblib.dump(self.best_model, 'models/best_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'trained_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'performance': {
                'val_r2': float(self.results[self.best_model_name]['val_r2']),
                'val_rmse': float(self.results[self.best_model_name]['val_rmse']),
                'val_mae': float(self.results[self.best_model_name]['val_mae'])
            }
        }
        
        if 'best_params' in self.results[self.best_model_name]:
            metadata['hyperparameters'] = self.results[self.best_model_name]['best_params']
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Models saved to 'models/' directory")
        print("   - best_model.pkl")
        print("   - scaler.pkl")
        print("   - model_metadata.json")
    
    def train_all(self):
        """Run complete training pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE MODEL TRAINING PIPELINE")
        print("="*80)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Train models
        self.train_baseline_model(X_train, X_val, y_train, y_val)
        self.train_random_forest(X_train, X_val, y_train, y_val)
        self.train_xgboost(X_train, X_val, y_train, y_val)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Test set evaluation
        test_results = self.evaluate_on_test_set(X_test, y_test)
        
        # Save models
        self.save_models()
        
        print("\n" + "="*80)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*80)
        
        return comparison_df, test_results


if __name__ == "__main__":
    trainer = ModelTrainer()
    comparison_df, test_results = trainer.train_all()
