# Data Preprocessing Pipeline
# Handles ETL operations as per 2nd Review specifications

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    Implements ETL (Extract, Transform, Load) operations
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def extract_data(self, file_path):
        """
        EXTRACT: Load data from various sources
        """
        print(f"📥 Extracting data from: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        print(f"✅ Extracted {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def transform_data(self, df, target_column=None):
        """
        TRANSFORM: Clean, engineer features, and prepare data
        """
        print("\n🔄 Transforming data...")
        
        # 1. Handle missing values
        print("  - Handling missing values...")
        missing_before = df.isnull().sum().sum()
        df = self._handle_missing_values(df)
        missing_after = df.isnull().sum().sum()
        print(f"    Reduced missing values: {missing_before} → {missing_after}")
        
        # 2. Remove duplicates
        print("  - Removing duplicates...")
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"    Removed {duplicates} duplicate rows")
        
        # 3. Feature engineering
        print("  - Engineering features...")
        df = self._engineer_features(df)
        
        # 4. Encode categorical variables
        print("  - Encoding categorical variables...")
        df = self._encode_categorical(df)
        
        # 5. Handle outliers
        print("  - Detecting and handling outliers...")
        df = self._handle_outliers(df)
        
        print("✅ Transformation complete")
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    # Numerical: fill with median
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # Categorical: fill with mode
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    
    def _engineer_features(self, df):
        """Create derived features"""
        # Extract datetime features if datetime column exists
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df = df.drop(columns=[col])
                except:
                    pass
        
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df
    
    def _handle_outliers(self, df, threshold=3):
        """Remove outliers using Z-score method"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        
        return df
    
    def load_data(self, df, output_path):
        """
        LOAD: Save processed data
        """
        print(f"\n💾 Loading data to: {output_path}")
        df.to_csv(output_path, index=False)
        print("✅ Data saved successfully")
        return df
    
    def run_etl_pipeline(self, input_path, output_path, target_column=None):
        """
        Run complete ETL pipeline
        """
        print("\n" + "="*60)
        print("🚀 STARTING ETL PIPELINE")
        print("="*60)
        
        # Extract
        df = self.extract_data(input_path)
        
        # Transform
        df = self.transform_data(df, target_column)
        
        # Load
        df = self.load_data(df, output_path)
        
        print("\n" + "="*60)
        print("✅ ETL PIPELINE COMPLETE")
        print("="*60)
        
        return df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_etl_pipeline(
        'data/solar_data.csv',
        'data/processed_data.csv'
    )
