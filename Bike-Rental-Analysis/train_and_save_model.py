# -*- coding: utf-8 -*-
"""
Script to train XGBoost model and save it for deployment
Run this script to train the model before deploying the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Remove rows with '?' values
    for col in df.columns:
        df = df[df[col] != '?']
    
    print(f"Initial shape: {df.shape}")
    print(f"Target (cnt) stats: min={df['cnt'].min()}, max={df['cnt'].max()}, mean={df['cnt'].mean():.2f}")
    
    # Convert data types
    cols_int = ['yr', 'mnth', 'hr', 'weekday', 'casual', 'registered']
    for col in cols_int:
        if col in df.columns:
            df[col] = df[col].astype('int64')
    
    cols_float = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    for col in cols_float:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    # One-hot encode categorical features
    categorical_cols = ['season', 'holiday', 'workingday', 'weathersit']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert boolean to int
    df = df.replace({True: 1, False: 0})
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop unnecessary columns
    cols_to_drop = ['instant', 'dteday']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def train_model(df):
    """Train XGBoost model with hyperparameter tuning"""
    print("\nPreparing data for training...")
    
    # Split features and target
    X = df.drop('cnt', axis=1)
    y = df['cnt']
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Feature count: {X.shape[1]}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Target (train) stats: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")
    
    # Train XGBoost model with optimized parameters
    print("\nTraining XGBoost model...")
    model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate model
    print("\nEvaluating model...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Training R² Score:   {train_r2:.4f}")
    print(f"Test R² Score:       {test_r2:.4f}")
    print(f"Training RMSE:       {train_rmse:.2f}")
    print(f"Test RMSE:           {test_rmse:.2f}")
    print(f"Training MAE:        {train_mae:.2f}")
    print(f"Test MAE:            {test_mae:.2f}")
    print(f"{'='*50}\n")
    
    # Sample predictions
    print("Sample predictions (first 5 test samples):")
    for i in range(min(5, len(y_test))):
        print(f"  Actual: {y_test.iloc[i]:.2f}, Predicted: {test_pred[i]:.2f}")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    return model, X_train.columns

def save_model(model, feature_names, model_path):
    """Save the trained model"""
    print(f"\nSaving model to {model_path}...")
    
    # Create parent directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_names_path = Path(model_path).parent / "feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    
    print("✅ Model saved successfully!")
    print(f"   Model path: {model_path}")
    print(f"   Features path: {feature_names_path}")

def main():
    """Main execution function"""
    # Define paths
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "raw" / "Dataset.csv"
    model_path = project_root / "models" / "best_model.pkl"
    
    # Check if data exists
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        return
    
    # Load and preprocess data
    df = load_and_preprocess_data(str(data_path))
    
    # Train model
    model, feature_names = train_model(df)
    
    # Save model
    save_model(model, feature_names, str(model_path))

if __name__ == "__main__":
    main()

def save_model(model, feature_names, model_path):
    """Save the trained model"""
    print(f"\nSaving model to {model_path}...")
    
    # Create parent directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save feature names
    feature_names_path = Path(model_path).parent / "feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    
    print("✅ Model saved successfully!")
    print(f"   Model path: {model_path}")
    print(f"   Features path: {feature_names_path}")

def main():
    """Main execution function"""
    # Define paths
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "raw" / "Dataset.csv"
    model_path = project_root / "models" / "best_model.pkl"
    
    # Check if data exists
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        return
    
    # Load and preprocess data
    df = load_and_preprocess_data(str(data_path))
    
    # Train model
    model, feature_names = train_model(df)
    
    # Save model
    save_model(model, feature_names, str(model_path))

if __name__ == "__main__":
    main()
