# -*- coding: utf-8 -*-
"""
Streamlit Deployment App for Bike Rental Demand Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ---- Page Configuration ----
st.set_page_config(
    page_title="🚴 Bike Rental Demand Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Styling ----
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---- App Title ----
st.title("🚴 Bike Rental Demand Analysis & Prediction")
st.markdown("---")

# ---- Sidebar Navigation ----
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Select a page:",
        ["Home", "Data Explorer", "Exploratory Analysis", "Make Prediction"]
    )
    st.markdown("---")
    st.info("📊 This app analyzes bike rental demand patterns and predicts future rental volumes.")

# ---- Load Data Function ----
@st.cache_data
def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

@st.cache_data
def load_model(model_path):
    """Load pre-trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.warning(f"Model not found at: {model_path}")
        return None

# ---- Load Data ----
data_path = Path(__file__).parent.parent / "data" / "processed" / "preprocessed_df.csv"
model_path = Path(__file__).parent.parent / "models" / "best_model.pkl"

df = load_data(str(data_path))

# ---- PAGE: Home ----
if page == "Home":
    st.header("Welcome to Bike Rental Demand Analysis! 🚴")
    
    col1, col2, col3 = st.columns(3)
    
    if df is not None:
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Target: Bike Rentals", "cnt")
    
    st.markdown("""
    ### 📌 About This Project
    This application provides comprehensive analysis and predictions for bike rental demand based on 
    various factors including weather conditions, season, and temporal features.
    
    ### 🎯 Features
    - **Data Explorer**: Browse and filter the dataset
    - **Exploratory Analysis**: Visualize patterns and correlations
    - **Make Prediction**: Use ML model to predict rental demand
    
    ### 📊 Dataset Information
    The dataset includes:
    - Temporal features (date, hour, day of week, month)
    - Weather conditions (temperature, humidity, windspeed)
    - Season and holiday indicators
    - Target variable: Number of bike rentals
    """)

# ---- PAGE: Data Explorer ----
elif page == "Data Explorer":
    st.header("📊 Data Explorer")
    
    if df is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df, use_container_width=True, height=300)
        
        with col2:
            st.subheader("Dataset Shape")
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
        
        st.markdown("---")
        
        # Summary Statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
        
        # Missing Values
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values detected!")
        else:
            st.dataframe(missing_data[missing_data > 0])
        
        # Data Types
        st.subheader("Data Types")
        st.dataframe(df.dtypes)
        
        # Filter by column
        st.markdown("---")
        st.subheader("Filter Data")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_cols = st.multiselect(
                "Select columns to display:",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]
            )
        
        if selected_cols:
            st.dataframe(df[selected_cols], use_container_width=True)
    else:
        st.warning("Dataset not available. Please check the data path.")

# ---- PAGE: Exploratory Analysis ----
elif page == "Exploratory Analysis":
    st.header("📈 Exploratory Data Analysis")
    
    if df is not None:
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Distributions", "Correlations", "Relationships", "Statistical Summary"]
        )
        
        if analysis_type == "Distributions":
            st.subheader("Feature Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Select feature to visualize:", numeric_cols)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df[selected_col], bins=30, edgecolor='black', color='skyblue')
                    ax.set_title(f"Distribution of {selected_col}")
                    ax.set_xlabel(selected_col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.boxplot(df[selected_col])
                    ax.set_title(f"Boxplot of {selected_col}")
                    ax.set_ylabel(selected_col)
                    st.pyplot(fig)
        
        elif analysis_type == "Correlations":
            st.subheader("Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title("Correlation Matrix of Numeric Features")
            st.pyplot(fig)
        
        elif analysis_type == "Relationships":
            st.subheader("Feature Relationships")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X-axis variable:", numeric_cols)
            with col2:
                y_var = st.selectbox("Select Y-axis variable:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            if x_var and y_var:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[x_var], df[y_var], alpha=0.5, color='steelblue')
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.set_title(f"{y_var} vs {x_var}")
                st.pyplot(fig)
        
        elif analysis_type == "Statistical Summary":
            st.subheader("Statistical Summary")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats = df[numeric_cols].describe().T
            
            st.dataframe(stats, use_container_width=True)
    else:
        st.warning("Dataset not available.")

# ---- PAGE: Make Prediction ----
elif page == "Make Prediction":
    st.header("🔮 Bike Rental Demand Prediction")
    
    st.info("Enter the input features below to predict bike rental demand.")
    
    model = load_model(str(model_path))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 20.0, step=0.1)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, step=1.0)
        windspeed = st.slider("Windspeed (km/h)", 0.0, 50.0, 10.0, step=0.1)
    
    with col2:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], index=1)
        weather = st.selectbox("Weather Condition", ["Clear", "Mist", "Light Snow", "Heavy Rain"], index=0)
        holiday = st.selectbox("Holiday", ["No", "Yes"])
    
    with col3:
        workingday = st.selectbox("Working Day", ["No", "Yes"])
        hour = st.slider("Hour of Day", 0, 23, 12)
        month = st.slider("Month", 1, 12, 6)
    
    if st.button("🔍 Predict Demand", use_container_width=True):
        if model is not None:
            try:
                # Create base feature dictionary with numeric values
                features = {
                    'yr': 1,  # Default year value
                    'mnth': month,
                    'hr': hour,
                    'weekday': 3,  # Default weekday (Wednesday)
                    'temp': temp,
                    'atemp': temp * 0.95,  # Approximate apparent temperature
                    'hum': humidity,
                    'windspeed': windspeed,
                    'casual': 0,  # Will be dropped
                    'registered': 0,  # Will be dropped
                }
                
                # Map season to one-hot encoded columns
                season_map = {
                    'Spring': {'season_springer': 0, 'season_summer': 0, 'season_winter': 0},
                    'Summer': {'season_springer': 0, 'season_summer': 1, 'season_winter': 0},
                    'Fall': {'season_springer': 0, 'season_summer': 0, 'season_winter': 0},
                    'Winter': {'season_springer': 0, 'season_summer': 0, 'season_winter': 1}
                }
                
                # Map weather to one-hot encoded columns
                weather_map = {
                    'Clear': {'weathersit_Heavy Rain': 0, 'weathersit_Light Snow': 0, 'weathersit_Mist': 0},
                    'Mist': {'weathersit_Heavy Rain': 0, 'weathersit_Light Snow': 0, 'weathersit_Mist': 1},
                    'Light Snow': {'weathersit_Heavy Rain': 0, 'weathersit_Light Snow': 1, 'weathersit_Mist': 0},
                    'Heavy Rain': {'weathersit_Heavy Rain': 1, 'weathersit_Light Snow': 0, 'weathersit_Mist': 0}
                }
                
                # Add encoded categorical features
                features.update(season_map[season])
                features.update(weather_map[weather])
                features['holiday_Yes'] = 1 if holiday == "Yes" else 0
                features['workingday_Working Day'] = 1 if workingday == "Yes" else 0
                
                # Convert to DataFrame
                prediction_df = pd.DataFrame([features])
                
                # Select only numeric columns and drop unnecessary ones
                numeric_cols = ['yr', 'mnth', 'hr', 'weekday', 'temp', 'atemp', 'hum', 'windspeed',
                               'casual', 'registered', 'season_springer', 'season_summer', 'season_winter', 
                               'holiday_Yes', 'workingday_Working Day', 'weathersit_Heavy Rain', 
                               'weathersit_Light Snow', 'weathersit_Mist']
                
                # Ensure all columns are present and numeric
                for col in numeric_cols:
                    if col not in prediction_df.columns:
                        prediction_df[col] = 0
                    prediction_df[col] = pd.to_numeric(prediction_df[col], errors='coerce')
                
                # Select only the required columns
                prediction_df = prediction_df[numeric_cols]
                
                # Make prediction
                prediction = model.predict(prediction_df)[0]
                
                # Debug: show the input data
                with st.expander("🔍 Debug Info - Input Features"):
                    st.write("Input DataFrame:")
                    st.dataframe(prediction_df)
                    st.write(f"Raw Prediction Value: {prediction}")
                    st.write(f"Prediction Type: {type(prediction)}")
                
                # Display results
                st.markdown("---")
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Rentals", f"{int(prediction):,} bikes")
                with col2:
                    st.metric("Temperature", f"{temp}°C")
                with col3:
                    st.metric("Season", season)
                
                # Additional insights
                st.markdown("---")
                st.subheader("📊 Prediction Details")
                st.write(f"**Input Parameters:**")
                for key, value in features.items():
                    st.write(f"- {key}: {value}")
                
                st.write(f"\n**Predicted bike rentals:** `{int(prediction)}` bikes")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Model not loaded. Please check if the model file exists at the expected location.")
