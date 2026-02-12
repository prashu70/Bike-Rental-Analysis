@echo off
REM Streamlit Bike Rental App Launcher for Windows
REM This script sets up the environment and runs the Streamlit app

echo 🚴 Bike Rental Demand Analysis - Streamlit App
echo ==============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version
echo.

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Dependencies installed
echo.

REM Check if model exists
if not exist "models\best_model.pkl" (
    echo ⚠️ Trained model not found!
    echo 🔄 Training model...
    python train_and_save_model.py
    if errorlevel 1 (
        echo ❌ Failed to train model
        pause
        exit /b 1
    )
)

echo ✅ Model ready
echo.

REM Run Streamlit app
echo 🚀 Starting Streamlit app...
echo 📍 App will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

streamlit run src\app.py

pause
