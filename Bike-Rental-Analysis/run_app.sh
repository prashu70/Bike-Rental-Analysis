#!/bin/bash

# Streamlit Bike Rental App Launcher
# This script sets up the environment and runs the Streamlit app

echo "🚴 Bike Rental Demand Analysis - Streamlit App"
echo "=============================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

echo "✅ Python found: $(python --version)"
echo ""

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️ Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies installed"
echo ""

# Check if model exists
if [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️ Trained model not found!"
    echo "🔄 Training model..."
    python train_and_save_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to train model"
        exit 1
    fi
fi

echo "✅ Model ready"
echo ""

# Run Streamlit app
echo "🚀 Starting Streamlit app..."
echo "📍 App will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run src/app.py
