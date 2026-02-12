# 🚴 Bike Rental Demand Analysis - Streamlit Deployment Guide

## 📋 Overview

This document provides comprehensive instructions for deploying the Bike Rental Demand Analysis application using Streamlit.

## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Before running the Streamlit app, train and save the machine learning model:

```bash
python train_and_save_model.py
```

This script will:

- Load the raw dataset from `data/raw/Dataset.csv`
- Preprocess and clean the data
- Train an XGBoost model with optimized hyperparameters
- Save the model to `models/best_model.pkl`
- Display performance metrics

**Output:**

```
MODEL PERFORMANCE METRICS
==================================================
Training R² Score:   0.8234
Test R² Score:       0.7956
Training RMSE:       23.45
Test RMSE:           27.89
Training MAE:        18.23
Test MAE:            21.34
==================================================
```

### Step 3: Run the Streamlit App

```bash
streamlit run src/app.py
```

The app will start on `http://localhost:8501`

## 🎯 Features

### 1. **Home Page**

- Project overview
- Dataset statistics
- Quick navigation to other sections

### 2. **Data Explorer**

- Browse the complete dataset
- View summary statistics
- Check for missing values
- Filter and display specific columns
- Explore data types

### 3. **Exploratory Analysis**

- **Distributions**: Histogram and boxplot visualizations
- **Correlations**: Heatmap showing feature relationships
- **Relationships**: Scatter plots between variables
- **Statistical Summary**: Detailed statistics for all features

### 4. **Make Prediction**

- Interactive sliders for continuous variables:
  - Temperature (0-50°C)
  - Humidity (0-100%)
  - Windspeed (0-50 km/h)
  - Hour (0-23)
  - Month (1-12)
- Dropdown menus for categorical variables:
  - Season (Spring, Summer, Fall, Winter)
  - Weather (Clear, Mist, Light Snow, Heavy Rain)
  - Holiday (Yes/No)
  - Working Day (Yes/No)
- Real-time prediction output
- Detailed input parameters display

## 📊 Dataset Structure

The preprocessed dataset includes:

### Temporal Features

- Year (`yr`)
- Month (`mnth`)
- Hour (`hr`)
- Day of week (`weekday`)

### Weather Features

- Temperature (`temp`)
- Apparent temperature (`atemp`)
- Humidity (`hum`)
- Windspeed (`windspeed`)

### Categorical Features (One-Hot Encoded)

- Season (Spring, Summer, Fall, Winter)
- Holiday indicator
- Working day indicator
- Weather situation

### Target Variable

- `cnt`: Count of bike rentals

## 🔧 Configuration

### Streamlit Configuration

The app is configured via `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"

[server]
port = 8501
maxUploadSize = 200
```

To customize:

1. Edit `.streamlit/config.toml`
2. Restart the app to apply changes

## 📁 Project Structure

```
Bike-Rental-Analysis/
├── data/
│   ├── raw/
│   │   └── Dataset.csv              # Original dataset
│   └── processed/
│       └── preprocessed_df.csv      # Preprocessed data
├── models/
│   └── best_model.pkl               # Trained model (generated)
├── src/
│   ├── __init__.py
│   ├── app.py                       # Main Streamlit app
│   ├── data_preprocessing.py        # Data preprocessing functions
│   ├── eda.py                       # EDA functions
│   ├── feature_engineering.py       # Feature engineering
│   ├── model_building.py            # Model building functions
│   └── training_and_evaluation.py   # Training functions
├── notebooks/
│   └── Bike_Rental.ipynb            # Jupyter notebook
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
├── .gitignore
├── requirements.txt                 # Python dependencies
├── DEPLOYMENT.md                    # This file
└── README.md                        # Project README
```

## 🎓 How to Use the App

### Making Predictions

1. Navigate to "Make Prediction" page
2. Adjust input parameters using sliders and dropdowns
3. Click "🔍 Predict Demand" button
4. View the predicted number of bike rentals

### Example Scenario

**Inputs:**

- Temperature: 25°C
- Humidity: 60%
- Windspeed: 15 km/h
- Season: Summer
- Weather: Clear
- Holiday: No
- Working Day: Yes
- Hour: 14
- Month: 6

**Expected Output:**

- Predicted Rentals: ~450-550 bikes

## 🐛 Troubleshooting

### Issue: "Model not found" error

**Solution:**

- Run `python train_and_save_model.py` to create the model
- Ensure `models/best_model.pkl` exists

### Issue: "File not found: Dataset.csv"

**Solution:**

- Ensure raw data exists at `data/raw/Dataset.csv`
- Verify file path and name are correct

### Issue: Port 8501 already in use

**Solution:**

```bash
streamlit run src/app.py --server.port 8502
```

### Issue: Slow performance

**Solution:**

- Clear Streamlit cache: `streamlit cache clear`
- Reduce dataset size or use sampling
- Check system resources

## 📈 Model Performance

The XGBoost model achieves:

- **R² Score**: ~0.80 on test data
- **RMSE**: ~28 bikes
- **MAE**: ~21 bikes

Feature importance (top features):

1. Temperature
2. Hour
3. Working Day
4. Season
5. Humidity

## 🔐 Security Considerations

When deploying to production:

1. **Environment Variables**: Store sensitive data in `.env`
2. **Authentication**: Add user authentication if needed
3. **API Rate Limiting**: Implement rate limits for predictions
4. **Data Privacy**: Ensure GDPR compliance if handling user data
5. **HTTPS**: Use HTTPS for all deployments
6. **Secrets Management**: Never commit API keys or credentials

## 📱 Deployment Platforms

### Local Deployment

```bash
streamlit run src/app.py
```

### Streamlit Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Deploy app

### Heroku

```bash
heroku create your-app-name
git push heroku main
```

### Docker

```bash
docker build -t bike-rental-app .
docker run -p 8501:8501 bike-rental-app
```

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Pandas Documentation](https://pandas.pydata.org)
- [Scikit-learn Documentation](https://scikit-learn.org)

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 👨‍💻 Support

For issues or questions:

1. Check the Troubleshooting section
2. Review the notebook for analysis details
3. Open an issue on GitHub

---

**Last Updated**: February 12, 2026
**Version**: 1.0.0
