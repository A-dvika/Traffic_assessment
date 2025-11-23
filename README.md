# 🚦 Traffic Volume Prediction System

An AI-powered traffic forecasting system using Machine Learning to predict highway traffic volume based on weather conditions, temporal patterns, and historical data.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green.svg)
![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Streamlit-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Overview

This project implements a complete end-to-end machine learning solution for predicting traffic volume on Interstate highways. The system combines real-time weather data with temporal patterns to provide accurate traffic forecasts, helping with:

- **Traffic Management**: Optimize signal timing and route planning
- **Urban Planning**: Data-driven infrastructure decisions
- **Navigation Apps**: Real-time traffic predictions
- **Emergency Services**: Route optimization during peak hours

### Key Highlights

- 🤖 **Random Forest Regressor** with 19 engineered features
- 📊 **Test RMSE**: ~125 vehicles | **R² Score**: 0.81
- 🌐 **REST API** backend with Flask
- 💻 **Interactive Dashboard** with Streamlit
- ☁️ **Cloud-Ready** deployment on Render
- 📈 **Real-time** and batch prediction capabilities

---

## ✨ Features

### 🎯 Core Functionality

- **Single Predictions**: Get instant traffic volume forecasts
- **Batch Processing**: Upload CSV files for bulk predictions
- **24-Hour Forecasts**: Generate hourly traffic predictions
- **Historical Analysis**: Visualize traffic patterns and trends

### 📊 Advanced Analytics

- Weather impact analysis (temperature, rain, snow, clouds)
- Rush hour detection (7-9 AM, 4-6 PM)
- Holiday vs. regular day patterns
- Day of week and seasonal trends
- Feature importance visualization

### 🎨 User Interface

- Modern gradient-based design
- Interactive Plotly charts and gauges
- Real-time backend status monitoring
- Responsive layout for all devices
- Traffic level indicators (Low/Moderate/Heavy/Very Heavy)

---

## 🎬 Demo

### Single Prediction Interface
![Single Prediction](https://via.placeholder.com/800x400.png?text=Single+Prediction+Dashboard)

### 24-Hour Forecast
![Forecast](https://via.placeholder.com/800x400.png?text=24-Hour+Traffic+Forecast)

### Historical Analysis
![Analysis](https://via.placeholder.com/800x400.png?text=Historical+Traffic+Analysis)

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← User Interface (Frontend)
└────────┬────────┘
         │ HTTP Requests
         ↓
┌─────────────────┐
│   Flask API     │  ← REST API (Backend)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  ML Model       │  ← Random Forest + Feature Engineering
│  - Scaler       │
│  - Encoders     │
└─────────────────┘
```

### Data Flow

1. **Input**: User provides weather + temporal data
2. **Processing**: Feature engineering (19 features)
3. **Prediction**: Random Forest model inference
4. **Output**: Traffic volume forecast + visualizations

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-prediction.git
cd traffic-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the **Metro Interstate Traffic Volume** dataset:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)
- [Kaggle](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume)

Place `Metro_Interstate_Traffic_Volume.csv` in the project root.

### Step 5: Train the Model

```bash
python model.py
```

This will:
- Perform EDA and generate visualizations
- Train the Random Forest model
- Save model artifacts to `models/` directory
- Create plots in `plots/` directory

Expected output:
```
✓ Model training completed!
✓ Test MAE: 106.16 vehicles
✓ Test RMSE: 124.98 vehicles
✓ Test R²: 0.8075
```

---

## 💻 Usage

### Running Locally

#### 1. Start the Backend API

Open a terminal and run:

```bash
python backend.py
```

You should see:
```
Loading models...
✓ All models loaded successfully!
 * Running on http://127.0.0.1:5000
```

#### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# .env
BACKEND_URL=http://localhost:5000
```

#### 3. Start the Frontend

Open a **new terminal** and run:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

#### Single Prediction

1. Navigate to **Single Prediction** mode
2. Set date and time
3. Configure weather conditions:
   - Temperature (Kelvin)
   - Cloud coverage (%)
   - Rain/Snow (mm/h)
   - Weather type
4. Optionally provide historical traffic data
5. Click **"Generate Prediction"**
6. View results with gauge chart and traffic level indicator

#### Batch Prediction

1. Navigate to **Batch Prediction** mode
2. **Option A**: Upload CSV file with required columns
3. **Option B**: Generate 24-hour forecast
   - Select date
   - Configure weather parameters
   - Click **"Generate 24h Forecast"**
4. Download results as CSV

#### Historical Analysis

1. Navigate to **Historical Analysis** mode
2. View training data insights:
   - Traffic distribution
   - Temporal patterns (hourly, daily, monthly)
   - Model performance metrics
   - Feature importance

---

## 📡 API Documentation

### Base URL

```
Local: http://localhost:5000
Production: https://your-app.onrender.com
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 2. Get Model Metrics

```http
GET /metrics
```

**Response:**
```json
{
  "test_metrics": {
    "MAE": 106.16,
    "RMSE": 124.98,
    "R2_Score": 0.8075
  },
  "train_metrics": {
    "MAE": 48.49,
    "RMSE": 76.46,
    "R2_Score": 0.9299
  }
}
```

#### 3. Single Prediction

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "temp": 280.5,
  "rain_1h": 0.0,
  "snow_1h": 0.0,
  "clouds_all": 40,
  "weather_main": "Clouds",
  "weather_description": "scattered clouds",
  "date_time": "2024-01-15 14:30:00",
  "is_holiday": 0,
  "traffic_lag_1": 3500,
  "traffic_lag_2": 3400,
  "traffic_lag_3": 3300
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 3542.75,
  "unit": "vehicles",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 4. Batch Prediction

```http
POST /predict_batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "predictions": [
    { /* prediction data 1 */ },
    { /* prediction data 2 */ },
    ...
  ]
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [3542.75, 3612.32, ...],
  "count": 24,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 5. Get Weather Options

```http
GET /weather_options
```

**Response:**
```json
{
  "weather_main": ["Clear", "Clouds", "Rain", "Snow", ...],
  "weather_description": ["clear sky", "scattered clouds", ...]
}
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temp": 280.5,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 40,
    "weather_main": "Clear",
    "weather_description": "clear sky",
    "is_holiday": 0
  }'
```

---

## 📊 Model Performance

### Training Results

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **MAE** | 48.49 vehicles | 106.16 vehicles |
| **RMSE** | 76.46 vehicles | 124.98 vehicles |
| **R² Score** | 0.9299 | 0.8075 |

### Feature Importance (Top 10)

1. **traffic_lag_1** (31.9%) - Traffic 1 hour ago
2. **day** (31.6%) - Day of month
3. **traffic_rolling_mean_3** (8.9%) - 3-hour rolling average
4. **traffic_rolling_std_3** (8.1%) - 3-hour rolling std dev
5. **temp** (5.6%) - Temperature
6. **month** (4.9%) - Month of year
7. **weather_main_encoded** (3.7%) - Weather condition
8. **clouds_all** (1.8%) - Cloud coverage
9. **weather_desc_encoded** (1.0%) - Weather description
10. **traffic_lag_3** (1.0%) - Traffic 3 hours ago

### Key Insights

- ✅ Historical traffic patterns are the strongest predictors
- ✅ Temporal features (hour, day, month) significantly impact predictions
- ✅ Weather conditions play a moderate role
- ✅ Model generalizes well to unseen data (R² = 0.81)

---

## ☁️ Deployment

### Deploy Backend to Render

#### Step 1: Prepare for Deployment

Ensure all model files are in the `models/` directory:
```
models/
├── traffic_model.pkl
├── scaler.pkl
├── le_weather_main.pkl
├── le_weather_desc.pkl
├── feature_columns.pkl
└── metrics.pkl
```

#### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/a-dvika/traffic-prediction.git
git push -u origin main
```

#### Step 3: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `traffic-prediction-api`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend:app`
   - **Instance Type**: Free
5. Click **"Create Web Service"**
6. Wait for deployment (5-10 minutes)
7. Copy your backend URL: `https://traffic-assessment-1.onrender.com/`

#### Step 4: Update Frontend Configuration

Update `.env` file:
```
BACKEND_URL=https://traffic-assessment-1.onrender.com/
```

### Deploy Frontend to Streamlit Cloud

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **"New app"**
4. Select your repository
5. Set main file: `app.py`
6. Add secrets:
   ```toml
   BACKEND_URL = "https://traffic-assessment-1.onrender.com/"
   ```
7. Click **"Deploy"**

---

## 📁 Project Structure

```
traffic-prediction/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
├── 📄 render.yaml                  # Render deployment config
│
├── 🐍 model.py                     # Model training script
├── 🐍 backend.py                   # Flask REST API
├── 🐍 app.py                       # Streamlit frontend
│
├── 📂 models/                      # Trained model artifacts
│   ├── traffic_model.pkl
│   ├── scaler.pkl
│   ├── le_weather_main.pkl
│   ├── le_weather_desc.pkl
│   ├── feature_columns.pkl
│   └── metrics.pkl
│
├── 📂 plots/                       # Generated visualizations
│   ├── 01_traffic_distribution.png
│   ├── 02_temporal_patterns.png
│   ├── 03_model_evaluation.png
│   └── 04_time_series_prediction.png
│
├── 📂 data/                        # Data files
│   └── feature_importance.csv
│
└── 📂 .streamlit/                  # Streamlit configuration
    └── secrets.toml                # Streamlit secrets (local only)
```

---

## 🛠️ Technologies Used

### Machine Learning
- **scikit-learn** - Model training and evaluation
- **pandas** - Data manipulation
- **numpy** - Numerical computations

### Backend
- **Flask** - REST API framework
- **Flask-CORS** - Cross-origin resource sharing
- **gunicorn** - Production WSGI server

### Frontend
- **Streamlit** - Interactive web application
- **Plotly** - Interactive visualizations
- **requests** - HTTP client

### Utilities
- **joblib** - Model serialization
- **python-dotenv** - Environment variable management

### Deployment
- **Render** - Backend hosting
- **Streamlit Cloud** - Frontend hosting
- **Git/GitHub** - Version control

---


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) - Metro Interstate Traffic Volume Dataset
- **Inspiration**: Smart City AI applications for traffic management
- **Tools**: Streamlit, Flask, scikit-learn communities

---


<div align="center">

**⭐ Star this repo if you find it helpful!**


</div>