"""
Flask Backend API for Traffic Volume Prediction
Deploy this on Render as a Web Service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder
CORS(app)  # Enable CORS for cross-origin requests

# Load models and artifacts
MODEL_DIR = 'models'

print("Loading models...")
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'traffic_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le_weather_main = joblib.load(os.path.join(MODEL_DIR, 'le_weather_main.pkl'))
    le_weather_desc = joblib.load(os.path.join(MODEL_DIR, 'le_weather_desc.pkl'))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    metrics = joblib.load(os.path.join(MODEL_DIR, 'metrics.pkl'))
    print("✓ All models loaded successfully!")
    print(f"✓ Feature columns: {feature_columns}")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODEL_LOADED = False
    model = None

# Helper function to prepare features
def prepare_features(data):
    """
    Prepare features from input data and return them in the correct order
    """
    # Parse datetime if provided, else use current time
    if 'date_time' in data and data['date_time']:
        dt = pd.to_datetime(data['date_time'])
    else:
        dt = datetime.now()
    
    # Extract temporal features
    features_dict = {
        'temp': float(data.get('temp', 280.0)),
        'rain_1h': float(data.get('rain_1h', 0.0)),
        'snow_1h': float(data.get('snow_1h', 0.0)),
        'clouds_all': float(data.get('clouds_all', 50)),
        'hour': float(dt.hour),
        'day_of_week': float(dt.weekday()),
        'month': float(dt.month),
        'day': float(dt.day),
        'is_weekend': float(1 if dt.weekday() >= 5 else 0),
        'is_morning_rush': float(1 if 7 <= dt.hour <= 9 else 0),
        'is_evening_rush': float(1 if 16 <= dt.hour <= 18 else 0),
        'is_holiday': float(data.get('is_holiday', 0))
    }
    
    # Encode weather
    weather_main = data.get('weather_main', 'Clear')
    weather_desc = data.get('weather_description', 'clear sky')
    
    try:
        features_dict['weather_main_encoded'] = float(le_weather_main.transform([weather_main])[0])
    except:
        features_dict['weather_main_encoded'] = 0.0
    
    try:
        features_dict['weather_desc_encoded'] = float(le_weather_desc.transform([weather_desc])[0])
    except:
        features_dict['weather_desc_encoded'] = 0.0
    
    # Lag features (use provided or defaults)
    features_dict['traffic_lag_1'] = float(data.get('traffic_lag_1', 3000))
    features_dict['traffic_lag_2'] = float(data.get('traffic_lag_2', 3000))
    features_dict['traffic_lag_3'] = float(data.get('traffic_lag_3', 3000))
    features_dict['traffic_rolling_mean_3'] = float(data.get('traffic_rolling_mean_3', 3000))
    features_dict['traffic_rolling_std_3'] = float(data.get('traffic_rolling_std_3', 500))
    
    return features_dict

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint with API information
    """
    return jsonify({
        'message': 'Traffic Volume Prediction API',
        'version': '1.0',
        'status': 'running',
        'model_loaded': MODEL_LOADED,
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /metrics': 'Model performance metrics',
            'POST /predict': 'Predict traffic volume',
            'POST /predict_batch': 'Batch predictions',
            'GET /weather_options': 'Available weather options'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'unhealthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get model performance metrics
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'test_metrics': {
            'MAE': round(metrics['test_mae'], 2),
            'RMSE': round(metrics['test_rmse'], 2),
            'R2_Score': round(metrics['test_r2'], 4)
        },
        'train_metrics': {
            'MAE': round(metrics['train_mae'], 2),
            'RMSE': round(metrics['train_rmse'], 2),
            'R2_Score': round(metrics['train_r2'], 4)
        }
    })

@app.route('/weather_options', methods=['GET'])
def weather_options():
    """
    Get available weather options
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'weather_main': le_weather_main.classes_.tolist(),
        'weather_description': le_weather_desc.classes_.tolist()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Expected JSON payload:
    {
        "temp": 280.5,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 40,
        "weather_main": "Clouds",
        "weather_description": "scattered clouds",
        "date_time": "2024-01-15 14:30:00",  (optional)
        "is_holiday": 0,
        "traffic_lag_1": 3500,  (optional)
        "traffic_lag_2": 3400,  (optional)
        "traffic_lag_3": 3300   (optional)
    }
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        features_dict = prepare_features(data)
        
        # Create feature array in the CORRECT ORDER from feature_columns
        feature_array = np.array([[features_dict.get(col, 0) for col in feature_columns]])
        
        print(f"Feature columns order: {feature_columns}")
        print(f"Feature values: {feature_array}")
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_array_scaled)[0]
        
        # Ensure non-negative
        prediction = max(0, float(prediction))
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'unit': 'vehicles',
            'input_features': features_dict,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON payload:
    {
        "predictions": [
            {data1},
            {data2},
            ...
        ]
    }
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'No predictions data provided'}), 400
        
        predictions = []
        
        for item in data['predictions']:
            try:
                features_dict = prepare_features(item)
                feature_array = np.array([[features_dict.get(col, 0) for col in feature_columns]])
                feature_array_scaled = scaler.transform(feature_array)
                prediction = model.predict(feature_array_scaled)[0]
                prediction = max(0, float(prediction))
                predictions.append(round(prediction, 2))
            except Exception as e:
                print(f"Error processing item: {e}")
                predictions.append(None)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len([p for p in predictions if p is not None]),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in predict_batch: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)