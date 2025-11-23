"""
Streamlit Frontend for Traffic Volume Prediction
Enhanced UI with professional design
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    pass  # python-dotenv not installed, will use default values

# Configuration
st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL - Try multiple sources in order of priority
# 1. Streamlit secrets (for Streamlit Cloud deployment)
# 2. Environment variable from .env file
# 3. Default localhost
try:
    BACKEND_URL = st.secrets.get("BACKEND_URL")
except:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card styles */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 300;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-unit {
        font-size: 0.8rem;
        opacity: 0.8;
        font-weight: 300;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Icon styles */
    .icon {
        font-size: 1.5rem;
    }
    
    /* Traffic level indicators */
    .traffic-level {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
    }
    
    .traffic-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    .traffic-moderate {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }
    
    .traffic-heavy {
        background: linear-gradient(135deg, #fff3cd 0%, #ffc107 100%);
        color: #856404;
    }
    
    .traffic-very-heavy {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🚦 Traffic Volume Prediction System</h1>
    <p>AI-Powered Real-Time Traffic Analytics & Forecasting</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Check backend connection
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            st.markdown('<div class="status-badge status-success">✓ Backend Connected</div>', unsafe_allow_html=True)
            backend_status = True
        else:
            st.markdown('<div class="status-badge status-error">✗ Backend Error</div>', unsafe_allow_html=True)
            backend_status = False
    except:
        st.markdown('<div class="status-badge status-error">✗ Backend Offline</div>', unsafe_allow_html=True)
        st.caption(f"URL: {BACKEND_URL}")
        backend_status = False
    
    st.markdown("---")
    
    # Mode selection
    st.markdown("### 📊 Select Mode")
    mode = st.radio(
        "Choose operation mode:",
        ["🎯 Single Prediction", "📦 Batch Prediction", "📈 Historical Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
   # Model info
    st.markdown("### 🤖 Model Information")

    st.markdown("""
    <style>
    .info-box {
        color: #000000 !important;      /* Force dark text */
        font-size: 16px;
    }
    </style>

    <div class="info-box">
        <strong>Algorithm:</strong> Random Forest<br>
        <strong>Features:</strong> 19 variables<br>
        <strong>Training:</strong> 38K+ samples
    </div>
    """, unsafe_allow_html=True)

        
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This ML-powered system predicts traffic volume using:
        - Historical traffic patterns
        - Weather conditions
        - Temporal features
        - Road network data
        
        **Capabilities:**
        - Real-time predictions
        - Batch processing
        - 24-hour forecasting
        - Pattern analysis
        """)

# Check backend status
if not backend_status:
    st.markdown("""
    <div class="error-box">
        <strong>⚠️ Connection Error</strong><br>
        Cannot connect to the backend API. Please ensure the backend service is running.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Get and display model metrics
try:
    metrics_response = requests.get(f"{BACKEND_URL}/metrics")
    if metrics_response.status_code == 200:
        metrics = metrics_response.json()
        
        # Display metrics in custom cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value">{metrics['test_metrics']['MAE']:.2f}</div>
                <div class="metric-unit">vehicles</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Root Mean Squared Error</div>
                <div class="metric-value">{metrics['test_metrics']['RMSE']:.2f}</div>
                <div class="metric-unit">vehicles</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{metrics['test_metrics']['R2_Score']:.4f}</div>
                <div class="metric-unit">accuracy</div>
            </div>
            """, unsafe_allow_html=True)
except:
    pass

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# SINGLE PREDICTION MODE
# ============================================================================
if mode == "🎯 Single Prediction":
    st.markdown('<div class="section-header"><span class="icon">🎯</span> Single Traffic Volume Prediction</div>', unsafe_allow_html=True)
    
    # Get weather options
    try:
        weather_response = requests.get(f"{BACKEND_URL}/weather_options")
        weather_data = weather_response.json()
        weather_main_options = weather_data['weather_main']
        weather_desc_options = weather_data['weather_description']
    except:
        weather_main_options = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist', 'Fog', 'Drizzle']
        weather_desc_options = ['clear sky', 'scattered clouds', 'light rain', 'few clouds', 'overcast clouds']
    
    # Input form in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Temporal Information")
        with st.container():
            date_input = st.date_input("Select Date", datetime.now())
            time_input = st.time_input("Select Time", datetime.now().time())
            is_holiday = st.checkbox("Public Holiday", value=False)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌡️ Weather Conditions")
        with st.container():
            temp = st.slider("Temperature (Kelvin)", 250.0, 320.0, 280.0, 0.5)
            temp_celsius = temp - 273.15
            temp_fahrenheit = temp_celsius * 9/5 + 32
            st.markdown(f"""
            <div class="info-box">
                📊 <strong>{temp_celsius:.1f}°C</strong> / <strong>{temp_fahrenheit:.1f}°F</strong>
            </div>
            """, unsafe_allow_html=True)
            
            clouds = st.slider("Cloud Coverage (%)", 0, 100, 50)
            
            col_rain, col_snow = st.columns(2)
            with col_rain:
                rain = st.number_input("Rain (mm/h)", 0.0, 100.0, 0.0, 0.1, format="%.1f")
            with col_snow:
                snow = st.number_input("Snow (mm/h)", 0.0, 10.0, 0.0, 0.1, format="%.1f")
    
    with col2:
        st.markdown("#### 🌦️ Weather Description")
        with st.container():
            weather_main = st.selectbox("Primary Weather", weather_main_options, index=0)
            weather_desc = st.selectbox("Detailed Condition", weather_desc_options, index=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Historical Traffic Context")
        st.markdown("""
        <div class="info-box">
            💡 <strong>Tip:</strong> Providing recent traffic data improves prediction accuracy
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            traffic_lag_1 = st.number_input("Traffic 1 hour ago", 0, 8000, 3000, 100, help="Number of vehicles")
            traffic_lag_2 = st.number_input("Traffic 2 hours ago", 0, 8000, 3000, 100, help="Number of vehicles")
            traffic_lag_3 = st.number_input("Traffic 3 hours ago", 0, 8000, 3000, 100, help="Number of vehicles")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("🚀 Generate Prediction", key="predict_single", use_container_width=True):
        dt = datetime.combine(date_input, time_input)
        
        payload = {
            "temp": temp,
            "rain_1h": rain,
            "snow_1h": snow,
            "clouds_all": clouds,
            "weather_main": weather_main,
            "weather_description": weather_desc,
            "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "is_holiday": 1 if is_holiday else 0,
            "traffic_lag_1": traffic_lag_1,
            "traffic_lag_2": traffic_lag_2,
            "traffic_lag_3": traffic_lag_3
        }
        
        with st.spinner("🔄 Processing prediction..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    
                    st.markdown("""
                    <div class="success-box">
                        <strong>✅ Prediction Generated Successfully</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create gauge chart with modern styling
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted Traffic Volume", 'font': {'size': 28, 'color': '#2c3e50', 'family': 'Inter'}},
                        delta={'reference': 3260, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#27ae60"}},
                        number={'font': {'size': 50, 'color': '#667eea'}},
                        gauge={
                            'axis': {
                                'range': [None, 8000],
                                'tickwidth': 2,
                                'tickcolor': "#34495e",
                                'tickfont': {'size': 14}
                            },
                            'bar': {'color': "#667eea", 'thickness': 0.75},
                            'bgcolor': "white",
                            'borderwidth': 3,
                            'bordercolor': "#ecf0f1",
                            'steps': [
                                {'range': [0, 2000], 'color': '#d4edda'},
                                {'range': [2000, 4000], 'color': '#fff3cd'},
                                {'range': [4000, 6000], 'color': '#ffe5b4'},
                                {'range': [6000, 8000], 'color': '#f8d7da'}
                            ],
                            'threshold': {
                                'line': {'color': "#c0392b", 'width': 6},
                                'thickness': 0.85,
                                'value': prediction
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=450,
                        margin=dict(l=20, r=20, t=80, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'family': 'Inter'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Traffic level interpretation
                    if prediction < 2000:
                        st.markdown("""
                        <div class="traffic-level traffic-low">
                            🟢 LOW TRAFFIC - Smooth Flow Expected
                        </div>
                        """, unsafe_allow_html=True)
                    elif prediction < 4000:
                        st.markdown("""
                        <div class="traffic-level traffic-moderate">
                            🟡 MODERATE TRAFFIC - Normal Conditions
                        </div>
                        """, unsafe_allow_html=True)
                    elif prediction < 6000:
                        st.markdown("""
                        <div class="traffic-level traffic-heavy">
                            🟠 HEAVY TRAFFIC - Expect Minor Delays
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="traffic-level traffic-very-heavy">
                            🔴 VERY HEAVY TRAFFIC - Significant Congestion
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional insights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        congestion_level = (prediction / 7280) * 100
                        st.metric("Congestion Level", f"{congestion_level:.1f}%")
                    with col2:
                        est_delay = max(0, (prediction - 3000) / 100)
                        st.metric("Est. Delay", f"{est_delay:.0f} min")
                    with col3:
                        capacity = max(0, 100 - congestion_level)
                        st.metric("Road Capacity", f"{capacity:.1f}%")
                    
                    # Show details
                    with st.expander("📋 View Detailed Results"):
                        st.json(result)
                else:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>❌ Error:</strong> {response.json().get('error', 'Unknown error')}
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <strong>❌ Error:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# BATCH PREDICTION MODE
# ============================================================================
elif mode == "📦 Batch Prediction":
    st.markdown('<div class="section-header"><span class="icon">📦</span> Batch Traffic Volume Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📤 Bulk Processing:</strong> Upload CSV files or generate sample forecasts for multiple time periods
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📄 Upload CSV", "🔮 Generate Forecast"])
    
    with tab1:
        st.markdown("#### Upload Your Data File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with required columns: temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("##### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown(f"""
            <div class="info-box">
                📈 <strong>Dataset:</strong> {len(df)} records loaded
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Process Batch Predictions", key="predict_batch_upload", use_container_width=True):
                with st.spinner("🔄 Processing batch predictions..."):
                    try:
                        predictions_payload = {
                            "predictions": df.to_dict('records')
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/predict_batch",
                            json=predictions_payload,
                            headers={'Content-Type': 'application/json'}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            df['predicted_traffic_volume'] = result['predictions']
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✅ Success:</strong> Predicted {result['count']} records
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show results
                            st.markdown("##### 📊 Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "⬇️ Download Results CSV",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                key='download-csv',
                                use_container_width=True
                            )
                            
                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=df['predicted_traffic_volume'],
                                mode='lines+markers',
                                name='Predicted Traffic',
                                line=dict(color='#667eea', width=3),
                                marker=dict(size=6, color='#764ba2')
                            ))
                            
                            fig.update_layout(
                                title="Traffic Volume Predictions",
                                xaxis_title="Record Index",
                                yaxis_title="Traffic Volume (vehicles)",
                                hovermode='x unified',
                                height=500,
                                template='plotly_white',
                                font=dict(family='Inter')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average", f"{df['predicted_traffic_volume'].mean():.0f}")
                            with col2:
                                st.metric("Maximum", f"{df['predicted_traffic_volume'].max():.0f}")
                            with col3:
                                st.metric("Minimum", f"{df['predicted_traffic_volume'].min():.0f}")
                            with col4:
                                st.metric("Std Dev", f"{df['predicted_traffic_volume'].std():.0f}")
                        else:
                            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("#### Generate 24-Hour Traffic Forecast")
        
        col1, col2 = st.columns(2)
        with col1:
            sample_date = st.date_input("Forecast Date", datetime.now())
            sample_temp = st.slider("Average Temperature (K)", 250.0, 320.0, 280.0)
        with col2:
            sample_weather = st.selectbox("Weather Condition", ['Clear', 'Clouds', 'Rain', 'Snow'])
            sample_clouds = st.slider("Cloud Coverage (%)", 0, 100, 50)
        
        if st.button("🔮 Generate 24h Forecast", key="generate_sample", use_container_width=True):
            with st.spinner("🔄 Generating hourly forecasts..."):
                predictions_list = []
                for hour in range(24):
                    dt = datetime.combine(sample_date, datetime.min.time()) + timedelta(hours=hour)
                    predictions_list.append({
                        "temp": sample_temp,
                        "rain_1h": 0.0,
                        "snow_1h": 0.0,
                        "clouds_all": sample_clouds,
                        "weather_main": sample_weather,
                        "weather_description": f"{sample_weather.lower()} sky",
                        "date_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "is_holiday": 0,
                        "traffic_lag_1": 3000,
                        "traffic_lag_2": 3000,
                        "traffic_lag_3": 3000
                    })
                
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/predict_batch",
                        json={"predictions": predictions_list},
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        results_df = pd.DataFrame(predictions_list)
                        results_df['predicted_traffic_volume'] = result['predictions']
                        results_df['hour'] = range(24)
                        
                        st.markdown("""
                        <div class="success-box">
                            <strong>✅ 24-Hour Forecast Generated</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Advanced visualization
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=results_df['hour'],
                            y=results_df['predicted_traffic_volume'],
                            mode='lines+markers',
                            name='Traffic Volume',
                            line=dict(color='#667eea', width=4),
                            marker=dict(size=10, color='#764ba2'),
                            fill='tozeroy',
                            fillcolor='rgba(102, 126, 234, 0.1)'
                        ))
                        
                        # Add rush hour zones
                        fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.1, layer="below", line_width=0)
                        fig.add_vrect(x0=16, x1=18, fillcolor="red", opacity=0.1, layer="below", line_width=0)
                        
                        fig.update_layout(
                            title={
                                'text': f"24-Hour Traffic Forecast - {sample_date.strftime('%B %d, %Y')}",
                                'x': 0.5,
                                'xanchor': 'center',
                                'font': {'size': 24, 'family': 'Inter', 'color': '#2c3e50'}
                            },
                            xaxis_title="Hour of Day",
                            yaxis_title="Traffic Volume (vehicles)",
                            hovermode='x unified',
                            height=500,
                            template='plotly_white',
                            font=dict(family='Inter', size=12),
                            xaxis=dict(
                                tickmode='linear',
                                tick0=0,
                                dtick=2,
                                showgrid=True,
                                gridcolor='rgba(0,0,0,0.1)'
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(0,0,0,0.1)'
                            ),
                            annotations=[
                                dict(x=8, y=results_df['predicted_traffic_volume'].max(),
                                     text="Morning Rush", showarrow=False,
                                     font=dict(size=10, color='red')),
                                dict(x=17, y=results_df['predicted_traffic_volume'].max(),
                                     text="Evening Rush", showarrow=False,
                                     font=dict(size=10, color='red'))
                            ]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            peak_hour = results_df.loc[results_df['predicted_traffic_volume'].idxmax(), 'hour']
                            st.metric("Peak Hour", f"{int(peak_hour)}:00")
                        with col2:
                            peak_volume = results_df['predicted_traffic_volume'].max()
                            st.metric("Peak Volume", f"{peak_volume:.0f}")
                        with col3:
                            avg_volume = results_df['predicted_traffic_volume'].mean()
                            st.metric("Average", f"{avg_volume:.0f}")
                        with col4:
                            low_hour = results_df.loc[results_df['predicted_traffic_volume'].idxmin(), 'hour']
                            st.metric("Lowest Hour", f"{int(low_hour)}:00")
                        
                        # Data table
                        with st.expander("📊 View Hourly Data"):
                            display_df = results_df[['hour', 'predicted_traffic_volume']].copy()
                            display_df['hour'] = display_df['hour'].apply(lambda x: f"{int(x):02d}:00")
                            display_df.columns = ['Time', 'Predicted Volume']
                            st.dataframe(display_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download 24h Forecast",
                            csv,
                            f"forecast_{sample_date.strftime('%Y%m%d')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# HISTORICAL ANALYSIS MODE
# ============================================================================
else:
    st.markdown('<div class="section-header"><span class="icon">📈</span> Historical Traffic Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📊 Insights:</strong> Explore patterns and trends from the training data analysis
    </div>
    """, unsafe_allow_html=True)
    
    # Show plots if available
    import os
    plots_dir = "plots"
    
    if os.path.exists(plots_dir):
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
        
        if plot_files:
            # Create tabs for different analyses
            tabs = st.tabs(["📊 Distribution", "⏰ Temporal Patterns", "🎯 Model Performance", "📈 Time Series"])
            
            for i, (tab, plot_file) in enumerate(zip(tabs, plot_files)):
                with tab:
                    # Create a nice title from filename
                    plot_title = plot_file.replace('_', ' ').replace('.png', '').title()
                    plot_title = plot_title.replace('01 ', '').replace('02 ', '').replace('03 ', '').replace('04 ', '')
                    
                    st.markdown(f"#### {plot_title}")
                    
                    # Display image with border
                    col1, col2, col3 = st.columns([1, 10, 1])
                    with col2:
                        st.image(
                            os.path.join(plots_dir, plot_file),
                            use_column_width=True
                        )
                    
                    # Add insights based on plot type
                    if 'distribution' in plot_file.lower():
                        st.markdown("""
                        <div class="info-box">
                            <strong>💡 Key Insights:</strong>
                            <ul>
                                <li>Traffic volume distribution shows typical highway patterns</li>
                                <li>Most readings fall between 1,000-5,000 vehicles</li>
                                <li>Peak hours show significantly higher volumes</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif 'temporal' in plot_file.lower():
                        st.markdown("""
                        <div class="info-box">
                            <strong>💡 Key Insights:</strong>
                            <ul>
                                <li>Clear rush hour patterns at 7-9 AM and 4-6 PM</li>
                                <li>Weekday traffic significantly higher than weekends</li>
                                <li>Weather conditions strongly impact traffic volume</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif 'evaluation' in plot_file.lower():
                        st.markdown("""
                        <div class="info-box">
                            <strong>💡 Key Insights:</strong>
                            <ul>
                                <li>Model shows strong predictive accuracy</li>
                                <li>Historical traffic (lag features) are most important</li>
                                <li>Temperature and time features also significant</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif 'time_series' in plot_file.lower():
                        st.markdown("""
                        <div class="info-box">
                            <strong>💡 Key Insights:</strong>
                            <ul>
                                <li>Model captures temporal patterns effectively</li>
                                <li>Predictions closely follow actual trends</li>
                                <li>Good performance across different traffic levels</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Additional statistics section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Dataset Statistics")
            
            # Try to load feature importance if available
            try:
                if os.path.exists('data/feature_importance.csv'):
                    feature_importance = pd.read_csv('data/feature_importance.csv')
                    
                    st.markdown("#### Top 10 Most Important Features")
                    
                    # Create horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=feature_importance.head(10)['importance'],
                        y=feature_importance.head(10)['feature'],
                        orientation='h',
                        marker=dict(
                            color=feature_importance.head(10)['importance'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=feature_importance.head(10)['importance'].round(3),
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title="Feature Importance Analysis",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature",
                        height=500,
                        template='plotly_white',
                        font=dict(family='Inter'),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show full table in expander
                    with st.expander("📋 View All Features"):
                        st.dataframe(feature_importance, use_container_width=True)
            except:
                pass
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ No Plots Available</strong><br>
                Run <code>model.py</code> to generate analysis plots.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Plots Directory Not Found</strong><br>
            Run <code>model.py</code> to generate the plots directory and analysis visualizations.
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="info-box">
            <strong>📝 Instructions:</strong>
            <ol>
                <li>Ensure you have the training dataset</li>
                <li>Run: <code>python model.py</code></li>
                <li>Wait for training to complete</li>
                <li>Refresh this page to view analysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #7f8c8d;'>
    <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
        <strong>Traffic Volume Prediction System</strong>
    </p>
    <p style='font-size: 0.8rem; margin-bottom: 0.5rem;'>
        Powered by Machine Learning | Random Forest Algorithm
    </p>
    <p style='font-size: 0.75rem; color: #95a5a6;'>
        Built with Streamlit 🎈 | Backend API on Render ☁️
    </p>
</div>
""", unsafe_allow_html=True)