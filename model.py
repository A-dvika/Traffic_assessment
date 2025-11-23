"""
Traffic Volume Forecasting Model
=================================
Complete pipeline: EDA -> Feature Engineering -> Training -> Model Saving
Dataset: Metro Interstate Traffic Volume
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create directories for outputs
import os
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*60)
print("TRAFFIC VOLUME FORECASTING - MODEL TRAINING PIPELINE")
print("="*60)

# ============================================================================
# STEP 1: DATA ACQUISITION & LOADING
# ============================================================================
print("\n[STEP 1] Loading Data...")

# Load dataset
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[STEP 2] Exploratory Data Analysis...")

# Basic info
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Statistical Summary ---")
print(df.describe())

# Check missing values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")

# Parse datetime
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n--- Duplicates: {duplicates} ---")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed duplicates. New shape: {df.shape}")

# Traffic volume distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['traffic_volume'], bins=50, edgecolor='black')
plt.title('Traffic Volume Distribution')
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(df['traffic_volume'])
plt.title('Traffic Volume Boxplot')
plt.ylabel('Traffic Volume')
plt.tight_layout()
plt.savefig('plots/01_traffic_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/01_traffic_distribution.png")
plt.close()

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 3] Feature Engineering...")

# Extract temporal features
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['day'] = df['date_time'].dt.day
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Rush hour indicators
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)

# Holiday encoding (0 if None, 1 if holiday)
df['is_holiday'] = (df['holiday'] != 'None').astype(int)

# Encode categorical weather variables
le_weather_main = LabelEncoder()
le_weather_desc = LabelEncoder()

df['weather_main_encoded'] = le_weather_main.fit_transform(df['weather_main'])
df['weather_desc_encoded'] = le_weather_desc.fit_transform(df['weather_description'])

# Save temporal patterns BEFORE creating lag features
print("Creating temporal pattern visualizations...")

# Temporal patterns visualization (BEFORE dropping NaN)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Hourly pattern
hourly_avg = df.groupby('hour')['traffic_volume'].mean()
axes[0, 0].bar(hourly_avg.index, hourly_avg.values, color='steelblue')
axes[0, 0].set_title('Average Traffic Volume by Hour', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Average Traffic Volume')
axes[0, 0].grid(axis='y', alpha=0.3)

# Day of week pattern
daily_avg = df.groupby('day_of_week')['traffic_volume'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0, 1].bar(daily_avg.index, daily_avg.values, color='coral')
axes[0, 1].set_title('Average Traffic Volume by Day of Week', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Average Traffic Volume')
axes[0, 1].set_xticks(daily_avg.index)
axes[0, 1].set_xticklabels([days[i] for i in daily_avg.index])
axes[0, 1].grid(axis='y', alpha=0.3)

# Monthly pattern
monthly_avg = df.groupby('month')['traffic_volume'].mean()
axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8)
axes[1, 0].set_title('Average Traffic Volume by Month', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Average Traffic Volume')
axes[1, 0].grid(True, alpha=0.3)

# Weather impact
weather_avg = df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False)
axes[1, 1].barh(range(len(weather_avg)), weather_avg.values, color='lightgreen')
axes[1, 1].set_yticks(range(len(weather_avg)))
axes[1, 1].set_yticklabels(weather_avg.index)
axes[1, 1].set_title('Average Traffic Volume by Weather', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Average Traffic Volume')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/02_temporal_patterns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/02_temporal_patterns.png")
plt.close()

# NOW create lagged features with forward fill for initial NaN values
print("Creating lag features...")
df['traffic_lag_1'] = df['traffic_volume'].shift(1)
df['traffic_lag_2'] = df['traffic_volume'].shift(2)
df['traffic_lag_3'] = df['traffic_volume'].shift(3)

# Rolling statistics
df['traffic_rolling_mean_3'] = df['traffic_volume'].rolling(window=3).mean()
df['traffic_rolling_std_3'] = df['traffic_volume'].rolling(window=3).std()

# Fill NaN in lag features with forward fill method (only for first few rows)
df['traffic_lag_1'].fillna(method='bfill', inplace=True)
df['traffic_lag_2'].fillna(method='bfill', inplace=True)
df['traffic_lag_3'].fillna(method='bfill', inplace=True)
df['traffic_rolling_mean_3'].fillna(method='bfill', inplace=True)
df['traffic_rolling_std_3'].fillna(method='bfill', inplace=True)

print(f"Features created. Dataset shape: {df.shape}")
print(f"Remaining NaN values: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================================================
print("\n[STEP 4] Preparing Data for Modeling...")

# Select features for modeling
feature_columns = [
    'temp', 'rain_1h', 'snow_1h', 'clouds_all',
    'hour', 'day_of_week', 'month', 'day', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_holiday',
    'weather_main_encoded', 'weather_desc_encoded',
    'traffic_lag_1', 'traffic_lag_2', 'traffic_lag_3',
    'traffic_rolling_mean_3', 'traffic_rolling_std_3'
]

X = df[feature_columns]
y = df['traffic_volume']

print(f"Features: {len(feature_columns)}")
print(f"Feature names: {feature_columns}")

# Train-test split (80-20, time-based)
split_idx = int(len(df) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================
print("\n[STEP 5] Training Model...")

# Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest model...")
model.fit(X_train_scaled, y_train)
print("✓ Model training completed!")

# ============================================================================
# STEP 6: MODEL EVALUATION
# ============================================================================
print("\n[STEP 6] Evaluating Model...")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- MODEL PERFORMANCE ---")
print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  R²:   {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  MAE:  {test_mae:.2f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  R²:   {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- TOP 10 IMPORTANT FEATURES ---")
print(feature_importance.head(10))

# Visualize predictions
plt.figure(figsize=(15, 5))

# Plot 1: Predicted vs Actual (Test Set)
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_test_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Predicted vs Actual (Test Set)')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 3, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Traffic Volume')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Feature Importance
plt.subplot(1, 3, 3)
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/03_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/03_model_evaluation.png")
plt.close()

# Time series prediction plot (all test points or last 500, whichever is smaller)
plt.figure(figsize=(15, 5))
plot_points = min(500, len(y_test))
plt.plot(range(plot_points), y_test.values[-plot_points:], label='Actual', linewidth=1.5, alpha=0.7)
plt.plot(range(plot_points), y_test_pred[-plot_points:], label='Predicted', linewidth=1.5, alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Traffic Volume')
plt.title(f'Traffic Volume: Actual vs Predicted (Test Set - {plot_points} Points)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/04_time_series_prediction.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/04_time_series_prediction.png")
plt.close()

# ============================================================================
# STEP 7: SAVE MODEL & ARTIFACTS
# ============================================================================
print("\n[STEP 7] Saving Model and Artifacts...")

# Save model
joblib.dump(model, 'models/traffic_model.pkl', compress=('gzip', 3))
print("✓ Saved: models/traffic_model.pkl")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved: models/scaler.pkl")

# Save label encoders
joblib.dump(le_weather_main, 'models/le_weather_main.pkl')
joblib.dump(le_weather_desc, 'models/le_weather_desc.pkl')
print("✓ Saved: Label encoders")

# Save feature columns
joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("✓ Saved: models/feature_columns.pkl")

# Save metrics
metrics = {
    'train_mae': train_mae,
    'train_rmse': train_rmse,
    'train_r2': train_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'test_r2': test_r2
}
joblib.dump(metrics, 'models/metrics.pkl')
print("✓ Saved: models/metrics.pkl")

# Save feature importance
feature_importance.to_csv('data/feature_importance.csv', index=False)
print("✓ Saved: data/feature_importance.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n✓ Model Type: Random Forest Regressor")
print(f"✓ Features Used: {len(feature_columns)}")
print(f"✓ Training Samples: {len(X_train)}")
print(f"✓ Test Samples: {len(X_test)}")
print(f"\n✓ Test MAE: {test_mae:.2f} vehicles")
print(f"✓ Test RMSE: {test_rmse:.2f} vehicles")
print(f"✓ Test R²: {test_r2:.4f}")
print(f"\n✓ All artifacts saved in 'models/' directory")
print(f"✓ All plots saved in 'plots/' directory")
print("\nReady for deployment!")
print("="*60)