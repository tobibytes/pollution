from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained models and scaler
# Note: You need to save these during your model training phase
def load_models():
    models = {
        'Linear Regression': joblib.load('models/linear_regression.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'Gradient Boosting': joblib.load('models/gradient_boosting.pkl'),
        'Neural Network': tf.keras.models.load_model('models/neural_network')
    }
    scaler = joblib.load('models/scaler.pkl')
    return models, scaler

# Load feature names
feature_names = joblib.load('models/feature_names.pkl')

# Prediction function (adapted from your code)
def predict_asthma(year, so2, pm2_5, location_id, models, scaler):
    # Convert PM2.5 to the same scale used in training
    pm2_5 = pm2_5 * 10**8
    
    # Create basic features
    features = {
        'YEAR': year,
        'SO2': so2,
        'PM2.5': pm2_5,
        'SO2_squared': so2 ** 2,
        'PM2.5_squared': pm2_5 ** 2,
        'SO2_PM2.5': so2 * pm2_5
    }
    
    # Create a DataFrame with all possible hospital IDs set to 0
    hospital_columns = [col for col in feature_names if col.startswith('Hospital_ID_')]
    for col in hospital_columns:
        features[col] = 0
    
    # Set the specific hospital ID if provided
    if location_id is not None and 0 <= location_id <= 11:
        features[f'Hospital_ID_{location_id}'] = 1
    
    # Convert to DataFrame with proper column order
    new_data = pd.DataFrame([features])
    new_data = new_data[feature_names]
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions with each model
    predictions = {}
    for name, model in models.items():
        if name == 'Neural Network':
            prediction = model.predict(new_data_scaled)[0][0]
        elif name == 'Linear Regression':
            prediction = model.predict(new_data_scaled)[0]
        else:
            prediction = model.predict(new_data)[0]
        
        # Ensure predictions are not negative
        prediction = max(0, prediction)
        predictions[name] = round(prediction)
    
    return predictions

# Create API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        year = float(data['year'])
        so2 = float(data['so2'])
        pm25 = float(data['pm25'])
        hospital_id = int(data['hospital']) if data['hospital'] else None
        
        # Load models (in production, load once at startup)
        models, scaler = load_models()
        
        # Make prediction
        predictions = predict_asthma(year, so2, pm25, hospital_id, models, scaler)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)