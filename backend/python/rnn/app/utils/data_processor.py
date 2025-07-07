import pandas as pd
import numpy as np
import math

def process_for_prediction(iot_id, df, plant_data):
    """Process data for traditional model prediction"""
    processed = df.copy()

    # Convert timestamp to datetime if needed
    if 'created_at' in processed.columns and not isinstance(processed['created_at'].iloc[0], pd.Timestamp):
        processed['created_at'] = pd.to_datetime(processed['created_at'])

    # Add time-based features
    processed['Hour'] = processed['created_at'].dt.hour
    processed['Day'] = processed['created_at'].dt.day
    processed['Month'] = processed['created_at'].dt.month

    # Get plant history for calculations
    plant_history = plant_data[iot_id].copy()
    plant_history['created_at'] = pd.to_datetime(plant_history['created_at'])
    plant_history = plant_history.sort_values('created_at')

    # Calculate rolling averages if enough history
    if len(plant_history) >= 6:
        for feature in ['soil_temp', 'humidity', 'soil_moisture', 'light_intensity',
                        'ph', 'Nitrogen_Level', 'Phosphorus_Leve', 'Potassium_Level',
                        'temperature', 'Chlorophyll_Content', 'Electrochemical_Signal']:
            if feature in plant_history.columns:
                # Calculate 24h average (equivalent to 6 readings at 4-hour intervals)
                processed[f'{feature}_24h_avg'] = plant_history[feature].rolling(
                    window=min(6, len(plant_history))).mean().iloc[-1]

                # Calculate trend (change per second)
                if len(plant_history) >= 2:
                    first_idx = max(0, len(plant_history) - 6)
                    time_diff = (plant_history['created_at'].iloc[-1] -
                                plant_history['created_at'].iloc[first_idx]).total_seconds()
                    if time_diff > 0:
                        value_diff = (plant_history[feature].iloc[-1] -
                                    plant_history[feature].iloc[first_idx])
                        processed[f'{feature}_trend'] = value_diff / time_diff
                    else:
                        processed[f'{feature}_trend'] = 0
                else:
                    processed[f'{feature}_trend'] = 0
    else:
        # Not enough history, use current values as averages
        for feature in ['soil_temp', 'humidity', 'soil_moisture', 'light_intensity',
                        'ph', 'Nitrogen_Level', 'Phosphorus_Leve', 'Potassium_Level',
                        'temperature', 'Chlorophyll_Content', 'Electrochemical_Signal']:
            if feature in processed.columns:
                processed[f'{feature}_24h_avg'] = processed[feature]
                processed[f'{feature}_trend'] = 0

    # Calculate interaction features
    processed['Temp_Humidity_Interaction'] = processed['soil_temp'] * processed['humidity']

    # NPK Balance calculations
    if all(col in processed.columns for col in ['Nitrogen_Level', 'Phosphorus_Leve', 'Potassium_Level']):
        processed['NPK_Balance'] = (processed['Nitrogen_Level'] +
                                   processed['Phosphorus_Leve'] +
                                   processed['Potassium_Level']) / 3
        # Avoid division by zero
        if processed['NPK_Balance'].iloc[0] > 0:
            processed['NPK_Ratio_N'] = processed['Nitrogen_Level'] / processed['NPK_Balance']
            processed['NPK_Ratio_P'] = processed['Phosphorus_Leve'] / processed['NPK_Balance']
            processed['NPK_Ratio_K'] = processed['Potassium_Level'] / processed['NPK_Balance']
        else:
            processed['NPK_Ratio_N'] = 0
            processed['NPK_Ratio_P'] = 0
            processed['NPK_Ratio_K'] = 0
    else:
        # Set default values if NPK data is missing
        processed['NPK_Balance'] = 0
        processed['NPK_Ratio_N'] = 0
        processed['NPK_Ratio_P'] = 0
        processed['NPK_Ratio_K'] = 0

    # Calculate stress indicators based on quantiles if enough history
    if len(plant_history) >= 10:
        moisture_q25 = plant_history['soil_moisture'].quantile(0.25)
        temp_q25 = plant_history['soil_temp'].quantile(0.25)
        temp_q75 = plant_history['soil_temp'].quantile(0.75)

        if 'light_intensity' in plant_history.columns:
            light_q25 = plant_history['light_intensity'].quantile(0.25)
            processed['Light_Stress'] = (processed['light_intensity'] < light_q25).astype(int)
        else:
            processed['Light_Stress'] = 0

        processed['Moisture_Stress'] = (processed['soil_moisture'] < moisture_q25).astype(int)
        processed['Temperature_Stress'] = ((processed['soil_temp'] > temp_q75) |
                                        (processed['soil_temp'] < temp_q25)).astype(int)
    else:
        # Default values if not enough history
        processed['Moisture_Stress'] = 0
        processed['Temperature_Stress'] = 0
        processed['Light_Stress'] = 0

    return processed

def process_for_lstm(df):
    """Create features needed for LSTM prediction"""
    processed = df.copy()
    
    # Ensure timestamp is processed
    if 'created_at' in processed.columns and not isinstance(processed['created_at'].iloc[0], pd.Timestamp):
        processed['created_at'] = pd.to_datetime(processed['created_at'])
    
    # Add temporal features
    processed['Hour'] = processed['created_at'].dt.hour
    processed['Day'] = processed['created_at'].dt.day
    processed['Month'] = processed['created_at'].dt.month
    
    # Add cyclical time features
    processed['Hour_Sin'] = np.sin(processed['Hour'] * (2 * np.pi / 24))
    processed['Hour_Cos'] = np.cos(processed['Hour'] * (2 * np.pi / 24))
    processed['Day_Sin'] = np.sin(processed['Day'] * (2 * np.pi / 31))
    processed['Day_Cos'] = np.cos(processed['Day'] * (2 * np.pi / 31))
    processed['Month_Sin'] = np.sin(processed['Month'] * (2 * np.pi / 12))
    processed['Month_Cos'] = np.cos(processed['Month'] * (2 * np.pi / 12))
    
    # Add interaction features
    if 'Soil_Moisture' in processed.columns and 'Soil_Temperature' in processed.columns:
        processed['Moisture_Temp_Interaction'] = processed['Soil_Moisture'] * processed['Soil_Temperature'] / 100
        
    if 'Humidity' in processed.columns and 'Ambient_Temperature' in processed.columns:
        processed['Humidity_Temp_Interaction'] = processed['Humidity'] * processed['Ambient_Temperature'] / 100
        
    if all(col in processed.columns for col in ['Nitrogen_Level', 'Phosphorus_Leve', 'Potassium_Level']):
        processed['NPK_Balance'] = (processed['Nitrogen_Level'] + processed['Phosphorus_Leve'] + 
                                  processed['Potassium_Level']) / 3
    
    # Calculate rolling features if enough data
    for feature in ['Soil_Moisture', 'Soil_Temperature', 'Humidity', 'Light_Intensity']:
        if feature in processed.columns and len(processed) > 2:
            processed[f'{feature}_rolling_mean'] = processed[feature].rolling(
                window=min(3, len(processed))).mean().fillna(processed[feature])
            processed[f'{feature}_change_rate'] = processed[feature].pct_change().fillna(0)
    
    return processed

def prepare_lstm_sequence(iot_id, plant_data, model_service):
    """Convert latest data to proper sequence format for LSTM"""
    if not model_service.model_config or not model_service.feature_columns or not model_service.feature_scaler:
        print("LSTM model not properly loaded")
        return None
        
    try:
        # Get sequence length from config
        sequence_length = model_service.get_sequence_length()
        
        # Get plant history for sequence
        plant_history = plant_data[iot_id].copy()
        plant_history['created_at'] = pd.to_datetime(plant_history['created_at'])
        plant_history = plant_history.sort_values('created_at')
        
        # Need at least sequence_length data points
        if len(plant_history) < sequence_length:
            print(f"Not enough history for plant {iot_id} to create sequence")
            return None
            
        # Get the last sequence_length records
        sequence_data = plant_history.iloc[-sequence_length:].copy()
        
        # Prepare features with engineering
        sequence_data = process_for_lstm(sequence_data)
        
        # Extract just the needed features in the right order
        X = np.zeros((1, sequence_length, len(model_service.feature_columns)))
        
        for i, feature in enumerate(model_service.feature_columns):
            if feature in sequence_data.columns:
                X[0, :, i] = sequence_data[feature].values
            # If feature is missing, leave as zeros
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = model_service.feature_scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        return X_scaled
        
    except Exception as e:
        print(f"Error preparing LSTM sequence: {e}")
        return None