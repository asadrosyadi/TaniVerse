import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from app.utils.data_processor import process_for_prediction, prepare_lstm_sequence
DEFAULT_iot_id = "jTZids5M"  # Default iot_id


class ForecastService:
    """Service for generating plant health forecasts"""
    
    def __init__(self, model_service, data_service):
        self.model_service = model_service
        self.data_service = data_service
    
    def generate_forecast(self, iot_id, days=3):
        """Generate forecast using best available model"""
        # Try LSTM model if available
        if self.model_service.lstm_model is not None:
            lstm_forecast = self.generate_lstm_forecast(iot_id, days)
            if lstm_forecast is not None:
                return lstm_forecast
        
        # Fall back to traditional model
        return self.generate_traditional_forecast(iot_id, days)
    
    def generate_lstm_forecast(self, iot_id, days=3):
        """Generate forecast using LSTM model"""
        try:
            # Prepare sequence for LSTM
            X_sequence = prepare_lstm_sequence(
                iot_id, 
                self.data_service.plant_data, 
                self.model_service
            )
            
            if X_sequence is None:
                print("Could not prepare LSTM sequence.")
                return None
                
            # Get current values for building forecast
            plant_history = self.data_service.get_plant_data(iot_id)
            plant_history['created_at'] = pd.to_datetime(plant_history['created_at'])
            plant_history = plant_history.sort_values('created_at')
            latest_data = plant_history.iloc[-1:].copy()
            timestamp = latest_data['created_at'].iloc[0]
                
            # Create forecast datapoints
            forecast = []
            
            # Save first data point as current
            current_entry = {
                'date': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                #'created_at': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'forecast_type': 'current',
                'soil_temp': float(latest_data['soil_temp'].iloc[0]),
                'humidity': float(latest_data['humidity'].iloc[0]),
                'soil_moisture': float(latest_data['soil_moisture'].iloc[0])
            }
            
            # Make prediction for current point
            prediction = self.model_service.predict_lstm(X_sequence)
            if prediction:
                current_entry['predicted_health'] = prediction['predicted_health']
                current_entry['confidence'] = prediction['confidence']
            
            # Add optional fields
            for src, dst in [('temperature', 'temperature'), 
                         ('light_intensity', 'light_intensity'),
                         ('ph', 'ph'),
                         ('Nitrogen_Level', 'nitrogen'),
                         ('Phosphorus_Level', 'phosphorus'),
                         ('Potassium_Level', 'potassium')]:
                if src in latest_data.columns:
                    current_entry[dst] = float(latest_data[src].iloc[0])
            
            forecast.append(current_entry)
            
            # Generate future forecasts
            current_sequence = X_sequence.copy()
            
            # Initial values for forecast
            current_values = {
                'soil_temp': float(latest_data['soil_temp'].iloc[0]),
                'humidity': float(latest_data['humidity'].iloc[0]),
                'soil_moisture': float(latest_data['soil_moisture'].iloc[0]),
                'light_intensity': float(latest_data['light_intensity'].iloc[0]) if 'light_intensity' in latest_data.columns else 500,
                'temperature': float(latest_data['temperature'].iloc[0]) if 'temperature' in latest_data.columns else 22,
                'ph': float(latest_data['ph'].iloc[0]) if 'ph' in latest_data.columns else 6.5,
                'Nitrogen_Level': float(latest_data['Nitrogen_Level'].iloc[0]) if 'Nitrogen_Level' in latest_data.columns else 30,
                'Phosphorus_Level': float(latest_data['Phosphorus_Level'].iloc[0]) if 'Phosphorus_Level' in latest_data.columns else 30,
                'Potassium_Level': float(latest_data['Potassium_Level'].iloc[0]) if 'Potassium_Level' in latest_data.columns else 30
            }
            
            # Generate future timestamps
            start_hour = timestamp.hour
            
            for i in range(1, days * 6):
                # Update timestamp by 4 hours
                timestamp = timestamp + pd.Timedelta(hours=4)
                
                # Calculate hour of day for patterns
                hour_of_day = (start_hour + i * 4) % 24
                day_factor = min(1, max(0, math.sin((hour_of_day - 6) * math.pi / 12))) \
                             if hour_of_day >= 6 and hour_of_day <= 18 else 0
                    
                # Simulate realistic patterns
                self._update_forecast_values(current_values, hour_of_day, day_factor)
                
                # Make new prediction
                pred = self.model_service.predict_lstm(current_sequence)
                if not pred:
                    continue
                    
                # Add to forecast
                forecast_entry = {
                    'date': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    #'created_at': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'forecast_type': 'forecast',
                    'soil_temp': float(current_values['soil_temp']),
                    'humidity': float(current_values['humidity']),
                    'soil_moisture': float(current_values['soil_moisture']),
                    'predicted_health': pred['predicted_health'],
                    'confidence': pred['confidence']
                }
                
                # Add additional readings
                for attr, key in [('temperature', 'temperature'), 
                                ('light_intensity', 'light_intensity'),
                                ('ph', 'ph'),
                                ('Nitrogen_Level', 'nitrogen'),
                                ('Phosphorus_Level', 'phosphorus'),
                                ('Potassium_Level', 'potassium')]:
                    if attr in current_values:
                        forecast_entry[key] = float(current_values[attr])
                        
                forecast.append(forecast_entry)
            
            return forecast
            
        except Exception as e:
            print(f"Error in LSTM forecasting: {e}")
            return None
    
    def generate_traditional_forecast(self, iot_id, days=3):
        """Generate forecast using traditional model"""
        # Get historical data
        plant_history = self.data_service.get_plant_data(iot_id)
        if plant_history is None:
            return None
            
        plant_history['created_at'] = pd.to_datetime(plant_history['created_at'])
        plant_history = plant_history.sort_values('created_at')

        # Get the most recent data point as our starting point
        latest_data = plant_history.iloc[-1:].copy()
        timestamp = latest_data['created_at'].iloc[0]

        # Calculate trends from historical data
        trends = self._calculate_trends(plant_history)

        # Create forecast datapoints
        forecast = []
        
        # Save first data point as current
        current_entry = {
            'date': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            #'created_at': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'forecast_type': 'current',  # Mark as current data
            'soil_temp': float(latest_data['soil_temp'].iloc[0]),
            'humidity': float(latest_data['humidity'].iloc[0]),
            'soil_moisture': float(latest_data['soil_moisture'].iloc[0])
        }
        
        # Process for prediction
        processed_data = process_for_prediction(
            iot_id, latest_data, self.data_service.plant_data
        )
        
        # Make prediction
        prediction = self.model_service.predict_traditional(processed_data)
        if prediction:
            current_entry['predicted_health'] = prediction['predicted_health']
            current_entry['confidence'] = prediction['confidence']
        
        # Add optional fields
        for src, dst in [('temperature', 'temperature'), 
                        ('light_intensity', 'light_intensity'),
                        ('ph', 'ph'),
                        ('Nitrogen_Level', 'nitrogen'),
                        ('Phosphorus_Level', 'phosphorus'),
                        ('Potassium_Level', 'potassium')]:
            if src in latest_data.columns:
                current_entry[dst] = float(latest_data[src].iloc[0])
        
        forecast.append(current_entry)

        # Use initial values as starting point for forecast
        current_values = {
            'soil_temp': float(latest_data['soil_temp'].iloc[0]),
            'humidity': float(latest_data['humidity'].iloc[0]),
            'soil_moisture': float(latest_data['soil_moisture'].iloc[0]),
            'light_intensity': float(latest_data['light_intensity'].iloc[0]) if 'light_intensity' in latest_data.columns else 500,
            'temperature': float(latest_data['temperature'].iloc[0]) if 'temperature' in latest_data.columns else 22,
            'ph': float(latest_data['ph'].iloc[0]) if 'ph' in latest_data.columns else 6.5,
            'Nitrogen_Level': float(latest_data['Nitrogen_Level'].iloc[0]) if 'Nitrogen_Level' in latest_data.columns else 30,
            'Phosphorus_Level': float(latest_data['Phosphorus_Level'].iloc[0]) if 'Phosphorus_Level' in latest_data.columns else 30,
            'Potassium_Level': float(latest_data['Potassium_Level'].iloc[0]) if 'Potassium_Level' in latest_data.columns else 30
        }

        # Apply day/night cycle patterns
        start_hour = timestamp.hour
        
        for i in range(1, days * 6):  # Start from 1 because we already added current data
            # Update timestamp by 4 hours
            timestamp = timestamp + pd.Timedelta(hours=4)
            
            # Create new data with projected values
            new_data = latest_data.copy()
            new_data['created_at'] = timestamp
            
            # Update time features
            new_data['Hour'] = timestamp.hour
            new_data['Day'] = timestamp.day
            new_data['Month'] = timestamp.month
            
            # Calculate hour of day for day/night cycle patterns
            hour_of_day = (start_hour + i * 4) % 24
            day_factor = min(1, max(0, math.sin((hour_of_day - 6) * math.pi / 12))) \
                        if hour_of_day >= 6 and hour_of_day <= 18 else 0
            
            # Update forecast values based on time and trends
            self._update_forecast_values(
                current_values, hour_of_day, day_factor, trends
            )
            
            # Update the data point with new values
            for feature, value in current_values.items():
                if feature in new_data.columns:
                    new_data[feature] = value

            # Process for prediction
            processed_data = process_for_prediction(
                iot_id, new_data, self.data_service.plant_data
            )

            # Make prediction
            prediction = self.model_service.predict_traditional(processed_data)
            if not prediction:
                continue

            # Add to forecast
            forecast_entry = {
                'date': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                #'created_at': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'forecast_type': 'forecast',
                'soil_temp': float(current_values['soil_temp']),
                'humidity': float(current_values['humidity']),
                'soil_moisture': float(current_values['soil_moisture']),
                'predicted_health': prediction['predicted_health'],
                'confidence': prediction['confidence']
            }
            
            # Add additional readings
            for src, dst in [('temperature', 'temperature'), 
                            ('light_intensity', 'light_intensity'),
                            ('ph', 'ph'),
                            ('Nitrogen_Level', 'nitrogen'),
                            ('Phosphorus_Level', 'phosphorus'),
                            ('Potassium_Level', 'potassium')]:
                if src in current_values:
                    forecast_entry[dst] = float(current_values[src])
                    
            forecast.append(forecast_entry)

        return forecast
    
    def _calculate_trends(self, plant_history):
        """Calculate trends from plant history"""
        trends = {}
        if len(plant_history) >= 6:
            # Get data from at least 24 hours back (or as far back as we have)
            first_idx = max(0, len(plant_history) - 6)
            for feature in ['soil_temp', 'humidity', 'soil_moisture', 'light_intensity',
                          'temperature', 'ph', 'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']:
                if feature in plant_history.columns:
                    time_diff = (plant_history['created_at'].iloc[-1] - plant_history['created_at'].iloc[first_idx]).total_seconds()
                    if time_diff > 0:
                        value_diff = plant_history[feature].iloc[-1] - plant_history[feature].iloc[first_idx]
                        # Calculate hourly trend
                        trends[feature] = (value_diff / time_diff) * 3600
                    else:
                        trends[feature] = 0
        else:
            # Default trends if not enough history
            trends = {
                'soil_temp': 0.05,
                'humidity': -0.2,
                'soil_moisture': -0.4,
                'light_intensity': 0,  # Will be handled by day/night cycle
                'temperature': 0.1,
                'ph': 0,
                'Nitrogen_Level': -0.01,
                'Phosphorus_Level': -0.01,
                'Potassium_Level': -0.01
            }
            
        return trends
    
    def _update_forecast_values(self, current_values, hour_of_day, day_factor, trends=None):
        """Update forecast values based on time of day and trends"""
        if trends is None:
            trends = {}
            
        for feature in current_values.keys():
            base_trend = trends.get(feature, 0)
            
            # Apply special patterns for certain features
            if feature == 'light_intensity':
                # Light follows day/night cycle
                if hour_of_day >= 6 and hour_of_day <= 18:  # Day time
                    target_light = 200 + day_factor * 600  # Peak at noon
                else:  # Night time
                    target_light = 100 * min(1, (6 - hour_of_day % 6) / 6 if hour_of_day < 6 else (hour_of_day - 18) / 6)
                current_values[feature] = current_values[feature] * 0.7 + target_light * 0.3
            
            elif feature == 'temperature':
                # Temperature follows day/night cycle with delay
                if 8 <= hour_of_day <= 16:  # Day warming
                    temp_target = 22 + day_factor * 6  # Warmer during day
                else:  # Night cooling
                    temp_target = 22 - (1 - day_factor) * 4  # Cooler at night
                current_values[feature] = current_values[feature] * 0.9 + temp_target * 0.1
            
            elif feature == 'soil_temp':
                # Soil temp follows ambient with delay
                current_values[feature] += (current_values['temperature'] - current_values[feature]) * 0.05
            
            elif feature == 'soil_moisture':
                # Moisture decreases gradually until watering
                moisture_change = base_trend - 0.3  # Natural loss rate
                
                # Simulate automatic watering when it gets too low
                if current_values[feature] < 20:  # Water if too dry
                    moisture_change = 15  # Significant increase
                
                current_values[feature] += moisture_change
                current_values[feature] = max(10, min(90, current_values[feature]))
            
            elif feature == 'humidity':
                # Humidity inversely related to ambient temp
                humidity_target = 70 - (current_values['temperature'] - 20) * 2
                current_values[feature] = current_values[feature] * 0.8 + humidity_target * 0.2
                current_values[feature] = max(30, min(90, current_values[feature]))
            
            else:
                # Other features follow their trends with some randomness
                random_factor = (np.random.random() - 0.5) * 0.2  # ±10% variation
                current_values[feature] += base_trend * (1 + random_factor)
            
            # Apply realistic constraints
            if feature in ['humidity', 'soil_moisture']:
                current_values[feature] = max(0, min(100, current_values[feature]))
            elif feature == 'ph':
                current_values[feature] = max(4.5, min(8.5, current_values[feature]))
            elif feature in ['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level']:
                current_values[feature] = max(5, min(50, current_values[feature]))
    
    def generate_plant_forecast_data(self, iot_id, days=3):
        """Generate forecast data for real-time updates"""
        plant_data = self.data_service.get_plant_data(iot_id)
        if plant_data is None or len(plant_data) < 6:
            return None

        try:
            # Generate forecast
            forecast = self.generate_forecast(iot_id, days)
            if not forecast:
                return None

            return {
                "iot_id": DEFAULT_iot_id,
                "forecast_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "days_forecasted": days,
                "forecast": forecast
            }

        except Exception as e:
            print(f"Error generating forecast data: {e}")
            return None
    
    def get_plant_health_data(self, iot_id):
        """Generate plant health data for real-time updates"""
        plant_df = self.data_service.get_plant_data(iot_id)
        if plant_df is None or len(plant_df) == 0:
            return None
            
        try:
            # Get the most recent data
            plant_df = plant_df.sort_values('created_at')
            latest_data = plant_df.iloc[-1:].copy()

            # Add derived features for prediction
            latest_data = process_for_prediction(
                iot_id, latest_data, self.data_service.plant_data
            )

            # Make prediction
            prediction = self.model_service.predict_traditional(latest_data)
            if not prediction:
                print('No prediction available.')
                return None
                
            # Prepare reading data for response
            readings = {
                "soil_temperature": float(latest_data['soil_temp'].iloc[0]),
                "humidity": float(latest_data['humidity'].iloc[0]),
                "soil_moisture": float(latest_data['soil_moisture'].iloc[0])
            }
            
            # Add additional readings if available
            additional_fields = [
                ('temperature', 'temperature'),
                ('light_intensity', 'light_intensity'),
                ('ph', 'ph'),
                ('Nitrogen_Level', 'nitrogen'),
                ('Phosphorus_Level', 'phosphorus'),
                ('Potassium_Level', 'potassium'),
                ('Chlorophyll_Content', 'chlorophyll'),
                ('Electrochemical_Signal', 'ec_signal')
            ]
            
            for orig_field, resp_field in additional_fields:
                if orig_field in latest_data.columns:
                    readings[resp_field] = float(latest_data[orig_field].iloc[0])

            return {
                "iot_id": DEFAULT_iot_id,
                "timestamp": latest_data['created_at'].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_health": prediction['predicted_health'],
                "confidence": prediction['confidence'],
                "current_readings": readings
            }

        except Exception as e:
            print(f"Error generating health data: {e}")
            return None