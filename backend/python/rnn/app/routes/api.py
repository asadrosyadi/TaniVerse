from flask import request, jsonify
from datetime import datetime, timedelta


def register_routes(app, socketio, model_service, data_service, forecast_service):
    """Register all API routes"""
    
    @app.route('/', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            "status": "healthy", 
            "model_loaded": model_service.model is not None,
            "lstm_model_loaded": model_service.lstm_model is not None
        })
    
    @app.route('/forecast/<int:iot_id>', methods=['GET'])
    def forecast_plant_health(iot_id):
        """Predict future plant health"""
        plant_df = data_service.get_plant_data(iot_id)
        if plant_df is None or len(plant_df) < 6:
            return jsonify({
                "error": f"Not enough data for Plant ID {iot_id}. Need at least 6 readings."
            }), 400
    
        try:
            # Get days parameter, default to 3
            days = int(request.args.get('days', 3))
            if days > 14:  # Limit forecast length
                days = 14
    
            forecast_data = forecast_service.generate_plant_forecast_data(iot_id, days)
            if forecast_data:
                return jsonify(forecast_data)
            else:
                return jsonify({"error": "Failed to generate forecast"}), 500
    
        except Exception as e:
            return jsonify({"error": str(e)}), 500