from flask import Blueprint, render_template, request, jsonify, current_app
from data.collector import DataCollector
from data.database import DatabaseManager
from utils.visualizer import ChartGenerator
from models.traditional import TraditionalModels
from models.neural import NeuralModels
import json

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@main.route('/api/instruments')
def get_instruments():
    """Get list of available financial instruments"""
    instruments = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'stock'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'type': 'stock'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'type': 'stock'},
        {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'type': 'crypto'},
        {'symbol': 'ETH-USD', 'name': 'Ethereum', 'type': 'crypto'},
    ]
    return jsonify(instruments)

@main.route('/api/data/<instrument>')
def get_historical_data(instrument):
    """Get historical data for an instrument"""
    try:
        db_manager = DatabaseManager()
        data = db_manager.get_historical_data(instrument, limit=100)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/forecast', methods=['POST'])
def generate_forecast():
    """Generate forecast for selected instrument and horizon"""
    try:
        data = request.get_json()
        instrument = data.get('instrument')
        horizon = data.get('horizon')  # '1hr', '3hrs', '24hrs', '72hrs'
        model_type = data.get('model', 'ensemble')  # 'traditional', 'neural', 'ensemble'
        
        # Get historical data
        db_manager = DatabaseManager()
        historical_data = db_manager.get_historical_data(instrument, limit=1000)
        
        # Generate forecast based on model type
        if model_type == 'traditional':
            models = TraditionalModels()
            forecast = models.predict(historical_data, horizon)
        elif model_type == 'neural':
            models = NeuralModels()
            forecast = models.predict(historical_data, horizon)
        else:  # ensemble
            traditional = TraditionalModels()
            neural = NeuralModels()
            # Combine predictions (implementation in models)
            forecast = traditional.ensemble_predict(historical_data, horizon, neural)
        
        # Save forecast to database
        db_manager.save_forecast(instrument, horizon, model_type, forecast)
        
        return jsonify({
            'success': True, 
            'forecast': forecast,
            'model_used': model_type
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/chart/<instrument>')
def get_chart_data(instrument):
    """Get chart data with forecasts"""
    try:
        horizon = request.args.get('horizon', '24hrs')
        
        db_manager = DatabaseManager()
        historical_data = db_manager.get_historical_data(instrument, limit=100)
        forecasts = db_manager.get_forecasts(instrument, horizon)
        
        chart_generator = ChartGenerator()
        chart_json = chart_generator.create_candlestick_chart(
            historical_data, forecasts
        )
        
        return jsonify({
            'success': True,
            'chart': chart_json
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/api/performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        db_manager = DatabaseManager()
        performance = db_manager.get_model_performance()
        return jsonify({'success': True, 'performance': performance})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main.route('/forecast')
def forecast_page():
    """Forecast results page"""
    instrument = request.args.get('instrument', 'AAPL')
    horizon = request.args.get('horizon', '24hrs')
    return render_template('forecast.html', instrument=instrument, horizon=horizon)