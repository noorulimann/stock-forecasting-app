import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # MongoDB Configuration
    MONGODB_URI = os.environ.get('MONGODB_URI') or 'mongodb://localhost:27017/'
    DATABASE_NAME = os.environ.get('DATABASE_NAME') or 'forecasting_db'
    
    # Application Settings
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Data Settings
    DEFAULT_INSTRUMENT = os.environ.get('DEFAULT_INSTRUMENT', 'AAPL')
    UPDATE_INTERVAL = int(os.environ.get('UPDATE_INTERVAL', 3600))  # seconds
    
    # Model Settings
    MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', 'saved_models/')
    
    # API Rate Limits
    MAX_REQUESTS_PER_MINUTE = int(os.environ.get('MAX_REQUESTS_PER_MINUTE', 60))