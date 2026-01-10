"""
Database operations for the forecasting application
Handles MongoDB Atlas connections and CRUD operations
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages MongoDB Atlas database operations"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self.demo_mode = False
        self.connect()
        if self.connected:
            self.setup_collections()
    
    def connect(self):
        """Establish connection to MongoDB Atlas"""
        try:
            mongodb_uri = os.getenv('MONGODB_URI') or Config.MONGODB_URI
            
            # Replace placeholder values in URI
            if '<username>' in mongodb_uri or '<password>' in mongodb_uri:
                logger.warning("Please update MONGODB_URI with your Atlas credentials")
                # For development, fall back to local MongoDB
                mongodb_uri = "mongodb://localhost:27017/"
            
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                maxPoolSize=50,
                retryWrites=True
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[Config.DATABASE_NAME]
            self.connected = True
            logger.info("✅ MongoDB connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            logger.info("🔄 Running in DEMO MODE (no database required)")
            self.connected = False
            self.demo_mode = True
            self.client = None
            self.db = None
            
            self.db = self.client[Config.DATABASE_NAME]
            logger.info(f"Successfully connected to MongoDB: {Config.DATABASE_NAME}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    def setup_collections(self):
        """Set up database collections and indexes"""
        try:
            # Create indexes for better performance
            
            # Historical prices indexes
            self.db.historical_prices.create_index([
                ("instrument", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            # Predictions indexes
            self.db.predictions.create_index([
                ("instrument", ASCENDING),
                ("forecast_horizon", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Model performance indexes
            self.db.model_performance.create_index([
                ("model_name", ASCENDING),
                ("instrument", ASCENDING)
            ])
            
            # Supported instruments indexes
            self.db.supported_instruments.create_index([
                ("symbol", ASCENDING)
            ], unique=True)
            
            logger.info("Database collections and indexes set up successfully")
            
            # Initialize supported instruments
            self._initialize_supported_instruments()
            
        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
            raise
    
    def _initialize_supported_instruments(self):
        """Initialize the supported instruments collection"""
        supported_instruments = [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "type": "stock",
                "active": True,
                "default": True,
                "data_source": "yfinance",
                "last_updated": datetime.utcnow()
            },
            {
                "symbol": "GOOGL", 
                "name": "Alphabet Inc.",
                "type": "stock",
                "active": True,
                "default": False,
                "data_source": "yfinance",
                "last_updated": datetime.utcnow()
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corp.",
                "type": "stock", 
                "active": True,
                "default": False,
                "data_source": "yfinance",
                "last_updated": datetime.utcnow()
            },
            {
                "symbol": "TSLA",
                "name": "Tesla Inc.",
                "type": "stock",
                "active": True,
                "default": False,
                "data_source": "yfinance", 
                "last_updated": datetime.utcnow()
            },
            {
                "symbol": "BTC-USD",
                "name": "Bitcoin",
                "type": "crypto",
                "active": True,
                "default": False,
                "data_source": "yfinance",
                "last_updated": datetime.utcnow()
            },
            {
                "symbol": "ETH-USD",
                "name": "Ethereum",
                "type": "crypto",
                "active": True,
                "default": False,
                "data_source": "yfinance",
                "last_updated": datetime.utcnow()
            }
        ]
        
        # Insert or update instruments
        for instrument in supported_instruments:
            self.db.supported_instruments.update_one(
                {"symbol": instrument["symbol"]},
                {"$set": instrument},
                upsert=True
            )
        
        logger.info(f"Initialized {len(supported_instruments)} supported instruments")
    
    def get_supported_instruments(self) -> List[Dict]:
        """Get list of supported instruments"""
        try:
            instruments = list(
                self.db.supported_instruments.find(
                    {"active": True},
                    {"_id": 0}
                ).sort("default", DESCENDING)
            )
            return instruments
        except Exception as e:
            logger.error(f"Error fetching supported instruments: {e}")
            return []
    
    def save_historical_data(self, instrument: str, data: pd.DataFrame) -> bool:
        """Save historical price data to database"""
        try:
            if data.empty:
                logger.warning(f"No data to save for {instrument}")
                return False
            
            # Get instrument info
            instrument_info = self.db.supported_instruments.find_one({"symbol": instrument})
            instrument_type = instrument_info.get("type", "unknown") if instrument_info else "unknown"
            
            # Prepare documents for insertion
            documents = []
            for timestamp, row in data.iterrows():
                doc = {
                    "instrument": instrument,
                    "instrument_type": instrument_type,
                    "timestamp": timestamp,
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']) if 'Volume' in row else 0,
                    "created_at": datetime.utcnow()
                }
                documents.append(doc)
            
            # Insert documents (replace existing for same timestamp)
            for doc in documents:
                self.db.historical_prices.update_one(
                    {
                        "instrument": instrument,
                        "timestamp": doc["timestamp"]
                    },
                    {"$set": doc},
                    upsert=True
                )
            
            logger.info(f"Saved {len(documents)} historical data points for {instrument}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving historical data for {instrument}: {e}")
            return False
    
    def get_historical_data(self, instrument: str, limit: int = 100, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve historical data for an instrument as DataFrame"""
        try:
            query = {"instrument": instrument}
            
            if start_date:
                query["timestamp"] = {"$gte": start_date}
            
            cursor = self.db.historical_prices.find(
                query,
                {"_id": 0}
            ).sort("timestamp", DESCENDING).limit(limit)
            
            data = list(cursor)
            
            if not data:
                logger.warning(f"No historical data found for {instrument}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to match expected format
            column_mapping = {
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Set Date as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
            
            logger.info(f"Retrieved {len(df)} historical data points for {instrument}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {instrument}: {e}")
            return pd.DataFrame()
    
    def save_forecast(self, instrument: str, horizon: str, model_name: str, forecast_data: Dict) -> bool:
        """Save forecast predictions"""
        try:
            doc = {
                "instrument": instrument,
                "model_name": model_name,
                "forecast_horizon": horizon,
                "prediction_timestamp": forecast_data.get("prediction_timestamp"),
                "predicted_price": forecast_data.get("predicted_price"),
                "confidence_interval": forecast_data.get("confidence_interval", []),
                "metadata": forecast_data.get("metadata", {}),
                "created_at": datetime.utcnow()
            }
            
            result = self.db.predictions.insert_one(doc)
            logger.info(f"Saved forecast for {instrument} ({horizon}) using {model_name}")
            return bool(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
            return False
    
    def get_forecasts(self, instrument: str, horizon: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent forecasts for an instrument"""
        try:
            cursor = self.db.predictions.find(
                {
                    "instrument": instrument,
                    "forecast_horizon": horizon
                },
                {"_id": 0}
            ).sort("created_at", DESCENDING).limit(limit)
            
            forecasts = list(cursor)
            logger.info(f"Retrieved {len(forecasts)} forecasts for {instrument} ({horizon})")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {e}")
            return []
    
    def save_model_performance(self, model_name: str, instrument: str, metrics: Dict) -> bool:
        """Save model performance metrics"""
        try:
            doc = {
                "model_name": model_name,
                "instrument": instrument,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mape": metrics.get("mape"),
                "directional_accuracy": metrics.get("directional_accuracy"),
                "training_period": metrics.get("training_period"),
                "last_updated": datetime.utcnow()
            }
            
            # Update or insert performance metrics
            result = self.db.model_performance.update_one(
                {
                    "model_name": model_name,
                    "instrument": instrument
                },
                {"$set": doc},
                upsert=True
            )
            
            logger.info(f"Saved performance metrics for {model_name} on {instrument}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")
            return False
    
    def get_model_performance(self, instrument: Optional[str] = None) -> List[Dict]:
        """Retrieve model performance metrics"""
        try:
            query = {}
            if instrument:
                query["instrument"] = instrument
            
            cursor = self.db.model_performance.find(query, {"_id": 0})
            performance = list(cursor)
            
            logger.info(f"Retrieved performance metrics for {len(performance)} model-instrument combinations")
            return performance
            
        except Exception as e:
            logger.error(f"Error retrieving model performance: {e}")
            return []
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics about stored data"""
        try:
            if not self.connected or self.demo_mode:
                # Return demo data when database is not connected
                return {
                    "historical_data": [
                        {"_id": "AAPL", "count": 500, "latest": datetime.now(), "earliest": datetime.now() - timedelta(days=365)},
                        {"_id": "GOOGL", "count": 400, "latest": datetime.now(), "earliest": datetime.now() - timedelta(days=365)},
                        {"_id": "MSFT", "count": 450, "latest": datetime.now(), "earliest": datetime.now() - timedelta(days=365)}
                    ],
                    "predictions": [
                        {"_id": "AAPL", "count": 10, "latest": datetime.now()},
                        {"_id": "GOOGL", "count": 8, "latest": datetime.now()},
                        {"_id": "MSFT", "count": 12, "latest": datetime.now()}
                    ],
                    "supported_instruments": 7
                }
            
            summary = {}
            
            # Count historical data points per instrument
            pipeline = [
                {"$group": {
                    "_id": "$instrument",
                    "count": {"$sum": 1},
                    "latest": {"$max": "$timestamp"},
                    "earliest": {"$min": "$timestamp"}
                }}
            ]
            
            historical_summary = list(self.db.historical_prices.aggregate(pipeline))
            summary["historical_data"] = historical_summary
            
            # Count predictions per instrument
            pipeline = [
                {"$group": {
                    "_id": "$instrument", 
                    "count": {"$sum": 1},
                    "latest": {"$max": "$created_at"}
                }}
            ]
            
            predictions_summary = list(self.db.predictions.aggregate(pipeline))
            summary["predictions"] = predictions_summary
            
            # Count supported instruments
            summary["supported_instruments"] = self.db.supported_instruments.count_documents({"active": True})
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {
                "historical_data": [],
                "predictions": [],
                "supported_instruments": 7
            }
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        """Save model prediction to database"""
        try:
            doc = {
                "symbol": prediction_data.get("symbol"),
                "model_name": prediction_data.get("model_name"),
                "model_type": prediction_data.get("model_type"),
                "prediction_date": prediction_data.get("prediction_date"),
                "forecast_horizon": prediction_data.get("forecast_horizon"),
                "predictions": prediction_data.get("predictions"),
                "forecast_dates": prediction_data.get("forecast_dates"),
                "metadata": prediction_data.get("metadata", {}),
                "created_at": datetime.utcnow()
            }
            
            result = self.db.predictions.insert_one(doc)
            logger.info(f"Saved prediction for {prediction_data.get('symbol')} using {prediction_data.get('model_name')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def save_model_performance(self, performance_data: Dict) -> bool:
        """Save model performance metrics to database"""
        try:
            doc = {
                "symbol": performance_data.get("symbol"),
                "model_name": performance_data.get("model_name"),
                "model_type": performance_data.get("model_type"),
                "evaluation_type": performance_data.get("evaluation_type"),
                "evaluation_date": performance_data.get("evaluation_date"),
                "metrics": performance_data.get("metrics", {}),
                "created_at": datetime.utcnow()
            }
            
            result = self.db.model_performance.insert_one(doc)
            logger.info(f"Saved performance metrics for {performance_data.get('symbol')} using {performance_data.get('model_name')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")
            return False
    
    def get_predictions(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent predictions for a symbol"""
        try:
            cursor = self.db.predictions.find(
                {"symbol": symbol},
                {"_id": 0}
            ).sort("created_at", DESCENDING).limit(limit)
            
            predictions = list(cursor)
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions for {symbol}: {e}")
            return []
    
    def get_model_performance(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get model performance metrics"""
        try:
            query = {}
            if symbol:
                query["symbol"] = symbol
            
            cursor = self.db.model_performance.find(
                query,
                {"_id": 0}
            ).sort("created_at", DESCENDING).limit(limit)
            
            performance = list(cursor)
            return performance
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return []
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")

# Singleton instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get database manager singleton instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager