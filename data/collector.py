"""
Data collection module for financial instruments
Supports stocks, crypto, and forex data collection
"""

import os
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from data.database import get_database_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from various financial data sources"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.supported_instruments = self._get_supported_instruments()
    
    def _get_supported_instruments(self) -> Dict[str, Dict]:
        """Get supported instruments from database"""
        try:
            instruments = self.db_manager.get_supported_instruments()
            return {inst['symbol']: inst for inst in instruments}
        except Exception as e:
            logger.error(f"Error fetching supported instruments: {e}")
            return {}
    
    def collect_data(self, instrument: str, period: str = "1y", interval: str = "1d") -> Tuple[bool, str]:
        """
        Collect historical data for a specific instrument
        
        Args:
            instrument: Symbol (e.g., 'AAPL', 'BTC-USD')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            Tuple[bool, str]: (Success status, Message)
        """
        try:
            if instrument not in self.supported_instruments:
                return False, f"Instrument {instrument} not supported"
            
            instrument_info = self.supported_instruments[instrument]
            instrument_type = instrument_info['type']
            
            logger.info(f"Collecting data for {instrument} ({instrument_type}) - Period: {period}, Interval: {interval}")
            
            # Use yfinance for all data sources (supports stocks, crypto, forex)
            ticker = yf.Ticker(instrument)
            
            # Get historical data
            hist_data = ticker.history(period=period, interval=interval)
            
            if hist_data.empty:
                return False, f"No data available for {instrument}"
            
            # Clean and validate data
            hist_data = self._clean_data(hist_data, instrument)
            
            if hist_data.empty:
                return False, f"No valid data after cleaning for {instrument}"
            
            # Save to database
            success = self.db_manager.save_historical_data(instrument, hist_data)
            
            if success:
                # Update last_updated timestamp for instrument
                self.db_manager.db.supported_instruments.update_one(
                    {"symbol": instrument},
                    {"$set": {"last_updated": datetime.utcnow()}}
                )
                
                message = f"Successfully collected {len(hist_data)} data points for {instrument}"
                logger.info(message)
                return True, message
            else:
                return False, f"Failed to save data for {instrument}"
                
        except Exception as e:
            error_msg = f"Error collecting data for {instrument}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _clean_data(self, data: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """Clean and validate historical data"""
        try:
            # Remove rows with missing critical data
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            # Ensure positive prices
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                data = data[data[col] > 0]
            
            # Ensure High >= Low
            data = data[data['High'] >= data['Low']]
            
            # Ensure High >= Open, Close and Low <= Open, Close
            data = data[
                (data['High'] >= data['Open']) & 
                (data['High'] >= data['Close']) &
                (data['Low'] <= data['Open']) & 
                (data['Low'] <= data['Close'])
            ]
            
            # Handle volume (some instruments might not have volume data)
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            else:
                data['Volume'] = data['Volume'].fillna(0)
            
            # Reset index to ensure timestamp is a column
            if data.index.name != 'Date':
                data.reset_index(inplace=True)
                data.set_index('Date', inplace=True)
            
            logger.info(f"Cleaned data for {instrument}: {len(data)} valid records")
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data for {instrument}: {e}")
            return pd.DataFrame()
    
    def collect_all_instruments(self, period: str = "6mo") -> Dict[str, Tuple[bool, str]]:
        """Collect data for all supported instruments"""
        results = {}
        
        for instrument in self.supported_instruments:
            try:
                success, message = self.collect_data(instrument, period)
                results[instrument] = (success, message)
                
                # Add small delay between requests to be respectful to APIs
                import time
                time.sleep(0.5)
                
            except Exception as e:
                results[instrument] = (False, f"Unexpected error: {str(e)}")
        
        # Log summary
        successful = sum(1 for success, _ in results.values() if success)
        total = len(results)
        logger.info(f"Data collection complete: {successful}/{total} instruments successful")
        
        return results
    
    def get_latest_data(self, instrument: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get latest data for an instrument from database"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            data = self.db_manager.get_historical_data(
                instrument, 
                limit=days * 24,  # Rough estimate for hourly data
                start_date=start_date
            )
            
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Rename columns to match yfinance format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting latest data for {instrument}: {e}")
            return None
    
    def update_instrument_data(self, instrument: str) -> Tuple[bool, str]:
        """Update data for a specific instrument (get only recent data)"""
        try:
            # Get last update time from database
            instrument_info = self.db_manager.db.supported_instruments.find_one(
                {"symbol": instrument}
            )
            
            if not instrument_info:
                return False, f"Instrument {instrument} not found"
            
            last_updated = instrument_info.get('last_updated')
            
            # Determine period based on last update
            if last_updated:
                days_since_update = (datetime.utcnow() - last_updated).days
                if days_since_update < 1:
                    period = "1d"
                elif days_since_update < 7:
                    period = "5d" 
                else:
                    period = "1mo"
            else:
                period = "6mo"  # First time collection
            
            return self.collect_data(instrument, period)
            
        except Exception as e:
            error_msg = f"Error updating data for {instrument}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_data_quality(self, instrument: str) -> Dict[str, any]:
        """Validate data quality for an instrument"""
        try:
            data = self.get_latest_data(instrument, days=90)
            
            if data is None or data.empty:
                return {
                    "valid": False,
                    "message": "No data available",
                    "data_points": 0
                }
            
            validation = {
                "valid": True,
                "data_points": len(data),
                "date_range": {
                    "start": data.index.min().isoformat(),
                    "end": data.index.max().isoformat()
                },
                "completeness": {
                    "missing_days": 0,  # Calculate missing trading days
                    "data_coverage": 100.0  # Percentage coverage
                },
                "price_consistency": {
                    "negative_prices": (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any(),
                    "invalid_ohlc": ((data['High'] < data['Low']) | 
                                   (data['High'] < data['Open']) |
                                   (data['High'] < data['Close']) |
                                   (data['Low'] > data['Open']) |
                                   (data['Low'] > data['Close'])).any()
                }
            }
            
            # Check for issues
            issues = []
            if validation["price_consistency"]["negative_prices"]:
                issues.append("Negative or zero prices found")
            if validation["price_consistency"]["invalid_ohlc"]:
                issues.append("Invalid OHLC relationships found")
            
            validation["issues"] = issues
            validation["valid"] = len(issues) == 0
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating data quality for {instrument}: {e}")
            return {
                "valid": False,
                "message": f"Validation error: {str(e)}",
                "data_points": 0
            }

# Singleton instance
_data_collector = None

def get_data_collector() -> DataCollector:
    """Get data collector singleton instance"""
    global _data_collector
    if _data_collector is None:
        _data_collector = DataCollector()
    return _data_collector