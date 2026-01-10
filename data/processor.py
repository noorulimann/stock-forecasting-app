"""
Data preprocessing module for financial time series
Handles feature engineering, normalization, and data preparation for ML models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles preprocessing of financial time series data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        try:
            df = data.copy()
            
            # Simple Moving Averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Volume indicators (if volume data available)
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                
                # On-Balance Volume
                df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
            else:
                # Set volume indicators to zero if no volume data
                df['Volume_SMA'] = 0
                df['Volume_Ratio'] = 1
                df['OBV'] = 0
            
            # Price-based indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Open_Close_Ratio'] = df['Open'] / df['Close']
            
            # Volatility indicators
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            df['ATR'] = self._calculate_atr(df)
            
            # Trend indicators
            df['Price_Position'] = (df['Close'] - df['Close'].rolling(window=20).min()) / \
                                 (df['Close'].rolling(window=20).max() - df['Close'].rolling(window=20).min())
            
            logger.info(f"Added technical indicators. Dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=data.index, data=0)
    
    def prepare_features(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """Prepare feature set for machine learning"""
        try:
            df = data.copy()
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Lag features (previous values)
            lag_periods = [1, 2, 3, 5, 10]
            for lag in lag_periods:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
                df[f'Price_Change_Lag_{lag}'] = df['Price_Change'].shift(lag)
            
            # Rolling statistics
            windows = [5, 10, 20]
            for window in windows:
                df[f'Close_Mean_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'Close_Std_{window}'] = df['Close'].rolling(window=window).std()
                df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
            
            # Time-based features
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
            
            # Market session indicators (assuming US market hours)
            df['Is_Market_Hours'] = ((df.index.hour >= 9) & (df.index.hour < 16)).astype(int)
            df['Is_Weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Drop rows with NaN values (from rolling calculations)
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with NaN values")
            
            # Store feature columns (excluding target and non-numeric columns)
            # Filter out string columns like 'instrument', 'created_at', 'instrument_type'
            non_numeric_cols = ['instrument', 'instrument_type', 'created_at', 'timestamp']
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            self.feature_columns = [col for col in numeric_columns if col != target_column]
            
            logger.info(f"Prepared {len(self.feature_columns)} features for {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return data
    
    def scale_data(self, data: pd.DataFrame, scaler_type: str = 'minmax', 
                   fit_scaler: bool = True) -> Tuple[pd.DataFrame, dict]:
        """Scale the data using specified scaler"""
        try:
            df = data.copy()
            scalers = {}
            
            # Separate numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_type == 'standard':
                    scaler = StandardScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                
                if fit_scaler:
                    df[column] = scaler.fit_transform(df[[column]])
                    scalers[column] = scaler
                else:
                    # Use existing scaler if available
                    if column in self.scalers:
                        df[column] = self.scalers[column].transform(df[[column]])
                    else:
                        logger.warning(f"No existing scaler for column {column}, fitting new one")
                        df[column] = scaler.fit_transform(df[[column]])
                        scalers[column] = scaler
            
            if fit_scaler:
                self.scalers.update(scalers)
            
            logger.info(f"Scaled data using {scaler_type} scaler")
            return df, scalers
            
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            return data, {}
    
    def create_sequences(self, data: pd.DataFrame, target_column: str, 
                        sequence_length: int = 60, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling"""
        try:
            # Prepare feature matrix
            features = data[self.feature_columns].values
            targets = data[target_column].values
            
            X, y = [], []
            
            for i in range(sequence_length, len(data) - forecast_horizon + 1):
                # Input sequence
                X.append(features[i-sequence_length:i])
                
                # Target (future value)
                if forecast_horizon == 1:
                    y.append(targets[i])
                else:
                    y.append(targets[i:i+forecast_horizon])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, validation_size: float = 0.1) -> Tuple:
        """Split data into train, validation, and test sets"""
        try:
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Second split: separate validation from remaining data
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, shuffle=False
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return X, np.array([]), np.array([]), y, np.array([]), np.array([])
    
    def inverse_transform_predictions(self, predictions: np.ndarray, 
                                    target_column: str) -> np.ndarray:
        """Inverse transform scaled predictions back to original scale"""
        try:
            if target_column in self.scalers:
                scaler = self.scalers[target_column]
                
                # Reshape predictions if needed
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                
                predictions_original = scaler.inverse_transform(predictions)
                
                # Flatten if single feature
                if predictions_original.shape[1] == 1:
                    predictions_original = predictions_original.flatten()
                
                logger.info("Inverse transformed predictions to original scale")
                return predictions_original
            else:
                logger.warning(f"No scaler found for {target_column}")
                return predictions
                
        except Exception as e:
            logger.error(f"Error inverse transforming predictions: {e}")
            return predictions
    
    def get_feature_importance_data(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic feature importance based on correlation with target"""
        try:
            df = data.copy()
            
            # Calculate correlation with Close price
            correlations = df.corr()['Close'].abs().sort_values(ascending=False)
            
            # Remove target itself
            if 'Close' in correlations:
                correlations = correlations.drop('Close')
            
            # Convert to dictionary
            importance = correlations.to_dict()
            
            logger.info(f"Calculated feature importance for {len(importance)} features")
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def prepare_data_for_model(self, data: pd.DataFrame, target_column: str = 'Close',
                             sequence_length: int = 60, forecast_horizon: int = 1,
                             test_size: float = 0.2, validation_size: float = 0.1) -> Dict:
        """Complete data preparation pipeline for model training"""
        try:
            logger.info("Starting complete data preparation pipeline")
            
            # Step 1: Prepare features
            df_features = self.prepare_features(data, target_column)
            
            # Step 2: Scale data
            df_scaled, scalers = self.scale_data(df_features, scaler_type='minmax', fit_scaler=True)
            
            # Step 3: Create sequences
            X, y = self.create_sequences(df_scaled, target_column, sequence_length, forecast_horizon)
            
            # Step 4: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
                X, y, test_size, validation_size
            )
            
            # Step 5: Calculate feature importance
            feature_importance = self.get_feature_importance_data(df_features)
            
            result = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scalers': scalers,
                'feature_columns': self.feature_columns,
                'feature_importance': feature_importance,
                'original_data': df_features,
                'scaled_data': df_scaled,
                'sequence_length': sequence_length,
                'forecast_horizon': forecast_horizon
            }
            
            logger.info("Data preparation pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in data preparation pipeline: {e}")
            return {}

# Singleton instance
_data_processor = None

def get_data_processor() -> DataProcessor:
    """Get data processor singleton instance"""
    global _data_processor
    if _data_processor is None:
        _data_processor = DataProcessor()
    return _data_processor