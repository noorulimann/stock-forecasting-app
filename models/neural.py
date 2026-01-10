#!/usr/bin/env python3
"""
Neural Network Models for Financial Forecasting
Implements LSTM, GRU, and Transformer models using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
import time

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("⚠️ MLflow not installed. Experiment tracking disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output

class GRUModel(nn.Module):
    """GRU model for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        
        # Take the last output
        last_output = gru_out[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        output = self.linear(output)
        
        return output

class TransformerModel(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=False  # Transformer expects (seq_len, batch, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (seq_len, batch_size, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        transformer_out = self.transformer(x)  # (seq_len, batch_size, d_model)
        
        # Take the last timestep
        last_output = transformer_out[-1]  # (batch_size, d_model)
        
        # Apply dropout and output layer
        output = self.dropout(last_output)
        output = self.output_layer(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class NeuralModels:
    def __init__(self):
        """Initialize neural models manager"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.models = {
            'lstm': {
                'class': LSTMModel,
                'description': 'Long Short-Term Memory network for sequence prediction',
                'requires_sequences': True
            },
            'gru': {
                'class': GRUModel, 
                'description': 'Gated Recurrent Unit network for sequence prediction',
                'requires_sequences': True
            },
            'transformer': {
                'class': TransformerModel,
                'description': 'Transformer model for sequence prediction',
                'requires_sequences': True
            }
        }
        
        self.trained_models = {}
        self.scalers = {}
    
    def get_available_models(self):
        """Get list of available neural models"""
        return list(self.models.keys())
    
    def get_model_info(self):
        """Get information about available models"""
        info = {}
        for name, config in self.models.items():
            info[name] = {
                'description': config['description'],
                'requires_sequences': config['requires_sequences'],
                'status': 'available'
            }
        return info
    
    def _prepare_data_for_training(self, data, target_column='Close', sequence_length=60, test_size=0.2):
        """Prepare data for neural network training"""
        try:
            if data is None or len(data) == 0:
                logger.error("No data provided for training preparation")
                return None
            
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Ensure we have the target column
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in data")
                return None
            
            # Sort by date if date column exists
            if 'date' in data.columns:
                data = data.sort_values('date').reset_index(drop=True)
            
            # Select features for training
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_features = [col for col in feature_columns if col in data.columns]
            
            if len(available_features) == 0:
                logger.error("No suitable features found for training")
                return None
            
            # Prepare feature matrix
            features = data[available_features].values
            
            # Scale the features
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y.append(features_scaled[i, available_features.index(target_column)])
            
            if len(X) == 0:
                logger.error(f"Not enough data to create sequences (need at least {sequence_length + 1} samples)")
                return None
            
            X, y = np.array(X), np.array(y)
            
            # Split into train/test
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Prepared data: {len(X_train)} train samples, {len(X_test)} test samples")
            
            return X_train, X_test, y_train, y_test, scaler
            
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            return None
    
    def train_model(self, data, model_type='lstm', symbol='AAPL', epochs=50, batch_size=32, **kwargs):
        """Train a specific neural model with MLflow tracking"""
        training_start_time = time.time()
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(f"stock-forecasting-{model_type}")
                mlflow.start_run(run_name=f"{model_type.upper()}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Log basic parameters
                mlflow.log_param("model_type", model_type.upper())
                mlflow.log_param("symbol", symbol)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
            except Exception as e:
                logger.warning(f"⚠️ MLflow logging initialization failed: {e}")
        
        try:
            if model_type not in self.models:
                return {
                    'success': False,
                    'error': f'Unknown model type: {model_type}'
                }
            
            # Prepare data
            sequence_length = kwargs.get('sequence_length', 60)
            prepared_data = self._prepare_data_for_training(
                data, 
                sequence_length=sequence_length,
                test_size=0.2
            )
            
            if prepared_data is None:
                return {
                    'success': False,
                    'error': 'Failed to prepare data for training'
                }
            
            X_train, X_test, y_train, y_test, scaler = prepared_data
            
            # Log data parameters to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("sequence_length", sequence_length)
                    mlflow.log_param("train_samples", len(X_train))
                    mlflow.log_param("test_samples", len(X_test))
                    mlflow.log_param("num_features", X_train.shape[2])
                except:
                    pass
            
            # Store scaler for later use
            scaler_key = f"{symbol}_{model_type}"
            self.scalers[scaler_key] = scaler
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            # Get model architecture parameters
            input_size = X_train.shape[2]  # Number of features
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            learning_rate = kwargs.get('learning_rate', 0.001)
            
            # Log hyperparameters to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("hidden_size", hidden_size)
                    mlflow.log_param("num_layers", num_layers)
                    mlflow.log_param("dropout", dropout)
                    mlflow.log_param("learning_rate", learning_rate)
                except:
                    pass
            
            # Initialize model
            if model_type == 'transformer':
                model = self.models[model_type]['class'](
                    input_size=input_size,
                    d_model=hidden_size,
                    nhead=kwargs.get('nhead', 8),
                    num_layers=num_layers,
                    dropout=dropout
                ).to(self.device)
            else:
                model = self.models[model_type]['class'](
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Create data loaders
            train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            model.train()
            train_losses = []
            
            logger.info(f"Starting {model_type.upper()} training for {symbol}...")
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    if model_type == 'transformer':
                        # Transformer expects (seq_len, batch_size, features)
                        batch_X = batch_X.permute(1, 0, 2)
                        output = model(batch_X)
                    else:
                        output = model(batch_X)
                    
                    loss = criterion(output.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                
                # Log training loss to MLflow every 10 epochs
                if MLFLOW_AVAILABLE and (epoch % 10 == 0 or epoch == epochs - 1):
                    try:
                        mlflow.log_metric("train_loss", avg_loss, step=epoch)
                    except:
                        pass
                
                if epoch % 10 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                if model_type == 'transformer':
                    X_test_transformed = X_test_tensor.permute(1, 0, 2)
                    test_predictions = model(X_test_transformed)
                else:
                    test_predictions = model(X_test_tensor)
                
                test_loss = criterion(test_predictions.squeeze(), y_test_tensor).item()
            
            # Calculate metrics
            test_pred_np = test_predictions.squeeze().cpu().numpy()
            test_true_np = y_test_tensor.cpu().numpy()
            
            mse = np.mean((test_pred_np - test_true_np) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(test_pred_np - test_true_np))
            
            # Calculate MAPE and R²
            mape = np.mean(np.abs((test_true_np - test_pred_np) / (test_true_np + 1e-10))) * 100
            r2 = 1 - (np.sum((test_true_np - test_pred_np) ** 2) / np.sum((test_true_np - np.mean(test_true_np)) ** 2))
            
            # Calculate directional accuracy
            if len(test_true_np) > 1:
                true_direction = np.diff(test_true_np) > 0
                pred_direction = np.diff(test_pred_np) > 0
                directional_accuracy = np.mean(true_direction == pred_direction) * 100
            else:
                directional_accuracy = 0.0
            
            # Calculate training time
            training_time = time.time() - training_start_time
            
            # Log all metrics to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("test_loss", test_loss)
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("mape", mape)
                    mlflow.log_metric("r2_score", r2)
                    mlflow.log_metric("directional_accuracy", directional_accuracy)
                    mlflow.log_metric("training_time_seconds", training_time)
                    mlflow.log_metric("final_train_loss", train_losses[-1])
                    
                    # Log the model
                    mlflow.pytorch.log_model(model, f"{model_type}_model")
                    
                    logger.info("✅ MLflow tracking: Metrics and model logged successfully")
                except Exception as e:
                    logger.warning(f"⚠️ MLflow logging failed: {e}")
            
            # Save model
            model_key = f"{symbol}_{model_type}"
            self.trained_models[model_key] = {
                'model': model,
                'scaler': scaler,
                'input_size': input_size,
                'sequence_length': sequence_length,
                'model_type': model_type
            }
            
            # Save to disk
            model_saved = self._save_model_to_disk(model, scaler, symbol, model_type)
            
            logger.info(f"✅ {model_type.upper()} training completed for {symbol}")
            logger.info(f"   Training time: {training_time:.2f}s")
            
            return {
                'success': True,
                'model_saved': model_saved,
                'metrics': {
                    'train_loss': train_losses[-1],
                    'test_loss': test_loss,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'r2_score': r2,
                    'directional_accuracy': directional_accuracy,
                    'training_time_seconds': training_time,
                    'epochs_trained': epochs
                },
                'training_history': {
                    'train_losses': train_losses
                }
            }
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            
            # Log error to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("error", str(e))
                except:
                    pass
            
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # End MLflow run
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.end_run()
                except:
                    pass
    
    def predict(self, data, model_type='lstm', symbol='AAPL', forecast_horizon=5, **kwargs):
        """Generate predictions using trained model"""
        try:
            model_key = f"{symbol}_{model_type}"
            
            # Check if model is loaded
            if model_key not in self.trained_models:
                # Try to load from disk
                if not self._load_model_from_disk(symbol, model_type):
                    return {
                        'success': False,
                        'error': f'No trained {model_type} model found for {symbol}'
                    }
            
            model_info = self.trained_models[model_key]
            model = model_info['model']
            scaler = model_info['scaler']
            sequence_length = model_info['sequence_length']
            
            # Prepare recent data for prediction
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Sort by date if available
            if 'date' in data.columns:
                data = data.sort_values('date').reset_index(drop=True)
            
            # Get recent data
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_features = [col for col in feature_columns if col in data.columns]
            
            if len(available_features) == 0:
                return {
                    'success': False,
                    'error': 'No suitable features found for prediction'
                }
            
            recent_data = data[available_features].tail(sequence_length).values
            
            if len(recent_data) < sequence_length:
                return {
                    'success': False,
                    'error': f'Not enough recent data for prediction (need {sequence_length}, got {len(recent_data)})'
                }
            
            # Scale data
            recent_scaled = scaler.transform(recent_data)
            
            # Generate predictions
            model.eval()
            predictions = []
            current_sequence = recent_scaled.copy()
            
            with torch.no_grad():
                for _ in range(forecast_horizon):
                    # Prepare input
                    input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                    
                    if model_type == 'transformer':
                        input_tensor = input_tensor.permute(1, 0, 2)
                        pred = model(input_tensor)
                    else:
                        pred = model(input_tensor)
                    
                    pred_value = pred.squeeze().cpu().numpy()
                    predictions.append(float(pred_value))
                    
                    # Update sequence for next prediction
                    # Create new row with predicted value
                    new_row = current_sequence[-1].copy()
                    close_idx = available_features.index('Close')
                    new_row[close_idx] = pred_value
                    
                    # Shift sequence
                    current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # Inverse transform predictions (for close price)
            close_idx = available_features.index('Close')
            dummy_array = np.zeros((len(predictions), len(available_features)))
            dummy_array[:, close_idx] = predictions
            
            inverse_predictions = scaler.inverse_transform(dummy_array)[:, close_idx]
            
            # Create forecast data with dates
            last_date = data['date'].iloc[-1] if 'date' in data.columns else pd.Timestamp.now()
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
            
            forecast = []
            for i, (date, price) in enumerate(zip(forecast_dates, inverse_predictions)):
                forecast.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': float(price),
                    'day_ahead': i + 1
                })
            
            logger.info(f"✅ Generated {forecast_horizon}-day forecast using {model_type} for {symbol}")
            
            return {
                'success': True,
                'forecast': forecast,
                'model_info': {
                    'model_type': model_type,
                    'symbol': symbol,
                    'forecast_horizon': forecast_horizon,
                    'sequence_length': sequence_length
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions with {model_type}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_model_to_disk(self, model, scaler, symbol, model_type):
        """Save trained model and scaler to disk"""
        try:
            os.makedirs('saved_models', exist_ok=True)
            
            model_filename = f"saved_models/{symbol}_{model_type}_model.pth"
            scaler_filename = f"saved_models/{symbol}_{model_type}_scaler.pkl"
            
            # Save PyTorch model
            torch.save(model.state_dict(), model_filename)
            
            # Save scaler
            with open(scaler_filename, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"✅ Saved {model_type} model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _load_model_from_disk(self, symbol, model_type):
        """Load trained model and scaler from disk"""
        try:
            model_filename = f"saved_models/{symbol}_{model_type}_model.pth"
            scaler_filename = f"saved_models/{symbol}_{model_type}_scaler.pkl"
            
            if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
                return False
            
            # Load scaler
            with open(scaler_filename, 'rb') as f:
                scaler = pickle.load(f)
            
            # We need to reconstruct the model architecture
            # For simplicity, use default parameters
            input_size = 5  # Default: open, high, low, close, volume
            
            if model_type == 'transformer':
                model = self.models[model_type]['class'](
                    input_size=input_size,
                    d_model=128,
                    nhead=8,
                    num_layers=2,
                    dropout=0.2
                ).to(self.device)
            else:
                model = self.models[model_type]['class'](
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2
                ).to(self.device)
            
            # Load model state
            model.load_state_dict(torch.load(model_filename, map_location=self.device))
            model.eval()
            
            # Store in memory
            model_key = f"{symbol}_{model_type}"
            self.trained_models[model_key] = {
                'model': model,
                'scaler': scaler,
                'input_size': input_size,
                'sequence_length': 60,  # Default
                'model_type': model_type
            }
            
            logger.info(f"✅ Loaded {model_type} model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate_all_models(self, data, symbol='AAPL', target_column='Close'):
        """Evaluate all neural models for a given symbol"""
        results = {}
        
        try:
            from utils.evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            
            # Get available models for this symbol
            available_models = []
            for model_type in self.model_types:
                model_path = self._get_model_path(symbol, model_type)
                if os.path.exists(model_path):
                    available_models.append(model_type)
            
            if not available_models:
                logger.warning(f"No trained models found for {symbol}")
                return results
            
            # Prepare test data
            X_test, y_test, scaler = self._prepare_data_for_training(data, symbol, split_ratio=0.8)
            
            for model_type in available_models:
                try:
                    # Load model and make predictions
                    if self.load_model(symbol, model_type):
                        model = self.models[symbol][model_type]['model']
                        model.eval()
                        
                        with torch.no_grad():
                            predictions = model(X_test)
                            
                        # Convert to numpy for evaluation
                        y_test_np = y_test.numpy()
                        predictions_np = predictions.numpy()
                        
                        # Calculate metrics
                        metrics = evaluator.calculate_metrics(y_test_np, predictions_np)
                        results[model_type] = metrics
                        
                        logger.info(f"Evaluated {model_type} model for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating {model_type} model: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluate_all_models: {e}")
            return results
    
    def predict_all_models(self, data, symbol='AAPL', forecast_horizon=5, steps=None):
        """Generate predictions using all available neural models"""
        results = {}
        
        try:
            # Use steps parameter if provided, otherwise use forecast_horizon
            if steps is not None:
                forecast_horizon = steps
            
            # Get available models for this symbol
            available_models = []
            for model_type in self.model_types:
                model_path = self._get_model_path(symbol, model_type)
                if os.path.exists(model_path):
                    available_models.append(model_type)
            
            if not available_models:
                logger.warning(f"No trained models found for {symbol}")
                return results
            
            for model_type in available_models:
                try:
                    # Generate predictions for this model
                    predictions = self.predict(
                        data=data,
                        model_type=model_type,
                        symbol=symbol,
                        forecast_horizon=forecast_horizon
                    )
                    
                    if predictions:
                        results[model_type] = predictions
                        logger.info(f"Generated predictions using {model_type} for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error generating predictions with {model_type}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in predict_all_models: {e}")
            return results

def get_neural_models():
    """Factory function to get NeuralModels instance"""
    return NeuralModels()

# Global instance
_neural_models_instance = None

def get_neural_models_instance():
    """Get singleton neural models instance"""
    global _neural_models_instance
    if _neural_models_instance is None:
        _neural_models_instance = NeuralModels()
    return _neural_models_instance