"""
LightGBM Model for Financial Forecasting

- Automated hyperparameter optimization
- Feature importance analysis
- Cross-validation
- Production-ready prediction pipeline
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import pickle
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("⚠️ MLflow not installed. Experiment tracking disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMForecaster:
    """
    LightGBM-based forecasting model with comprehensive features:
    - Hyperparameter optimization
    - Feature engineering integration
    - Cross-validation
    - Model persistence
    - Feature importance analysis
    """
    
    def __init__(self, model_save_path: str = 'saved_models/'):
        """
        Initialize LightGBM forecaster
        
        Args:
            model_save_path: Directory to save trained models
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.best_params = {}
        self.feature_importance = {}
        self.training_history = {}
        self.model_save_path = model_save_path
        self.is_fitted = False
        
        # Default hyperparameters (optimized for financial time series)
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'seed': 42
        }
        
        # Create save directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
        logger.info("✅ LightGBM Forecaster initialized")
    
    def prepare_features(self, data: pd.DataFrame, target_column: str = 'Close', 
                        lookback: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction
        
        Args:
            data: Input dataframe with technical indicators
            target_column: Name of the target column
            lookback: Number of past days to use as features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            df = data.copy()
            
            # Ensure data is sorted by date
            if 'Date' in df.columns:
                df = df.sort_values('Date')
            
            # Create lagged features
            for lag in range(1, lookback + 1):
                df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            
            # Create rolling statistics
            for window in [5, 10, 20]:
                df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
                df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
                df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window).min()
                df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window).max()
            
            # Create percentage change features
            df[f'{target_column}_pct_change_1'] = df[target_column].pct_change(1)
            df[f'{target_column}_pct_change_5'] = df[target_column].pct_change(5)
            df[f'{target_column}_pct_change_10'] = df[target_column].pct_change(10)
            
            # Drop rows with NaN values (from lagging and rolling)
            df = df.dropna()
            
            if len(df) == 0:
                raise ValueError("No data remaining after feature engineering")
            
            # Select feature columns (exclude target and non-numeric columns)
            exclude_cols = ['Date', 'Datetime', target_column, 'instrument', 'instrument_type', 
                          'created_at', 'timestamp', 'updated_at', '_id']
            
            # Get only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            
            # Prepare features and target
            X = df[feature_columns]
            y = df[target_column]
            
            # Store feature names
            self.feature_names = feature_columns
            
            logger.info(f"✅ Features prepared: {len(feature_columns)} features, {len(df)} samples")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            raise
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of parameter combinations to try
            
        Returns:
            Dictionary of best parameters
        """
        try:
            logger.info(f"🔍 Starting hyperparameter optimization ({n_trials} trials)...")
            
            # Define parameter grid
            param_grid = {
                'num_leaves': [15, 31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.6, 0.8, 1.0],
                'bagging_fraction': [0.6, 0.8, 1.0],
                'max_depth': [5, 10, 15, -1],
                'min_child_samples': [10, 20, 30],
                'lambda_l1': [0, 0.1, 1.0],
                'lambda_l2': [0, 0.1, 1.0],
            }
            
            best_score = float('inf')
            best_params = self.default_params.copy()
            
            # Simple random search (can be replaced with optuna for more sophisticated optimization)
            import random
            
            for trial in range(min(n_trials, 20)):  # Limit trials for speed
                # Sample random parameters
                trial_params = self.default_params.copy()
                trial_params.update({
                    'num_leaves': random.choice(param_grid['num_leaves']),
                    'learning_rate': random.choice(param_grid['learning_rate']),
                    'feature_fraction': random.choice(param_grid['feature_fraction']),
                    'bagging_fraction': random.choice(param_grid['bagging_fraction']),
                    'max_depth': random.choice(param_grid['max_depth']),
                    'min_child_samples': random.choice(param_grid['min_child_samples']),
                    'lambda_l1': random.choice(param_grid['lambda_l1']),
                    'lambda_l2': random.choice(param_grid['lambda_l2']),
                })
                
                # Train model with trial parameters
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    trial_params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
                )
                
                # Evaluate on validation set
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
                
                if score < best_score:
                    best_score = score
                    best_params = trial_params.copy()
                    best_params['num_boost_round'] = model.best_iteration
                    logger.info(f"   Trial {trial+1}: New best RMSE = {score:.4f}")
            
            self.best_params = best_params
            logger.info(f"✅ Optimization complete. Best RMSE: {best_score:.4f}")
            logger.info(f"   Best params: num_leaves={best_params['num_leaves']}, lr={best_params['learning_rate']}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"❌ Error optimizing hyperparameters: {e}")
            return self.default_params
    
    def fit(self, data: pd.DataFrame, target_column: str = 'Close', 
            optimize: bool = True, test_size: float = 0.2,
            lookback: int = 5, num_boost_round: int = 500,
            symbol: str = None) -> Dict[str, Any]:
        """
        Train LightGBM model on the data with MLflow tracking
        
        Args:
            data: Training dataframe with features
            target_column: Name of target column to predict
            optimize: Whether to optimize hyperparameters
            test_size: Proportion of data for validation
            lookback: Number of past days for lagged features
            num_boost_round: Number of boosting rounds
            symbol: Stock symbol for MLflow tracking
            
        Returns:
            Dictionary with training results
        """
        training_start_time = time.time()
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            try:
                # Set experiment name
                mlflow.set_experiment("stock-forecasting-lightgbm")
                mlflow.start_run(run_name=f"LightGBM_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Log basic parameters
                mlflow.log_param("model_type", "LightGBM")
                mlflow.log_param("symbol", symbol or "unknown")
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("optimize_hyperparameters", optimize)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("lookback", lookback)
                mlflow.log_param("num_boost_round", num_boost_round)
                mlflow.log_param("train_samples_total", len(data))
            except Exception as e:
                logger.warning(f"⚠️ MLflow logging initialization failed: {e}")
        
        try:
            logger.info(f"🚀 Training LightGBM model on {len(data)} samples...")
            
            # Prepare features
            X, y = self.prepare_features(data, target_column, lookback)
            
            # Split into train and validation sets (time-aware split)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"   Train size: {len(X_train)}, Validation size: {len(X_val)}")
            
            # Log dataset info to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("train_samples", len(X_train))
                    mlflow.log_param("validation_samples", len(X_val))
                    mlflow.log_param("num_features", len(self.feature_names))
                except:
                    pass
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Convert back to DataFrame to preserve column names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_names)
            
            # Optimize hyperparameters if requested
            if optimize:
                params = self.optimize_hyperparameters(X_train_scaled, y_train, 
                                                      X_val_scaled, y_val)
            else:
                params = self.default_params.copy()
            
            # Log hyperparameters to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    for key, value in params.items():
                        if key not in ['verbosity']:  # Skip non-essential params
                            mlflow.log_param(f"lgb_{key}", value)
                except:
                    pass
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            # Train final model
            logger.info("🔄 Training final model...")
            
            evals_result = {}
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=50),
                    lgb.record_evaluation(evals_result)
                ]
            )
            
            # Calculate feature importance
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain')
            ))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Make predictions on validation set
            y_pred_val = self.model.predict(X_val_scaled, num_iteration=self.model.best_iteration)
            
            # Calculate metrics
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                'mae': float(mean_absolute_error(y_val, y_pred_val)),
                'r2': float(r2_score(y_val, y_pred_val)),
                'mape': float(np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100),
            }
            
            # Calculate directional accuracy
            if len(y_val) > 1:
                true_direction = np.diff(y_val) > 0
                pred_direction = np.diff(y_pred_val) > 0
                metrics['directional_accuracy'] = float(np.mean(true_direction == pred_direction) * 100)
            
            # Calculate training time
            training_time = time.time() - training_start_time
            metrics['training_time_seconds'] = training_time
            
            # Log metrics to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("rmse", metrics['rmse'])
                    mlflow.log_metric("mae", metrics['mae'])
                    mlflow.log_metric("r2_score", metrics['r2'])
                    mlflow.log_metric("mape", metrics['mape'])
                    mlflow.log_metric("directional_accuracy", metrics.get('directional_accuracy', 0))
                    mlflow.log_metric("training_time_seconds", training_time)
                    mlflow.log_metric("best_iteration", self.model.best_iteration)
                    
                    # Log top 10 feature importances
                    for idx, (feat, importance) in enumerate(list(self.feature_importance.items())[:10]):
                        mlflow.log_metric(f"feature_importance_{idx+1}_{feat[:20]}", importance)
                    
                    # Log the model
                    mlflow.lightgbm.log_model(self.model, "model")
                    
                    logger.info("✅ MLflow tracking: Metrics and model logged successfully")
                except Exception as e:
                    logger.warning(f"⚠️ MLflow logging failed: {e}")
            
            # Store training history
            self.training_history = {
                'params': params,
                'metrics': metrics,
                'best_iteration': self.model.best_iteration,
                'training_date': datetime.now().isoformat(),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'num_features': len(self.feature_names),
                'evals_result': evals_result
            }
            
            self.is_fitted = True
            
            logger.info(f"✅ Training complete!")
            logger.info(f"   RMSE: {metrics['rmse']:.4f}")
            logger.info(f"   MAE: {metrics['mae']:.4f}")
            logger.info(f"   R²: {metrics['r2']:.4f}")
            logger.info(f"   MAPE: {metrics['mape']:.2f}%")
            logger.info(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.2f}%")
            logger.info(f"   Best iteration: {self.model.best_iteration}")
            logger.info(f"   Training time: {training_time:.2f}s")
            
            return {
                'success': True,
                'metrics': metrics,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history
            }
            
        except Exception as e:
            logger.error(f"❌ Error training LightGBM model: {e}")
            import traceback
            traceback.print_exc()
            
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
    
    def predict(self, data: pd.DataFrame, target_column: str = 'Close',
                forecast_horizon: int = 1, lookback: int = 5) -> Dict[str, Any]:
        """
        Generate predictions using trained model
        
        Args:
            data: Input dataframe
            target_column: Target column name
            forecast_horizon: Number of steps to forecast
            lookback: Number of past days for features
            
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            logger.info(f"🔮 Generating {forecast_horizon}-step forecast...")
            
            predictions = []
            current_data = data.copy()
            
            for step in range(forecast_horizon):
                # Prepare features
                X, _ = self.prepare_features(current_data, target_column, lookback)
                
                if len(X) == 0:
                    logger.warning(f"⚠️ No data available for step {step+1}")
                    break
                
                # Use the last row for prediction
                X_last = X.iloc[[-1]]
                
                # Scale features
                X_scaled = self.scaler.transform(X_last)
                X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
                
                # Make prediction
                pred = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)[0]
                predictions.append(float(pred))
                
                # For multi-step forecasting, append prediction to data
                if step < forecast_horizon - 1:
                    new_row = current_data.iloc[-1:].copy()
                    new_row[target_column] = pred
                    
                    # Update date if available
                    if 'Date' in new_row.columns:
                        last_date = pd.to_datetime(new_row['Date'].values[0])
                        new_row['Date'] = last_date + pd.Timedelta(days=1)
                    
                    current_data = pd.concat([current_data, new_row], ignore_index=True)
            
            logger.info(f"✅ Generated {len(predictions)} predictions")
            
            return {
                'success': True,
                'predictions': predictions,
                'forecast_horizon': forecast_horizon,
                'prediction_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating predictions: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.feature_importance:
            logger.warning("⚠️ No feature importance available. Train model first.")
            return {}
        
        return dict(list(self.feature_importance.items())[:top_n])
    
    def cross_validate(self, data: pd.DataFrame, target_column: str = 'Close',
                      n_splits: int = 5, lookback: int = 5) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            data: Input dataframe
            target_column: Target column name
            n_splits: Number of CV splits
            lookback: Lookback period for features
            
        Returns:
            Cross-validation results
        """
        try:
            logger.info(f"🔄 Performing {n_splits}-fold time series cross-validation...")
            
            # Prepare features
            X, y = self.prepare_features(data, target_column, lookback)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_scores = {
                'rmse': [],
                'mae': [],
                'r2': [],
                'mape': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                logger.info(f"   Fold {fold}/{n_splits}...")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train model
                train_data = lgb.Dataset(X_train_scaled, label=y_train)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                
                model = lgb.train(
                    self.default_params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
                )
                
                # Predict
                y_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
                
                # Calculate metrics
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
                cv_scores['r2'].append(r2_score(y_val, y_pred))
                cv_scores['mape'].append(np.mean(np.abs((y_val - y_pred) / y_val)) * 100)
            
            # Calculate mean and std
            results = {
                'mean_rmse': float(np.mean(cv_scores['rmse'])),
                'std_rmse': float(np.std(cv_scores['rmse'])),
                'mean_mae': float(np.mean(cv_scores['mae'])),
                'std_mae': float(np.std(cv_scores['mae'])),
                'mean_r2': float(np.mean(cv_scores['r2'])),
                'std_r2': float(np.std(cv_scores['r2'])),
                'mean_mape': float(np.mean(cv_scores['mape'])),
                'std_mape': float(np.std(cv_scores['mape'])),
                'n_splits': n_splits
            }
            
            logger.info(f"✅ Cross-validation complete")
            logger.info(f"   Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
            logger.info(f"   Mean MAE: {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
            logger.info(f"   Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in cross-validation: {e}")
            return {}
    
    def save_model(self, symbol: str, model_name: str = 'lightgbm') -> bool:
        """
        Save trained model to disk
        
        Args:
            symbol: Stock symbol
            model_name: Name for the saved model
            
        Returns:
            Success status
        """
        try:
            if not self.is_fitted:
                logger.warning("⚠️ No trained model to save")
                return False
            
            # Create filename
            filename = f"{symbol}_{model_name}_model.txt"
            filepath = os.path.join(self.model_save_path, filename)
            
            # Save LightGBM model
            self.model.save_model(filepath)
            
            # Save scaler and metadata
            metadata_file = os.path.join(self.model_save_path, f"{symbol}_{model_name}_metadata.pkl")
            metadata = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'best_params': self.best_params,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, symbol: str, model_name: str = 'lightgbm') -> bool:
        """
        Load trained model from disk
        
        Args:
            symbol: Stock symbol
            model_name: Name of the saved model
            
        Returns:
            Success status
        """
        try:
            filename = f"{symbol}_{model_name}_model.txt"
            filepath = os.path.join(self.model_save_path, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"❌ Model file not found: {filepath}")
                return False
            
            # Load LightGBM model
            self.model = lgb.Booster(model_file=filepath)
            
            # Load metadata
            metadata_file = os.path.join(self.model_save_path, f"{symbol}_{model_name}_metadata.pkl")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.scaler = metadata['scaler']
                self.feature_names = metadata['feature_names']
                self.best_params = metadata['best_params']
                self.feature_importance = metadata['feature_importance']
                self.training_history = metadata['training_history']
            
            self.is_fitted = True
            logger.info(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False


# Singleton instance
_lightgbm_forecaster_instance = None

def get_lightgbm_forecaster(model_save_path: str = 'saved_models/') -> LightGBMForecaster:
    """Get singleton instance of LightGBM forecaster"""
    global _lightgbm_forecaster_instance
    if _lightgbm_forecaster_instance is None:
        _lightgbm_forecaster_instance = LightGBMForecaster(model_save_path)
    return _lightgbm_forecaster_instance


if __name__ == "__main__":
    # Demo usage
    logger.info("LightGBM Forecaster Module - Demo")
    forecaster = get_lightgbm_forecaster()
    logger.info("✅ Module ready for use")
