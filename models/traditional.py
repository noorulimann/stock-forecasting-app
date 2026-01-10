"""
Traditional forecasting models implementation
Includes ARIMA, Moving Averages, and VAR models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Traditional ML imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("⚠️ MLflow not installed. Experiment tracking disabled.")

# Data processing
from data.database import get_database_manager
from data.collector import get_data_collector
from data.processor import get_data_processor
from utils.evaluator import get_evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovingAverageModel:
    """Simple and Exponential Moving Average models"""
    
    def __init__(self):
        self.model_params = {}
        self.is_fitted = False
    
    def fit(self, data: pd.Series, window: int = 20, alpha: float = 0.3):
        """
        Fit moving average models
        
        Args:
            data: Time series data
            window: Window size for SMA
            alpha: Smoothing parameter for EMA
        """
        try:
            self.data = data.copy()
            self.window = window
            self.alpha = alpha
            
            # Calculate moving averages
            self.sma = data.rolling(window=window).mean()
            self.ema = data.ewm(alpha=alpha).mean()
            
            self.model_params = {
                'window': window,
                'alpha': alpha,
                'data_length': len(data)
            }
            
            self.is_fitted = True
            logger.info(f"Moving Average models fitted with window={window}, alpha={alpha}")
            
        except Exception as e:
            logger.error(f"Error fitting Moving Average models: {e}")
            self.is_fitted = False
    
    def predict(self, steps: int = 1, method: str = 'sma') -> np.ndarray:
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
            method: 'sma' or 'ema'
            
        Returns:
            Array of predictions
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            if method == 'sma':
                # SMA: use last window values
                last_values = self.data.tail(self.window)
                prediction = last_values.mean()
                
            elif method == 'ema':
                # EMA: use last EMA value
                prediction = self.ema.iloc[-1]
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # For multiple steps, assume constant prediction (naive approach)
            predictions = np.full(steps, prediction)
            
            logger.info(f"Generated {steps} predictions using {method.upper()}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return np.array([])
    
    def get_fitted_values(self, method: str = 'sma') -> pd.Series:
        """Get fitted values for evaluation"""
        if not self.is_fitted:
            return pd.Series()
        
        if method == 'sma':
            return self.sma
        elif method == 'ema':
            return self.ema
        else:
            return pd.Series()

class ARIMAModel:
    """ARIMA (AutoRegressive Integrated Moving Average) model"""
    
    def __init__(self):
        self.model = None
        self.model_fit = None
        self.order = None
        self.is_fitted = False
    
    def _check_stationarity(self, data: pd.Series) -> Tuple[bool, Dict]:
        """Check if series is stationary using ADF test"""
        try:
            result = adfuller(data.dropna())
            
            is_stationary = result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
            
            adf_result = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': is_stationary
            }
            
            logger.info(f"ADF test: p-value={result[1]:.4f}, stationary={is_stationary}")
            return is_stationary, adf_result
            
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return False, {}
    
    def _auto_order_selection(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Automatic order selection using AIC"""
        try:
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Check stationarity to guide d parameter
            is_stationary, _ = self._check_stationarity(data)
            d_range = [0] if is_stationary else [1, 2]
            
            for p in range(max_p + 1):
                for d in d_range:
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            model_fit = model.fit()
                            
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                best_order = (p, d, q)
                                
                        except:
                            continue
            
            logger.info(f"Auto-selected ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            return best_order
            
        except Exception as e:
            logger.error(f"Error in auto order selection: {e}")
            return (1, 1, 1)  # Default order
    
    def fit(self, data: pd.Series, order: Optional[Tuple[int, int, int]] = None):
        """
        Fit ARIMA model
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q). If None, auto-select
        """
        try:
            self.data = data.copy()
            
            # Auto-select order if not provided
            if order is None:
                self.order = self._auto_order_selection(data)
            else:
                self.order = order
            
            # Fit model
            self.model = ARIMA(data, order=self.order)
            self.model_fit = self.model.fit()
            
            self.is_fitted = True
            
            # Log model summary
            logger.info(f"ARIMA{self.order} fitted successfully")
            logger.info(f"AIC: {self.model_fit.aic:.2f}, BIC: {self.model_fit.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            self.is_fitted = False
    
    def predict(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts with confidence intervals
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            # Generate forecast
            forecast_result = self.model_fit.forecast(steps=steps, alpha=0.05)  # 95% confidence
            
            predictions = forecast_result.values if hasattr(forecast_result, 'values') else forecast_result
            
            # Get confidence intervals
            conf_int = self.model_fit.get_forecast(steps=steps).conf_int()
            
            logger.info(f"Generated {steps} ARIMA predictions")
            return predictions, conf_int.values
            
        except Exception as e:
            logger.error(f"Error generating ARIMA predictions: {e}")
            return np.array([]), np.array([])
    
    def get_fitted_values(self) -> pd.Series:
        """Get fitted values for evaluation"""
        if not self.is_fitted:
            return pd.Series()
        
        return self.model_fit.fittedvalues
    
    def diagnostic_tests(self) -> Dict[str, Any]:
        """Run diagnostic tests on residuals"""
        try:
            if not self.is_fitted:
                return {}
            
            residuals = self.model_fit.resid
            
            # Ljung-Box test for autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            diagnostics = {
                'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std(),
                'aic': self.model_fit.aic,
                'bic': self.model_fit.bic,
                'order': self.order
            }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error in diagnostic tests: {e}")
            return {}

class VARModel:
    """Vector AutoRegression model for multivariate time series"""
    
    def __init__(self):
        self.model = None
        self.model_fit = None
        self.lag_order = None
        self.is_fitted = False
    
    def _prepare_multivariate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare multivariate data for VAR"""
        try:
            # Select relevant columns (OHLC + Volume if available)
            var_columns = ['Close', 'Open', 'High', 'Low']
            if 'Volume' in data.columns and data['Volume'].sum() > 0:
                var_columns.append('Volume')
            
            # Create feature matrix
            var_data = data[var_columns].copy()
            
            # Add some technical indicators
            var_data['Returns'] = data['Close'].pct_change()
            var_data['SMA_5'] = data['Close'].rolling(5).mean()
            var_data['Volatility'] = data['Close'].rolling(10).std()
            
            # Drop NaN values
            var_data = var_data.dropna()
            
            logger.info(f"Prepared VAR data with {len(var_data.columns)} variables, {len(var_data)} observations")
            return var_data
            
        except Exception as e:
            logger.error(f"Error preparing VAR data: {e}")
            return pd.DataFrame()
    
    def _select_lag_order(self, data: pd.DataFrame, max_lags: int = 10) -> int:
        """Select optimal lag order using information criteria"""
        try:
            model = VAR(data)
            lag_order_results = model.select_order(maxlags=max_lags)
            
            # Use AIC for selection
            optimal_lag = lag_order_results.aic
            
            logger.info(f"Selected VAR lag order: {optimal_lag}")
            return optimal_lag
            
        except Exception as e:
            logger.error(f"Error selecting lag order: {e}")
            return 1  # Default to 1 lag
    
    def fit(self, data: pd.DataFrame, lag_order: Optional[int] = None):
        """
        Fit VAR model
        
        Args:
            data: Multivariate time series data
            lag_order: Number of lags. If None, auto-select
        """
        try:
            # Prepare data
            self.var_data = self._prepare_multivariate_data(data)
            
            if self.var_data.empty:
                raise ValueError("Failed to prepare VAR data")
            
            # Select lag order
            if lag_order is None:
                self.lag_order = self._select_lag_order(self.var_data)
            else:
                self.lag_order = lag_order
            
            # Fit model
            self.model = VAR(self.var_data)
            self.model_fit = self.model.fit(self.lag_order)
            
            self.is_fitted = True
            
            logger.info(f"VAR model fitted with {self.lag_order} lags")
            logger.info(f"AIC: {self.model_fit.aic:.2f}, BIC: {self.model_fit.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting VAR model: {e}")
            self.is_fitted = False
    
    def predict(self, steps: int = 1, target_variable: str = 'Close') -> np.ndarray:
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
            target_variable: Variable to forecast
            
        Returns:
            Array of predictions for target variable
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            # Generate forecast
            forecast = self.model_fit.forecast(self.var_data.values[-self.lag_order:], steps)
            
            # Extract target variable predictions
            if target_variable in self.var_data.columns:
                target_idx = self.var_data.columns.get_loc(target_variable)
                predictions = forecast[:, target_idx]
            else:
                # Default to first column
                predictions = forecast[:, 0]
            
            logger.info(f"Generated {steps} VAR predictions for {target_variable}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating VAR predictions: {e}")
            return np.array([])
    
    def get_fitted_values(self, target_variable: str = 'Close') -> pd.Series:
        """Get fitted values for evaluation"""
        try:
            if not self.is_fitted:
                return pd.Series()
            
            fitted_values = self.model_fit.fittedvalues
            
            if target_variable in fitted_values.columns:
                return fitted_values[target_variable]
            else:
                return fitted_values.iloc[:, 0]  # Default to first column
                
        except Exception as e:
            logger.error(f"Error getting VAR fitted values: {e}")
            return pd.Series()

class TraditionalModels:
    """Unified interface for all traditional models"""
    
    def __init__(self):
        self.models = {
            'SMA': MovingAverageModel(),
            'EMA': MovingAverageModel(),
            'ARIMA': ARIMAModel(),
            'VAR': VARModel()
        }
        self.fitted_models = {}
        self.evaluator = get_evaluator()
    
    def fit_all_models(self, data: pd.DataFrame, target_column: str = 'Close', symbol: str = None) -> Dict[str, bool]:
        """
        Fit all traditional models with MLflow tracking
        
        Args:
            data: Historical price data
            target_column: Target variable for forecasting
            symbol: Stock symbol for MLflow tracking
            
        Returns:
            Dictionary with model fit status
        """
        training_start_time = time.time()
        
        # Start MLflow experiment
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment("stock-forecasting-traditional")
                mlflow.start_run(run_name=f"TraditionalModels_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Log basic parameters
                mlflow.log_param("model_types", "SMA,EMA,ARIMA,VAR")
                mlflow.log_param("symbol", symbol or "unknown")
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("data_size", len(data))
            except Exception as e:
                logger.warning(f"⚠️ MLflow logging initialization failed: {e}")
        
        results = {}
        model_metrics = {}
        
        try:
            target_series = data[target_column].dropna()
            
            # Fit Moving Averages
            try:
                sma_model = MovingAverageModel()
                sma_model.fit(target_series, window=20)
                self.fitted_models['SMA'] = sma_model
                results['SMA'] = True
                
                # Log SMA metrics
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_param("SMA_window", 20)
                        mlflow.log_metric("SMA_status", 1)
                    except:
                        pass
                
                logger.info("✅ SMA model fitted successfully")
            except Exception as e:
                logger.error(f"❌ SMA model fitting failed: {e}")
                results['SMA'] = False
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metric("SMA_status", 0)
                    except:
                        pass
            
            try:
                ema_model = MovingAverageModel()
                ema_model.fit(target_series, alpha=0.3)
                self.fitted_models['EMA'] = ema_model
                results['EMA'] = True
                
                # Log EMA metrics
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_param("EMA_alpha", 0.3)
                        mlflow.log_metric("EMA_status", 1)
                    except:
                        pass
                
                logger.info("✅ EMA model fitted successfully")
            except Exception as e:
                logger.error(f"❌ EMA model fitting failed: {e}")
                results['EMA'] = False
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metric("EMA_status", 0)
                    except:
                        pass
            
            # Fit ARIMA
            try:
                arima_start = time.time()
                arima_model = ARIMAModel()
                arima_model.fit(target_series)
                arima_time = time.time() - arima_start
                self.fitted_models['ARIMA'] = arima_model
                results['ARIMA'] = True
                
                # Log ARIMA metrics
                if MLFLOW_AVAILABLE:
                    try:
                        if arima_model.order:
                            mlflow.log_param("ARIMA_order", str(arima_model.order))
                        mlflow.log_metric("ARIMA_status", 1)
                        mlflow.log_metric("ARIMA_training_time", arima_time)
                    except:
                        pass
                
                logger.info("✅ ARIMA model fitted successfully")
            except Exception as e:
                logger.error(f"❌ ARIMA model fitting failed: {e}")
                results['ARIMA'] = False
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metric("ARIMA_status", 0)
                    except:
                        pass
            
            # Fit VAR
            try:
                var_start = time.time()
                var_model = VARModel()
                var_model.fit(data)
                var_time = time.time() - var_start
                self.fitted_models['VAR'] = var_model
                results['VAR'] = True
                
                # Log VAR metrics
                if MLFLOW_AVAILABLE:
                    try:
                        if var_model.lag_order:
                            mlflow.log_param("VAR_lag_order", var_model.lag_order)
                        mlflow.log_metric("VAR_status", 1)
                        mlflow.log_metric("VAR_training_time", var_time)
                    except:
                        pass
                
                logger.info("✅ VAR model fitted successfully")
            except Exception as e:
                logger.error(f"❌ VAR model fitting failed: {e}")
                results['VAR'] = False
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_metric("VAR_status", 0)
                    except:
                        pass
            
            # Calculate overall training time
            total_training_time = time.time() - training_start_time
            
            successful = sum(results.values())
            total = len(results)
            logger.info(f"Model fitting complete: {successful}/{total} models successful")
            logger.info(f"Total training time: {total_training_time:.2f}s")
            
            # Log summary metrics
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("successful_models", successful)
                    mlflow.log_metric("total_models", total)
                    mlflow.log_metric("total_training_time_seconds", total_training_time)
                    logger.info("✅ MLflow tracking: Summary metrics logged successfully")
                except Exception as e:
                    logger.warning(f"⚠️ MLflow summary logging failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fit_all_models: {e}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("error", str(e))
                except:
                    pass
            return {model: False for model in self.models.keys()}
        finally:
            # End MLflow run
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.end_run()
                except:
                    pass
    
    def predict_all_models(self, steps: int = 1) -> Dict[str, np.ndarray]:
        """Generate predictions from all fitted models"""
        predictions = {}
        
        for model_name, model in self.fitted_models.items():
            try:
                if model_name in ['SMA', 'EMA']:
                    method = model_name.lower()
                    pred = model.predict(steps, method=method)
                elif model_name == 'ARIMA':
                    pred, _ = model.predict(steps)
                elif model_name == 'VAR':
                    pred = model.predict(steps)
                else:
                    continue
                
                predictions[model_name] = pred
                logger.info(f"✅ {model_name} predictions generated: {len(pred)} steps")
                
            except Exception as e:
                logger.error(f"❌ {model_name} prediction failed: {e}")
                predictions[model_name] = np.array([])
        
        return predictions
    
    def evaluate_all_models(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """Evaluate all fitted models on historical data"""
        try:
            target_series = data[target_column].dropna()
            evaluation_results = {}
            
            for model_name, model in self.fitted_models.items():
                try:
                    # Get fitted values
                    if model_name in ['SMA', 'EMA']:
                        method = model_name.lower()
                        fitted_values = model.get_fitted_values(method=method)
                    elif model_name == 'ARIMA':
                        fitted_values = model.get_fitted_values()
                    elif model_name == 'VAR':
                        fitted_values = model.get_fitted_values(target_column)
                    else:
                        continue
                    
                    # Align data
                    common_index = target_series.index.intersection(fitted_values.index)
                    if len(common_index) > 0:
                        y_true = target_series.loc[common_index]
                        y_pred = fitted_values.loc[common_index]
                        
                        # Calculate metrics
                        metrics = self.evaluator.calculate_metrics(y_true, y_pred, model_name)
                        evaluation_results[model_name] = metrics
                        
                        logger.info(f"✅ {model_name} evaluation complete")
                    else:
                        logger.warning(f"⚠️ No common index for {model_name} evaluation")
                        
                except Exception as e:
                    logger.error(f"❌ {model_name} evaluation failed: {e}")
            
            # Compare models
            comparison_df = self.evaluator.compare_models(evaluation_results)
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error in evaluate_all_models: {e}")
            return pd.DataFrame()
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about fitted models"""
        info = {}
        
        for model_name, model in self.fitted_models.items():
            try:
                if model_name == 'ARIMA' and hasattr(model, 'order'):
                    info[model_name] = {
                        'type': 'ARIMA',
                        'order': model.order,
                        'fitted': model.is_fitted
                    }
                elif model_name in ['SMA', 'EMA'] and hasattr(model, 'model_params'):
                    info[model_name] = {
                        'type': 'Moving Average',
                        'parameters': model.model_params,
                        'fitted': model.is_fitted
                    }
                elif model_name == 'VAR' and hasattr(model, 'lag_order'):
                    info[model_name] = {
                        'type': 'VAR',
                        'lag_order': model.lag_order,
                        'fitted': model.is_fitted
                    }
                    
            except Exception as e:
                logger.error(f"Error getting info for {model_name}: {e}")
        
        return info

# Singleton instance
_traditional_models = None

def get_traditional_models() -> TraditionalModels:
    """Get traditional models singleton instance"""
    global _traditional_models
    if _traditional_models is None:
        _traditional_models = TraditionalModels()
    return _traditional_models