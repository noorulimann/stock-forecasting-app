"""
Performance evaluation module for forecasting models
Handles calculation of various metrics and model comparison
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try importing plotting libraries with fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Basic plotting functionality disabled.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive plotting functionality disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates forecasting model performance"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Unknown") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing all metrics
        """
        try:
            # Ensure arrays are 1D
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            
            # Basic validation
            if len(y_true) != len(y_pred):
                raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
            
            if len(y_true) == 0:
                raise ValueError("Empty arrays provided")
            
            # Calculate metrics
            metrics = {}
            
            # Mean Squared Error and Root Mean Squared Error
            mse = mean_squared_error(y_true, y_pred)
            metrics['mse'] = float(mse)
            metrics['rmse'] = float(np.sqrt(mse))
            
            # Mean Absolute Error
            mae = mean_absolute_error(y_true, y_pred)
            metrics['mae'] = float(mae)
            
            # Mean Absolute Percentage Error
            # Avoid division by zero
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics['mape'] = float(mape)
            else:
                metrics['mape'] = float('inf')
            
            # Directional Accuracy (percentage of correct direction predictions)
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(true_direction == pred_direction) * 100
                metrics['directional_accuracy'] = float(directional_accuracy)
            else:
                metrics['directional_accuracy'] = 0.0
            
            # R-squared (coefficient of determination)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            metrics['r2'] = float(r2)
            
            # Mean Error (bias)
            me = np.mean(y_pred - y_true)
            metrics['mean_error'] = float(me)
            
            # Standard deviation of errors
            metrics['std_error'] = float(np.std(y_pred - y_true))
            
            # Min and Max errors
            errors = y_pred - y_true
            metrics['min_error'] = float(np.min(errors))
            metrics['max_error'] = float(np.max(errors))
            
            # Symmetric Mean Absolute Percentage Error (SMAPE)
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mask = denominator != 0
            if np.any(mask):
                smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
                metrics['smape'] = float(smape)
            else:
                metrics['smape'] = float('inf')
            
            logger.info(f"Calculated metrics for {model_name}: RMSE={metrics['rmse']:.4f}, "
                       f"MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return {
                'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf'),
                'mape': float('inf'), 'directional_accuracy': 0.0, 'r2': -float('inf'),
                'mean_error': 0.0, 'std_error': float('inf'), 'min_error': 0.0,
                'max_error': 0.0, 'smape': float('inf')
            }
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models and rank them
        
        Args:
            results: Dictionary with model_name -> metrics mapping
            
        Returns:
            DataFrame with model comparison
        """
        try:
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results).T
            
            # Sort by RMSE (lower is better)
            df = df.sort_values('rmse')
            
            # Add ranking
            df['rank'] = range(1, len(df) + 1)
            
            # Calculate relative performance (best model = 100%)
            best_rmse = df['rmse'].min()
            df['relative_rmse'] = (df['rmse'] / best_rmse) * 100
            
            # Round numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(4)
            
            logger.info(f"Model comparison completed for {len(results)} models")
            return df
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()
    
    def calculate_forecast_intervals(self, predictions: np.ndarray, 
                                   residuals: np.ndarray, 
                                   confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for forecasts
        
        Args:
            predictions: Point forecasts
            residuals: Historical prediction residuals
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        try:
            predictions = np.array(predictions)
            residuals = np.array(residuals)
            
            # Calculate residual standard deviation
            residual_std = np.std(residuals)
            
            # Calculate z-score for confidence level
            from scipy import stats
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            # Calculate intervals
            margin = z_score * residual_std
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
            
            logger.info(f"Calculated {confidence_level*100}% prediction intervals")
            return lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Error calculating forecast intervals: {e}")
            # Return simple intervals based on predictions
            margin = np.std(predictions) * 0.1  # 10% margin as fallback
            return predictions - margin, predictions + margin
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str, timestamps: List = None) -> go.Figure:
        """
        Create interactive plot comparing actual vs predicted values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Model name for title
            timestamps: Optional timestamps for x-axis
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            x_axis = timestamps if timestamps else list(range(len(y_true)))
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=y_true,
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=y_pred,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Calculate metrics for subtitle
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            subtitle = f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | MAPE: {metrics['mape']:.2f}%"
            
            fig.update_layout(
                title=f"{model_name} Predictions<br><sub>{subtitle}</sub>",
                xaxis_title="Time",
                yaxis_title="Price",
                hovermode='x unified',
                showlegend=True,
                width=800,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction plot: {e}")
            return go.Figure()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str) -> go.Figure:
        """
        Create residual analysis plots
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Model name for title
            
        Returns:
            Plotly figure with residual plots
        """
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly not available. Skipping residual analysis plots.")
                return ""
            
            residuals = y_pred - y_true
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Residuals vs Fitted', 'Residuals Distribution', 
                              'Q-Q Plot', 'Residuals vs Time'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Residuals vs Fitted
            fig.add_trace(go.Scatter(
                x=y_pred, y=residuals, mode='markers',
                name='Residuals', showlegend=False
            ), row=1, col=1)
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residuals histogram
            fig.add_trace(go.Histogram(
                x=residuals, nbinsx=20, name='Distribution', showlegend=False
            ), row=1, col=2)
            
            # Q-Q plot (simplified)
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles, y=sample_quantiles, mode='markers',
                name='Q-Q', showlegend=False
            ), row=2, col=1)
            
            # Diagonal line for Q-Q plot
            min_val, max_val = min(theoretical_quantiles.min(), sample_quantiles.min()), \
                              max(theoretical_quantiles.max(), sample_quantiles.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                line=dict(dash='dash', color='red'), name='Perfect Fit', showlegend=False
            ), row=2, col=1)
            
            # Residuals vs Time
            fig.add_trace(go.Scatter(
                x=list(range(len(residuals))), y=residuals, mode='lines+markers',
                name='Time Series', showlegend=False
            ), row=2, col=2)
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(
                title=f"{model_name} - Residual Analysis",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating residual plots: {e}")
            return go.Figure()
    
    def save_metrics(self, model_name: str, instrument: str, metrics: Dict[str, float]):
        """Save metrics to history for later comparison"""
        key = f"{model_name}_{instrument}"
        self.metrics_history[key] = {
            'model_name': model_name,
            'instrument': instrument,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        }
        logger.info(f"Saved metrics for {key}")
    
    def get_best_model(self, instrument: str = None) -> Tuple[str, Dict]:
        """
        Get the best performing model for an instrument
        
        Args:
            instrument: Optional instrument filter
            
        Returns:
            Tuple of (model_name, metrics)
        """
        try:
            if not self.metrics_history:
                return "None", {}
            
            # Filter by instrument if specified
            if instrument:
                filtered_history = {k: v for k, v in self.metrics_history.items() 
                                  if v['instrument'] == instrument}
            else:
                filtered_history = self.metrics_history
            
            if not filtered_history:
                return "None", {}
            
            # Find model with lowest RMSE
            best_key = min(filtered_history.keys(), 
                          key=lambda k: filtered_history[k]['metrics'].get('rmse', float('inf')))
            
            best_model = filtered_history[best_key]
            return best_model['model_name'], best_model['metrics']
            
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return "None", {}

# Singleton instance
_evaluator = None

def get_evaluator() -> ModelEvaluator:
    """Get evaluator singleton instance"""
    global _evaluator
    if _evaluator is None:
        _evaluator = ModelEvaluator()
    return _evaluator