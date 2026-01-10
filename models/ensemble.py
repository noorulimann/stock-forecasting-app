"""
Ensemble Methods for Stock Forecasting

traditional and neural models for improved forecasting accuracy.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import time

# MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("⚠️ MLflow not installed. Experiment tracking disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleForecaster:
    """
    Ensemble forecasting system that combines predictions from multiple models
    using various weighting strategies and selection methods.
    """
    
    def __init__(self):
        """Initialize the ensemble forecaster"""
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.ensemble_strategies = [
            'simple_average',
            'weighted_average',
            'performance_weighted',
            'dynamic_selection',
            'rank_based'
        ]
        
    def register_model(self, model_name: str, model_instance: Any, weight: float = 1.0):
        """Register a model for ensemble"""
        self.models[model_name] = model_instance
        self.model_weights[model_name] = weight
        self.model_performance[model_name] = {}
        logger.info(f"Registered model: {model_name} with weight: {weight}")
    
    def update_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Update performance metrics for a model"""
        if model_name in self.models:
            self.model_performance[model_name] = metrics
            logger.info(f"Updated performance for {model_name}: {metrics}")
    
    def simple_average_ensemble(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Simple average ensemble - equal weight to all models
        """
        if not predictions:
            return []
        
        # Convert to numpy arrays for easier computation
        pred_arrays = [np.array(pred) for pred in predictions.values()]
        
        # Calculate simple average
        ensemble_pred = np.mean(pred_arrays, axis=0)
        
        logger.info(f"Simple average ensemble of {len(predictions)} models")
        return ensemble_pred.tolist()
    
    def weighted_average_ensemble(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Weighted average ensemble using predefined weights
        """
        if not predictions:
            return []
        
        total_weight = 0
        weighted_sum = None
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            pred_array = np.array(pred)
            
            if weighted_sum is None:
                weighted_sum = weight * pred_array
            else:
                weighted_sum += weight * pred_array
            
            total_weight += weight
        
        # Normalize by total weight
        ensemble_pred = weighted_sum / total_weight
        
        logger.info(f"Weighted average ensemble with total weight: {total_weight}")
        return ensemble_pred.tolist()
    
    def performance_weighted_ensemble(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Performance-based weighted ensemble using model accuracy
        """
        if not predictions:
            return []
        
        # Calculate weights based on inverse of MAPE (lower MAPE = higher weight)
        weights = {}
        total_weight = 0
        
        for model_name in predictions.keys():
            if model_name in self.model_performance:
                mape = self.model_performance[model_name].get('mape', 10.0)  # Default high MAPE
                # Inverse weight: better models (lower MAPE) get higher weight
                weight = 1.0 / (mape + 0.01)  # Add small epsilon to avoid division by zero
            else:
                weight = 1.0  # Default weight for models without performance data
            
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        # Calculate weighted ensemble
        ensemble_pred = None
        for model_name, pred in predictions.items():
            weight = weights[model_name]
            pred_array = np.array(pred)
            
            if ensemble_pred is None:
                ensemble_pred = weight * pred_array
            else:
                ensemble_pred += weight * pred_array
        
        logger.info(f"Performance-weighted ensemble: {weights}")
        return ensemble_pred.tolist()
    
    def dynamic_selection_ensemble(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Dynamic model selection - choose best performing model for each prediction
        """
        if not predictions:
            return []
        
        # Select the model with the best overall performance (lowest MAPE)
        best_model = None
        best_mape = float('inf')
        
        for model_name in predictions.keys():
            if model_name in self.model_performance:
                mape = self.model_performance[model_name].get('mape', float('inf'))
                if mape < best_mape:
                    best_mape = mape
                    best_model = model_name
        
        if best_model:
            logger.info(f"Dynamic selection chose: {best_model} (MAPE: {best_mape:.3f})")
            return predictions[best_model]
        else:
            # Fallback to simple average if no performance data
            logger.warning("No performance data available, falling back to simple average")
            return self.simple_average_ensemble(predictions)
    
    def rank_based_ensemble(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Rank-based ensemble using Borda count method
        """
        if not predictions:
            return []
        
        pred_arrays = {name: np.array(pred) for name, pred in predictions.items()}
        n_steps = len(next(iter(pred_arrays.values())))
        ensemble_pred = np.zeros(n_steps)
        
        # For each time step, rank predictions and use Borda count
        for i in range(n_steps):
            step_predictions = {name: arr[i] for name, arr in pred_arrays.items()}
            
            # Sort predictions by value
            sorted_preds = sorted(step_predictions.items(), key=lambda x: x[1])
            
            # Assign Borda count scores (highest value gets highest score)
            weighted_sum = 0
            total_weight = 0
            
            for rank, (model_name, pred_value) in enumerate(sorted_preds):
                score = rank + 1  # Borda count score
                weighted_sum += score * pred_value
                total_weight += score
            
            ensemble_pred[i] = weighted_sum / total_weight
        
        logger.info("Rank-based ensemble using Borda count method")
        return ensemble_pred.tolist()
    
    def generate_ensemble_prediction(self, 
                                   predictions: Dict[str, List[float]], 
                                   strategy: str = 'performance_weighted',
                                   symbol: str = None) -> Dict[str, Any]:
        """
        Generate ensemble prediction using specified strategy with MLflow tracking
        """
        generation_start_time = time.time()
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment("stock-forecasting-ensemble")
                mlflow.start_run(run_name=f"Ensemble_{strategy}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Log parameters
                mlflow.log_param("ensemble_strategy", strategy)
                mlflow.log_param("symbol", symbol or "unknown")
                mlflow.log_param("num_models", len(predictions))
                mlflow.log_param("component_models", str(list(predictions.keys())))
            except Exception as e:
                logger.warning(f"⚠️ MLflow logging initialization failed: {e}")
        
        try:
            if strategy not in self.ensemble_strategies:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {self.ensemble_strategies}")
            
            if not predictions:
                raise ValueError("No predictions provided for ensemble")
            
            # Generate ensemble prediction based on strategy
            if strategy == 'simple_average':
                ensemble_pred = self.simple_average_ensemble(predictions)
            elif strategy == 'weighted_average':
                ensemble_pred = self.weighted_average_ensemble(predictions)
            elif strategy == 'performance_weighted':
                ensemble_pred = self.performance_weighted_ensemble(predictions)
            elif strategy == 'dynamic_selection':
                ensemble_pred = self.dynamic_selection_ensemble(predictions)
            elif strategy == 'rank_based':
                ensemble_pred = self.rank_based_ensemble(predictions)
            else:
                # Fallback to simple average
                ensemble_pred = self.simple_average_ensemble(predictions)
            
            # Calculate ensemble statistics
            ensemble_stats = self._calculate_ensemble_stats(predictions, ensemble_pred)
            
            # Calculate generation time
            generation_time = time.time() - generation_start_time
            
            # Log ensemble metrics to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("ensemble_mean_absolute_deviation", 
                                    ensemble_stats['mean_absolute_deviation'])
                    mlflow.log_metric("ensemble_prediction_spread", 
                                    ensemble_stats['prediction_spread'])
                    mlflow.log_metric("ensemble_mean_variance", 
                                    float(np.mean(ensemble_stats['prediction_variance'])))
                    mlflow.log_metric("generation_time_seconds", generation_time)
                    
                    # Log prediction range
                    ensemble_range = ensemble_stats['ensemble_range']
                    mlflow.log_metric("prediction_min", ensemble_range[0])
                    mlflow.log_metric("prediction_max", ensemble_range[1])
                    
                    logger.info("✅ MLflow tracking: Ensemble metrics logged successfully")
                except Exception as e:
                    logger.warning(f"⚠️ MLflow logging failed: {e}")
            
            logger.info(f"Ensemble prediction generated using {strategy} in {generation_time:.3f}s")
            
            return {
                'ensemble_prediction': ensemble_pred,
                'strategy': strategy,
                'component_models': list(predictions.keys()),
                'n_models': len(predictions),
                'ensemble_stats': ensemble_stats,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {e}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("error", str(e))
                except:
                    pass
            raise
        finally:
            # End MLflow run
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.end_run()
                except:
                    pass
    
    def _calculate_ensemble_stats(self, predictions: Dict[str, List[float]], 
                                 ensemble_pred: List[float]) -> Dict[str, Any]:
        """Calculate statistics about the ensemble prediction"""
        
        # Convert predictions to arrays
        pred_arrays = [np.array(pred) for pred in predictions.values()]
        ensemble_array = np.array(ensemble_pred)
        
        # Calculate variance across models for each time step
        prediction_variance = np.var(pred_arrays, axis=0)
        
        # Calculate mean absolute deviation from ensemble
        mean_abs_deviation = np.mean([
            np.mean(np.abs(arr - ensemble_array)) 
            for arr in pred_arrays
        ])
        
        # Calculate confidence interval (based on prediction spread)
        confidence_interval = []
        for i in range(len(ensemble_pred)):
            step_preds = [arr[i] for arr in pred_arrays]
            ci_lower = np.percentile(step_preds, 25)  # Q1
            ci_upper = np.percentile(step_preds, 75)  # Q3
            confidence_interval.append([ci_lower, ci_upper])
        
        return {
            'prediction_variance': prediction_variance.tolist(),
            'mean_absolute_deviation': float(mean_abs_deviation),
            'confidence_intervals': confidence_interval,
            'ensemble_range': [float(np.min(ensemble_pred)), float(np.max(ensemble_pred))],
            'prediction_spread': float(np.std([np.std(arr) for arr in pred_arrays]))
        }
    
    def compare_strategies(self, predictions: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Compare all ensemble strategies on the given predictions
        """
        strategy_results = {}
        
        for strategy in self.ensemble_strategies:
            try:
                result = self.generate_ensemble_prediction(predictions, strategy)
                strategy_results[strategy] = result
                logger.info(f"Successfully generated prediction using {strategy}")
            except Exception as e:
                logger.error(f"Error with strategy {strategy}: {e}")
                strategy_results[strategy] = None
        
        # Find the strategy with lowest variance (most stable)
        best_strategy = None
        lowest_variance = float('inf')
        
        for strategy, result in strategy_results.items():
            if result and result['ensemble_stats']:
                avg_variance = np.mean(result['ensemble_stats']['prediction_variance'])
                if avg_variance < lowest_variance:
                    lowest_variance = avg_variance
                    best_strategy = strategy
        
        return {
            'strategy_results': strategy_results,
            'recommended_strategy': best_strategy,
            'comparison_timestamp': datetime.now().isoformat()
        }
    
    def save_ensemble_results(self, results: Dict[str, Any], filepath: str):
        """Save ensemble results to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Ensemble results saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving ensemble results: {e}")


class ModelComparison:
    """
    Model comparison framework for evaluating and ranking different forecasting models
    """
    
    def __init__(self):
        """Initialize model comparison framework"""
        self.model_results = {}
        self.comparison_metrics = ['rmse', 'mae', 'mape', 'r2', 'directional_accuracy']
        
    def add_model_results(self, model_name: str, predictions: List[float], 
                         actual_values: List[float], metrics: Dict[str, float]):
        """Add model results for comparison"""
        self.model_results[model_name] = {
            'predictions': predictions,
            'actual_values': actual_values,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        logger.info(f"Added results for model: {model_name}")
    
    def rank_models(self, metric: str = 'mape') -> List[Tuple[str, float]]:
        """
        Rank models based on a specific metric
        Lower is better for RMSE, MAE, MAPE
        Higher is better for R², directional_accuracy
        """
        if metric not in self.comparison_metrics:
            raise ValueError(f"Unknown metric: {metric}. Available: {self.comparison_metrics}")
        
        model_scores = []
        for model_name, results in self.model_results.items():
            if metric in results['metrics']:
                score = results['metrics'][metric]
                model_scores.append((model_name, score))
        
        # Sort based on metric type
        reverse = metric in ['r2', 'directional_accuracy']  # Higher is better for these
        model_scores.sort(key=lambda x: x[1], reverse=reverse)
        
        logger.info(f"Ranked models by {metric}: {[name for name, _ in model_scores]}")
        return model_scores
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive model comparison report"""
        
        report = {
            'summary': {
                'total_models': len(self.model_results),
                'comparison_metrics': self.comparison_metrics,
                'generated_at': datetime.now().isoformat()
            },
            'rankings': {},
            'detailed_results': self.model_results
        }
        
        # Generate rankings for each metric
        for metric in self.comparison_metrics:
            try:
                rankings = self.rank_models(metric)
                report['rankings'][metric] = rankings
            except Exception as e:
                logger.error(f"Error ranking by {metric}: {e}")
                report['rankings'][metric] = []
        
        # Identify best overall model (lowest average rank)
        model_ranks = {model: [] for model in self.model_results.keys()}
        
        for metric, rankings in report['rankings'].items():
            for rank, (model_name, score) in enumerate(rankings):
                model_ranks[model_name].append(rank + 1)  # 1-based ranking
        
        # Calculate average rank for each model
        avg_ranks = {}
        for model, ranks in model_ranks.items():
            if ranks:
                avg_ranks[model] = np.mean(ranks)
        
        # Best model has lowest average rank
        if avg_ranks:
            best_model = min(avg_ranks.items(), key=lambda x: x[1])
            report['summary']['best_overall_model'] = {
                'name': best_model[0],
                'average_rank': best_model[1]
            }
        
        return report


def get_ensemble_forecaster():
    """Factory function to get EnsembleForecaster instance"""
    return EnsembleForecaster()


def get_model_comparison():
    """Factory function to get ModelComparison instance"""
    return ModelComparison()


# Global instances
_ensemble_forecaster_instance = None
_model_comparison_instance = None


def get_ensemble_forecaster_instance():
    """Get singleton ensemble forecaster instance"""
    global _ensemble_forecaster_instance
    if _ensemble_forecaster_instance is None:
        _ensemble_forecaster_instance = EnsembleForecaster()
    return _ensemble_forecaster_instance


def get_model_comparison_instance():
    """Get singleton model comparison instance"""
    global _model_comparison_instance
    if _model_comparison_instance is None:
        _model_comparison_instance = ModelComparison()
    return _model_comparison_instance