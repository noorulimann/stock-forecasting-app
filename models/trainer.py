"""
Model training pipeline for coordinated model training and evaluation
Handles data preparation, model fitting, prediction generation, and performance tracking
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import time

# MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("⚠️ MLflow not installed. Experiment tracking disabled.")

# Import model components
from data.database import get_database_manager
from data.collector import get_data_collector
from data.processor import get_data_processor
from models.traditional import get_traditional_models
from models.neural import get_neural_models
from models.lightgbm_model import get_lightgbm_forecaster
from models.ensemble import get_ensemble_forecaster_instance, get_model_comparison_instance
from utils.evaluator import get_evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Unified training pipeline for all forecasting models"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.data_collector = get_data_collector()
        self.data_processor = get_data_processor()
        self.traditional_models = get_traditional_models()
        self.neural_models = get_neural_models()
        self.lightgbm_forecaster = get_lightgbm_forecaster()
        self.ensemble_forecaster = get_ensemble_forecaster_instance()
        self.model_comparison = get_model_comparison_instance()
        self.evaluator = get_evaluator()
        
        self.training_results = {}
        self.model_performance = {}
    
    def prepare_training_data(self, symbol: str, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and testing datasets
        
        Args:
            symbol: Stock/crypto symbol
            train_size: Proportion of data for training
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            logger.info(f"🔄 Preparing training data for {symbol}")
            
            # Get historical data
            historical_data = self.db_manager.get_historical_data(symbol)
            
            if historical_data.empty:
                logger.error(f"❌ No historical data found for {symbol}")
                return pd.DataFrame(), pd.DataFrame()
            
            # Sort by date
            historical_data = historical_data.sort_values('Date')
            
            # Add technical indicators
            processed_data = self.data_processor.prepare_features(historical_data)
            
            if processed_data.empty:
                logger.error(f"❌ Failed to prepare features for {symbol}")
                return pd.DataFrame(), pd.DataFrame()
            
            # Split into train/test
            split_index = int(len(processed_data) * train_size)
            train_data = processed_data.iloc[:split_index].copy()
            test_data = processed_data.iloc[split_index:].copy()
            
            logger.info(f"✅ Data prepared - Train: {len(train_data)} samples, Test: {len(test_data)} samples")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data for {symbol}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def train_traditional_models(self, symbol: str, train_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Train all traditional models for a symbol
        
        Args:
            symbol: Stock/crypto symbol
            train_data: Training dataset
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"🔄 Training traditional models for {symbol}")
            
            # Fit all traditional models
            training_results = self.traditional_models.fit_all_models(train_data, target_column='Close', symbol=symbol)
            
            # Store training results
            self.training_results[symbol] = {
                'traditional': training_results,
                'train_data_size': len(train_data),
                'training_date': datetime.now().isoformat()
            }
            
            successful_models = [model for model, success in training_results.items() if success]
            total_models = len(training_results)
            
            logger.info(f"✅ Traditional model training complete for {symbol}: {len(successful_models)}/{total_models} successful")
            logger.info(f"   Successful models: {successful_models}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error training traditional models for {symbol}: {e}")
            return {}
    
    def train_neural_models(self, symbol: str, model_types: List[str] = None, **kwargs) -> Dict:
        """
        Train neural models for a symbol
        
        Args:
            symbol: Stock/crypto symbol
            model_types: List of neural model types to train (lstm, gru, transformer)
            **kwargs: Additional training parameters (epochs, batch_size, etc.)
            
        Returns:
            Dictionary with training results and predictions
        """
        try:
            logger.info(f"🔄 Training neural models for {symbol}")
            
            # Get training data
            train_data = self.db_manager.get_historical_data(symbol, limit=500)
            if train_data is None or len(train_data) < 100:
                return {
                    'success': False,
                    'error': f'Insufficient training data for {symbol}'
                }
            
            # Default neural models to train
            if model_types is None:
                model_types = ['lstm', 'gru']  # Skip transformer for faster training
            
            # Training parameters
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            forecast_horizon = kwargs.get('forecast_horizon', 5)
            
            training_results = {}
            predictions = {}
            
            for model_type in model_types:
                logger.info(f"🤖 Training {model_type.upper()} model for {symbol}")
                
                # Train the model
                train_result = self.neural_models.train_model(
                    data=train_data,
                    model_type=model_type,
                    symbol=symbol,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                training_results[model_type] = train_result
                
                if train_result.get('success', False):
                    # Generate predictions
                    pred_result = self.neural_models.predict(
                        data=train_data,
                        model_type=model_type,
                        symbol=symbol,
                        forecast_horizon=forecast_horizon
                    )
                    
                    if pred_result.get('success', False):
                        predictions[model_type] = pred_result.get('forecast', [])
                        
                        # Save to database
                        self.save_results_to_database(
                            symbol=symbol,
                            predictions={model_type: pred_result.get('forecast', [])},
                            evaluation_results={}
                        )
                
                logger.info(f"✅ {model_type.upper()} training completed for {symbol}")
            
            # Update training results
            if symbol not in self.training_results:
                self.training_results[symbol] = {}
            
            self.training_results[symbol]['neural'] = training_results
            
            # Calculate success metrics
            successful_models = [m for m, r in training_results.items() if r.get('success', False)]
            
            return {
                'success': len(successful_models) > 0,
                'training_results': training_results,
                'predictions': predictions,
                'models_trained': successful_models,
                'training_summary': {
                    'total_models': len(model_types),
                    'successful_models': len(successful_models),
                    'failed_models': len(model_types) - len(successful_models)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error training neural models for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            self.training_results[symbol]['neural_train_data_size'] = len(train_data)
            self.training_results[symbol]['neural_training_date'] = datetime.now().isoformat()
            
            successful_models = [model for model, success in training_results.items() if success]
            total_models = len(training_results)
            
            logger.info(f"✅ Neural model training complete for {symbol}: {len(successful_models)}/{total_models} successful")
            logger.info(f"   Successful models: {successful_models}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error training neural models for {symbol}: {e}")
            return {}
    
    def evaluate_models(self, symbol: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Evaluate trained models on train and test data
        
        Args:
            symbol: Stock/crypto symbol
            train_data: Training dataset
            test_data: Testing dataset
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"🔄 Evaluating models for {symbol}")
            
            evaluation_results = {}
            
            # Evaluate traditional models on training data (in-sample)
            traditional_train_eval = self.traditional_models.evaluate_all_models(train_data, target_column='Close')
            if not traditional_train_eval.empty:
                evaluation_results['traditional_train'] = traditional_train_eval
                logger.info("✅ Traditional training data evaluation complete")
            
            # Evaluate traditional models on test data (out-of-sample)
            traditional_test_eval = self.traditional_models.evaluate_all_models(test_data, target_column='Close')
            if not traditional_test_eval.empty:
                evaluation_results['traditional_test'] = traditional_test_eval
                logger.info("✅ Traditional test data evaluation complete")
            
            # Evaluate neural models on training data (in-sample)
            neural_train_eval = self.neural_models.evaluate_all_models(train_data, target_column='Close')
            if not neural_train_eval.empty:
                evaluation_results['neural_train'] = neural_train_eval
                logger.info("✅ Neural training data evaluation complete")
            
            # Evaluate neural models on test data (out-of-sample)
            neural_test_eval = self.neural_models.evaluate_all_models(test_data, target_column='Close')
            if not neural_test_eval.empty:
                evaluation_results['neural_test'] = neural_test_eval
                logger.info("✅ Neural test data evaluation complete")
            
            # Store performance results
            self.model_performance[symbol] = evaluation_results
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating models for {symbol}: {e}")
            return {}
    
    def generate_predictions(self, symbol: str, forecast_horizon: int = 5) -> Dict[str, Any]:
        """
        Generate future predictions using trained models
        
        Args:
            symbol: Stock/crypto symbol
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            logger.info(f"🔄 Generating {forecast_horizon}-day predictions for {symbol}")
            
            # Get latest data for neural models
            historical_data = self.db_manager.get_historical_data(symbol)
            processed_data = self.data_processor.prepare_features(historical_data)
            
            # Generate predictions from all traditional models
            traditional_predictions = self.traditional_models.predict_all_models(steps=forecast_horizon)
            
            # Generate predictions from all neural models
            neural_predictions = {}
            if not processed_data.empty:
                neural_predictions = self.neural_models.predict_all_models(processed_data, steps=forecast_horizon)
            
            # Combine all predictions
            all_predictions = {**traditional_predictions, **neural_predictions}
            
            if not all_predictions:
                logger.warning(f"⚠️ No predictions generated for {symbol}")
                return {}
            
            # Create prediction metadata
            prediction_data = {
                'symbol': symbol,
                'forecast_horizon': forecast_horizon,
                'prediction_date': datetime.now().isoformat(),
                'predictions': {},
                'model_info': {
                    **self.traditional_models.get_model_info(),
                    **self.neural_models.get_model_info()
                }
            }
            
            # Process predictions
            for model_name, pred_values in all_predictions.items():
                if len(pred_values) > 0:
                    # Convert numpy arrays to lists if needed
                    if hasattr(pred_values, 'tolist'):
                        values_list = pred_values.tolist()
                    else:
                        values_list = list(pred_values)
                    
                    prediction_data['predictions'][model_name] = {
                        'values': values_list,
                        'forecast_dates': [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                                         for i in range(len(values_list))]
                    }
            
            logger.info(f"✅ Predictions generated for {symbol}: {list(all_predictions.keys())}")
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"❌ Error generating predictions for {symbol}: {e}")
            return {}
    
    def save_results_to_database(self, symbol: str, predictions: Dict[str, Any], evaluation_results: Dict[str, pd.DataFrame]):
        """Save training results to database"""
        try:
            logger.info(f"🔄 Saving results to database for {symbol}")
            
            # Save predictions
            if predictions and 'predictions' in predictions:
                for model_name, pred_data in predictions['predictions'].items():
                    prediction_record = {
                        'symbol': symbol,
                        'model_name': model_name,
                        'model_type': 'traditional',
                        'prediction_date': predictions['prediction_date'],
                        'forecast_horizon': predictions['forecast_horizon'],
                        'predictions': pred_data['values'],
                        'forecast_dates': pred_data['forecast_dates'],
                        'metadata': predictions.get('model_info', {}).get(model_name, {})
                    }
                    
                    self.db_manager.save_prediction(prediction_record)
            
            # Save performance metrics
            if evaluation_results:
                for eval_type, results_df in evaluation_results.items():
                    for model_name in results_df.index:
                        performance_record = {
                            'symbol': symbol,
                            'model_name': model_name,
                            'model_type': 'traditional',
                            'evaluation_type': eval_type,  # 'train' or 'test'
                            'evaluation_date': datetime.now().isoformat(),
                            'metrics': results_df.loc[model_name].to_dict()
                        }
                        
                        self.db_manager.save_model_performance(performance_record)
            
            logger.info(f"✅ Results saved to database for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results to database for {symbol}: {e}")
    
    def train_single_instrument_with_model_types(self, symbol: str, model_types: List[str] = ['traditional', 'neural'], 
                                                train_size: float = 0.8, forecast_horizon: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Complete training pipeline for a single instrument with specific model types
        
        Args:
            symbol: Stock/crypto symbol
            model_types: List of model types to train ['traditional', 'neural']
            train_size: Proportion of data for training
            forecast_horizon: Number of days to forecast
            **kwargs: Additional training parameters
            
        Returns:
            Complete training results
        """
        try:
            logger.info(f"🚀 Starting training pipeline for {symbol} with models: {model_types}")
            
            # Step 1: Prepare data
            train_data, test_data = self.prepare_training_data(symbol, train_size)
            
            if train_data.empty or test_data.empty:
                logger.error(f"❌ Insufficient data for {symbol}")
                return {'success': False, 'error': 'Insufficient data'}
            
            training_results = {}
            
            # Step 2: Train selected model types
            if 'traditional' in model_types:
                traditional_results = self.train_traditional_models(symbol, train_data)
                training_results['traditional'] = traditional_results
            
            if 'neural' in model_types:
                neural_results = self.train_neural_models(symbol, **kwargs)
                training_results['neural'] = neural_results
            
            # Step 3: Evaluate models
            evaluation_results = self.evaluate_models(symbol, train_data, test_data)
            
            # Step 4: Generate predictions
            predictions = self.generate_predictions(symbol, forecast_horizon)
            
            # Step 5: Save to database
            self.save_results_to_database(symbol, predictions, evaluation_results)
            
            # Compile results
            pipeline_results = {
                'success': True,
                'symbol': symbol,
                'model_types_trained': model_types,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'predictions': predictions,
                'data_info': {
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_ratio': train_size
                },
                'completion_time': datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Training pipeline finished for {symbol} with models: {model_types}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_single_instrument(self, symbol: str, train_size: float = 0.8, forecast_horizon: int = 5) -> Dict[str, Any]:
        """
        Complete training pipeline for a single instrument
        
        Args:
            symbol: Stock/crypto symbol
            train_size: Proportion of data for training
            forecast_horizon: Number of days to forecast
            
        Returns:
            Complete training results
        """
        try:
            logger.info(f"🚀 Starting complete training pipeline for {symbol}")
            
            # Step 1: Prepare data
            train_data, test_data = self.prepare_training_data(symbol, train_size)
            
            if train_data.empty or test_data.empty:
                logger.error(f"❌ Insufficient data for {symbol}")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Step 2: Train traditional models
            traditional_training_results = self.train_traditional_models(symbol, train_data)
            
            # Step 3: Train neural models
            neural_training_results = self.train_neural_models(symbol, train_data)
            
            # Step 4: Evaluate models
            evaluation_results = self.evaluate_models(symbol, train_data, test_data)
            
            # Step 5: Generate predictions
            predictions = self.generate_predictions(symbol, forecast_horizon)
            
            # Step 6: Save to database
            self.save_results_to_database(symbol, predictions, evaluation_results)
            
            # Compile results
            pipeline_results = {
                'success': True,
                'symbol': symbol,
                'training_results': {
                    'traditional': traditional_training_results,
                    'neural': neural_training_results
                },
                'evaluation_results': evaluation_results,
                'predictions': predictions,
                'data_info': {
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_ratio': train_size
                },
                'completion_time': datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Complete training pipeline finished for {symbol}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_all_instruments(self, train_size: float = 0.8, forecast_horizon: int = 5) -> Dict[str, Any]:
        """
        Train models for all supported instruments
        
        Args:
            train_size: Proportion of data for training
            forecast_horizon: Number of days to forecast
            
        Returns:
            Complete training results for all instruments
        """
        try:
            logger.info("🚀 Starting training pipeline for ALL instruments")
            
            # Get supported instruments
            supported_instruments = self.db_manager.get_supported_instruments()
            
            if not supported_instruments:
                logger.error("❌ No supported instruments found")
                return {'success': False, 'error': 'No supported instruments'}
            
            all_results = {
                'success': True,
                'start_time': datetime.now().isoformat(),
                'instruments': {},
                'summary': {
                    'total_instruments': len(supported_instruments),
                    'successful_instruments': 0,
                    'failed_instruments': 0,
                    'total_models_trained': 0,
                    'successful_models': 0
                }
            }
            
            # Train each instrument
            for instrument in supported_instruments:
                symbol = instrument['symbol']
                logger.info(f"🔄 Processing instrument {symbol}")
                
                try:
                    # Train single instrument
                    instrument_results = self.train_single_instrument(symbol, train_size, forecast_horizon)
                    
                    all_results['instruments'][symbol] = instrument_results
                    
                    if instrument_results.get('success', False):
                        all_results['summary']['successful_instruments'] += 1
                        
                        # Count successful models
                        training_results = instrument_results.get('training_results', {})
                        successful_models = sum(1 for success in training_results.values() if success)
                        all_results['summary']['successful_models'] += successful_models
                        all_results['summary']['total_models_trained'] += len(training_results)
                        
                    else:
                        all_results['summary']['failed_instruments'] += 1
                    
                    logger.info(f"✅ Completed processing {symbol}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {symbol}: {e}")
                    all_results['instruments'][symbol] = {'success': False, 'error': str(e)}
                    all_results['summary']['failed_instruments'] += 1
            
            all_results['completion_time'] = datetime.now().isoformat()
            
            # Log summary
            summary = all_results['summary']
            logger.info(f"🎉 Training pipeline complete!")
            logger.info(f"   Instruments: {summary['successful_instruments']}/{summary['total_instruments']} successful")
            logger.info(f"   Models: {summary['successful_models']}/{summary['total_models_trained']} successful")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in train_all_instruments: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_ensemble_predictions(self, symbol: str, forecast_horizon: int = 5, 
                                    strategy: str = 'performance_weighted') -> Dict[str, Any]:
        """
        Generate ensemble predictions combining traditional and neural models
        
        Args:
            symbol: Stock/crypto symbol
            forecast_horizon: Number of steps to forecast
            strategy: Ensemble strategy to use
            
        Returns:
            Dictionary containing ensemble predictions and metadata
        """
        try:
            logger.info(f"🔄 Generating ensemble predictions for {symbol} using {strategy}")
            
            # Collect predictions from all available models
            model_predictions = {}
            
            # Get predictions from data processor
            processed_data = self.data_processor.get_processed_data(symbol, limit=100)
            if processed_data.empty:
                raise ValueError(f"No processed data available for {symbol}")
            
            # Traditional model predictions
            try:
                traditional_predictions = self.traditional_models.predict_all_models(
                    processed_data, 
                    symbol=symbol, 
                    steps=forecast_horizon
                )
                model_predictions.update(traditional_predictions)
                logger.info(f"✅ Got traditional predictions: {list(traditional_predictions.keys())}")
            except Exception as e:
                logger.warning(f"⚠️ Traditional models prediction failed: {e}")
            
            # Neural model predictions
            try:
                neural_predictions = self.neural_models.predict_all_models(
                    processed_data, 
                    symbol=symbol, 
                    forecast_horizon=forecast_horizon
                )
                model_predictions.update(neural_predictions)
                logger.info(f"✅ Got neural predictions: {list(neural_predictions.keys())}")
            except Exception as e:
                logger.warning(f"⚠️ Neural models prediction failed: {e}")
            
            if not model_predictions:
                raise ValueError("No model predictions available for ensemble")
            
            # Update model performance data in ensemble forecaster
            for model_name in model_predictions.keys():
                if model_name in self.model_performance:
                    self.ensemble_forecaster.update_model_performance(
                        model_name, 
                        self.model_performance[model_name]
                    )
            
            # Generate ensemble prediction
            ensemble_result = self.ensemble_forecaster.generate_ensemble_prediction(
                model_predictions, 
                strategy=strategy
            )
            
            # Add forecast dates
            forecast_dates = [
                (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(forecast_horizon)
            ]
            ensemble_result['forecast_dates'] = forecast_dates
            ensemble_result['symbol'] = symbol
            
            logger.info(f"✅ Ensemble prediction generated for {symbol}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"❌ Error generating ensemble predictions for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def compare_all_models(self, symbol: str, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Compare performance of all models (traditional, neural, ensemble) for a symbol
        
        Args:
            symbol: Stock/crypto symbol
            test_data: Test dataset for evaluation
            
        Returns:
            Dictionary containing model comparison results
        """
        try:
            logger.info(f"🔄 Comparing all models for {symbol}")
            
            if test_data is None:
                # Get recent data for comparison
                processed_data = self.data_processor.get_processed_data(symbol, limit=100)
                if processed_data.empty:
                    raise ValueError(f"No processed data available for {symbol}")
                
                # Use last 20% as test data
                split_idx = int(len(processed_data) * 0.8)
                test_data = processed_data.iloc[split_idx:]
            
            actual_values = test_data['Close'].tolist()
            
            # Get predictions from all models for comparison
            forecast_horizon = len(actual_values)
            
            # Traditional models
            traditional_predictions = {}
            try:
                traditional_predictions = self.traditional_models.predict_all_models(
                    test_data, 
                    symbol=symbol, 
                    steps=forecast_horizon
                )
            except Exception as e:
                logger.warning(f"Traditional models comparison failed: {e}")
            
            # Neural models
            neural_predictions = {}
            try:
                neural_predictions = self.neural_models.predict_all_models(
                    test_data, 
                    symbol=symbol, 
                    forecast_horizon=forecast_horizon
                )
            except Exception as e:
                logger.warning(f"Neural models comparison failed: {e}")
            
            # Ensemble predictions
            ensemble_predictions = {}
            all_predictions = {**traditional_predictions, **neural_predictions}
            
            if all_predictions:
                for strategy in self.ensemble_forecaster.ensemble_strategies:
                    try:
                        ensemble_result = self.ensemble_forecaster.generate_ensemble_prediction(
                            all_predictions, 
                            strategy=strategy
                        )
                        ensemble_predictions[f'ensemble_{strategy}'] = ensemble_result['ensemble_prediction']
                    except Exception as e:
                        logger.warning(f"Ensemble strategy {strategy} failed: {e}")
            
            # Combine all predictions
            all_model_predictions = {**traditional_predictions, **neural_predictions, **ensemble_predictions}
            
            # Add results to model comparison
            for model_name, predictions in all_model_predictions.items():
                try:
                    # Calculate metrics
                    metrics = self.evaluator.calculate_metrics(actual_values, predictions)
                    self.model_comparison.add_model_results(
                        model_name, 
                        predictions, 
                        actual_values, 
                        metrics
                    )
                except Exception as e:
                    logger.warning(f"Error adding {model_name} to comparison: {e}")
            
            # Generate comparison report
            comparison_report = self.model_comparison.generate_comparison_report()
            comparison_report['symbol'] = symbol
            comparison_report['test_data_size'] = len(actual_values)
            
            logger.info(f"✅ Model comparison completed for {symbol}")
            return comparison_report
            
        except Exception as e:
            logger.error(f"❌ Error comparing models for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def optimize_ensemble_weights(self, symbol: str, validation_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Optimize ensemble weights based on historical performance
        
        Args:
            symbol: Stock/crypto symbol
            validation_data: Validation dataset for optimization
            
        Returns:
            Dictionary containing optimized weights and performance
        """
        try:
            logger.info(f"🔄 Optimizing ensemble weights for {symbol}")
            
            if validation_data is None:
                # Get recent data for validation
                processed_data = self.data_processor.get_processed_data(symbol, limit=150)
                if processed_data.empty:
                    raise ValueError(f"No processed data available for {symbol}")
                
                # Use middle portion as validation data
                start_idx = int(len(processed_data) * 0.6)
                end_idx = int(len(processed_data) * 0.8)
                validation_data = processed_data.iloc[start_idx:end_idx]
            
            actual_values = validation_data['Close'].tolist()
            forecast_horizon = len(actual_values)
            
            # Get predictions from all models
            all_predictions = {}
            
            try:
                traditional_predictions = self.traditional_models.predict_all_models(
                    validation_data, 
                    symbol=symbol, 
                    steps=forecast_horizon
                )
                all_predictions.update(traditional_predictions)
            except Exception as e:
                logger.warning(f"Traditional models failed in optimization: {e}")
            
            try:
                neural_predictions = self.neural_models.predict_all_models(
                    validation_data, 
                    symbol=symbol, 
                    forecast_horizon=forecast_horizon
                )
                all_predictions.update(neural_predictions)
            except Exception as e:
                logger.warning(f"Neural models failed in optimization: {e}")
            
            if not all_predictions:
                raise ValueError("No predictions available for weight optimization")
            
            # Test different ensemble strategies
            strategy_performance = {}
            
            for strategy in self.ensemble_forecaster.ensemble_strategies:
                try:
                    ensemble_result = self.ensemble_forecaster.generate_ensemble_prediction(
                        all_predictions, 
                        strategy=strategy
                    )
                    ensemble_pred = ensemble_result['ensemble_prediction']
                    
                    # Calculate performance metrics
                    metrics = self.evaluator.calculate_metrics(actual_values, ensemble_pred)
                    strategy_performance[strategy] = metrics
                    
                except Exception as e:
                    logger.warning(f"Strategy {strategy} failed in optimization: {e}")
            
            # Find best strategy
            best_strategy = None
            best_mape = float('inf')
            
            for strategy, metrics in strategy_performance.items():
                if metrics['mape'] < best_mape:
                    best_mape = metrics['mape']
                    best_strategy = strategy
            
            optimization_result = {
                'symbol': symbol,
                'best_strategy': best_strategy,
                'best_performance': strategy_performance.get(best_strategy, {}),
                'all_strategy_performance': strategy_performance,
                'validation_data_size': len(actual_values),
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Ensemble optimization completed for {symbol}. Best: {best_strategy}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Error optimizing ensemble weights for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
    
    def train_lightgbm_model(self, symbol: str, optimize: bool = True, 
                            cross_validate: bool = True) -> Dict[str, Any]:
        """
        Train LightGBM model for a symbol
        
        Args:
            symbol: Stock/crypto symbol
            optimize: Whether to optimize hyperparameters
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info(f"🚀 Training LightGBM model for {symbol}")
            
            # Get historical data
            historical_data = self.db_manager.get_historical_data(symbol)
            
            if historical_data.empty or len(historical_data) < 100:
                logger.error(f"❌ Insufficient data for {symbol}: {len(historical_data)} samples")
                return {
                    'success': False,
                    'error': f'Insufficient training data: {len(historical_data)} samples (need ≥100)'
                }
            
            # Prepare features
            processed_data = self.data_processor.prepare_features(historical_data)
            
            if processed_data.empty:
                logger.error(f"❌ Failed to prepare features for {symbol}")
                return {
                    'success': False,
                    'error': 'Feature preparation failed'
                }
            
            # Train the model
            training_result = self.lightgbm_forecaster.fit(
                data=processed_data,
                target_column='Close',
                optimize=optimize,
                test_size=0.2,
                lookback=5,
                num_boost_round=500,
                symbol=symbol  # Pass symbol for MLflow tracking
            )
            
            if not training_result.get('success', False):
                return training_result
            
            # Perform cross-validation if requested
            cv_results = {}
            if cross_validate:
                logger.info(f"🔄 Performing cross-validation for {symbol}...")
                cv_results = self.lightgbm_forecaster.cross_validate(
                    data=processed_data,
                    target_column='Close',
                    n_splits=5,
                    lookback=5
                )
            
            # Generate predictions for the next 7 days
            prediction_result = self.lightgbm_forecaster.predict(
                data=processed_data,
                target_column='Close',
                forecast_horizon=7,
                lookback=5
            )
            
            # Save the model
            save_success = self.lightgbm_forecaster.save_model(symbol, 'lightgbm')
            
            # Get feature importance
            feature_importance = self.lightgbm_forecaster.get_feature_importance(top_n=20)
            
            # Store results
            if symbol not in self.training_results:
                self.training_results[symbol] = {}
            
            self.training_results[symbol]['lightgbm'] = {
                'training_result': training_result,
                'cross_validation': cv_results,
                'predictions': prediction_result.get('predictions', []),
                'feature_importance': feature_importance,
                'model_saved': save_success,
                'training_date': datetime.now().isoformat()
            }
            
            # Save predictions to database
            if prediction_result.get('success', False):
                self.save_results_to_database(
                    symbol=symbol,
                    predictions={'lightgbm': prediction_result.get('predictions', [])},
                    evaluation_results={'lightgbm': training_result.get('metrics', {})}
                )
            
            logger.info(f"✅ LightGBM training complete for {symbol}")
            logger.info(f"   Top 5 features: {list(feature_importance.keys())[:5]}")
            
            return {
                'success': True,
                'symbol': symbol,
                'metrics': training_result.get('metrics', {}),
                'cross_validation': cv_results,
                'predictions': prediction_result.get('predictions', []),
                'feature_importance': feature_importance,
                'model_saved': save_success
            }
            
        except Exception as e:
            logger.error(f"❌ Error training LightGBM for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def train_all_models(self, symbol: str, include_lightgbm: bool = True,
                        include_neural: bool = False) -> Dict[str, Any]:
        """
        Train all available models for a symbol with MLflow tracking
        
        Args:
            symbol: Stock/crypto symbol
            include_lightgbm: Whether to include LightGBM training
            include_neural: Whether to include neural network models
            
        Returns:
            Comprehensive training results
        """
        training_start_time = time.time()
        
        # Start parent MLflow run for overall training
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment("stock-forecasting-pipeline")
                mlflow.start_run(run_name=f"Pipeline_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Log overall parameters
                mlflow.log_param("symbol", symbol)
                mlflow.log_param("include_lightgbm", include_lightgbm)
                mlflow.log_param("include_neural", include_neural)
                mlflow.log_param("training_date", datetime.now().strftime('%Y-%m-%d'))
            except Exception as e:
                logger.warning(f"⚠️ MLflow pipeline logging initialization failed: {e}")
        
        try:
            logger.info(f"🎯 Starting comprehensive model training for {symbol}")
            
            all_results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'models_trained': []
            }
            
            # Prepare data
            train_data, test_data = self.prepare_training_data(symbol, train_size=0.8)
            
            if train_data.empty or test_data.empty:
                logger.error(f"❌ Failed to prepare training data for {symbol}")
                return {
                    'success': False,
                    'error': 'Data preparation failed'
                }
            
            # Log data info to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("train_data_size", len(train_data))
                    mlflow.log_param("test_data_size", len(test_data))
                except:
                    pass
            
            # 1. Train Traditional Models
            logger.info("📊 Training traditional models...")
            traditional_results = self.train_traditional_models(symbol, train_data)
            all_results['traditional'] = traditional_results
            if traditional_results:
                all_results['models_trained'].append('traditional')
            
            # 2. Train LightGBM Model (NEW - Phase 1)
            if include_lightgbm:
                logger.info("🌟 Training LightGBM model (Gradient Boosting)...")
                lightgbm_results = self.train_lightgbm_model(symbol, optimize=True, cross_validate=True)
                all_results['lightgbm'] = lightgbm_results
                if lightgbm_results.get('success', False):
                    all_results['models_trained'].append('lightgbm')
            
            # 3. Train Neural Models (if requested)
            if include_neural:
                logger.info("🤖 Training neural network models...")
                neural_results = self.train_neural_models(symbol, model_types=['lstm', 'gru'])
                all_results['neural'] = neural_results
                if neural_results.get('success', False):
                    all_results['models_trained'].append('neural')
            
            # 4. Evaluate all models
            logger.info("📊 Evaluating model performance...")
            evaluation_results = self.evaluate_models(symbol, train_data, test_data)
            all_results['evaluation'] = evaluation_results
            
            all_results['success'] = len(all_results['models_trained']) > 0
            
            # Calculate total training time
            total_training_time = time.time() - training_start_time
            all_results['total_training_time'] = total_training_time
            
            # Log pipeline summary metrics to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metric("num_models_trained", len(all_results['models_trained']))
                    mlflow.log_metric("total_training_time_seconds", total_training_time)
                    mlflow.log_param("models_trained", str(all_results['models_trained']))
                    mlflow.log_metric("pipeline_success", 1 if all_results['success'] else 0)
                    
                    logger.info("✅ MLflow tracking: Pipeline summary logged successfully")
                except Exception as e:
                    logger.warning(f"⚠️ MLflow pipeline summary logging failed: {e}")
            
            logger.info(f"✅ Complete training finished for {symbol}")
            logger.info(f"   Models trained: {all_results['models_trained']}")
            logger.info(f"   Total time: {total_training_time:.2f}s")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive training for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
            # Log error to MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_param("error", str(e))
                    mlflow.log_metric("pipeline_success", 0)
                except:
                    pass
            
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
        finally:
            # End parent MLflow run
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.end_run()
                except:
                    pass
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training activities"""
        try:
            # Get recent predictions and performance from database
            recent_predictions = []
            recent_performance = []
            
            # Summary data
            summary = {
                'training_sessions': len(self.training_results),
                'instruments_trained': list(self.training_results.keys()),
                'performance_evaluations': len(self.model_performance),
                'recent_predictions': recent_predictions[:10],  # Last 10
                'recent_performance': recent_performance[:10],  # Last 10
                'model_types': {
                    'traditional': ['SMA', 'EMA', 'ARIMA', 'VAR'],
                    'neural': []  # Will be added in Phase 4
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting training summary: {e}")
            return {}

# Singleton instance
_model_trainer = None

def get_model_trainer() -> ModelTrainer:
    """Get model trainer singleton instance"""
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = ModelTrainer()
    return _model_trainer