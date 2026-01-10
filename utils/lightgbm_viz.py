"""
Visualization Functions for LightGBM Model Analysis

"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

# Try importing plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Some visualizations will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class LightGBMVisualizer:
    """
    Visualization utilities for LightGBM model analysis
    """
    
    def __init__(self, output_dir: str = 'static/visualizations/'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"✅ LightGBM Visualizer initialized - outputs to {output_dir}")
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                               top_n: int = 20, save_path: str = None,
                               symbol: str = None) -> str:
        """
        Plot feature importance as horizontal bar chart
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            top_n: Number of top features to display
            save_path: Path to save the plot
            symbol: Stock symbol for title
            
        Returns:
            Path to saved plot
        """
        try:
            logger.info(f"📊 Creating feature importance plot (top {top_n})...")
            
            # Get top N features
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:top_n]
            
            features, importances = zip(*sorted_features)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
            
            bars = ax.barh(y_pos, importances, color=colors, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Highest importance on top
            ax.set_xlabel('Importance Score (Gain)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Features', fontsize=12, fontweight='bold')
            
            title = f'LightGBM Feature Importance - Top {top_n} Features'
            if symbol:
                title += f' ({symbol})'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{imp:.0f}',
                       ha='left', va='center', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Grid
            ax.grid(True, axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.output_dir, 
                                        f'lightgbm_feature_importance_{symbol or "model"}_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Feature importance plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error creating feature importance plot: {e}")
            return ""
    
    def plot_training_history(self, training_history: Dict[str, Any],
                             save_path: str = None, symbol: str = None) -> str:
        """
        Plot training and validation loss over iterations
        
        Args:
            training_history: Training history from LightGBM
            save_path: Path to save the plot
            symbol: Stock symbol for title
            
        Returns:
            Path to saved plot
        """
        try:
            logger.info("📈 Creating training history plot...")
            
            evals_result = training_history.get('evals_result', {})
            
            if not evals_result:
                logger.warning("⚠️ No evaluation results available")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot training and validation metrics
            for dataset_name, metrics in evals_result.items():
                for metric_name, values in metrics.items():
                    label = f'{dataset_name} - {metric_name}'
                    ax.plot(values, label=label, linewidth=2)
            
            # Mark best iteration
            best_iter = training_history.get('best_iteration', 0)
            ax.axvline(x=best_iter, color='red', linestyle='--', linewidth=2,
                      label=f'Best Iteration ({best_iter})')
            
            # Customize plot
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
            
            title = 'LightGBM Training History'
            if symbol:
                title += f' - {symbol}'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.output_dir,
                                        f'lightgbm_training_history_{symbol or "model"}_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Training history plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error creating training history plot: {e}")
            return ""
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   save_path: str = None, symbol: str = None) -> str:
        """
        Plot predicted vs actual values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save the plot
            symbol: Stock symbol for title
            
        Returns:
            Path to saved plot
        """
        try:
            logger.info("📊 Creating predictions vs actual plot...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Scatter plot
            ax1.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
            
            ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            
            title = 'LightGBM: Predicted vs Actual'
            if symbol:
                title += f' - {symbol}'
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax2.set_title('Residuals Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.output_dir,
                                        f'lightgbm_predictions_{symbol or "model"}_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Predictions plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error creating predictions plot: {e}")
            return ""
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             save_path: str = None, symbol: str = None) -> str:
        """
        Plot comparison of different models
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            save_path: Path to save the plot
            symbol: Stock symbol for title
            
        Returns:
            Path to saved plot
        """
        try:
            logger.info("📊 Creating model comparison plot...")
            
            if comparison_df.empty:
                logger.warning("⚠️ Empty comparison dataframe")
                return ""
            
            # Select metrics to plot
            metrics = ['RMSE', 'MAE', 'MAPE (%)']
            available_metrics = [m for m in metrics if m in comparison_df.columns]
            
            if not available_metrics:
                logger.warning("⚠️ No metrics available for plotting")
                return ""
            
            # Create subplots
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for ax, metric in zip(axes, available_metrics):
                models = comparison_df['Model']
                values = comparison_df[metric]
                
                # Create bar chart
                colors = ['#2ecc71' if model == 'LIGHTGBM' else '#3498db' 
                         for model in models]
                bars = ax.bar(range(len(models)), values, color=colors, 
                            edgecolor='black', linewidth=1.5)
                
                # Highlight LightGBM
                for i, model in enumerate(models):
                    if model == 'LIGHTGBM':
                        bars[i].set_edgecolor('gold')
                        bars[i].set_linewidth(3)
                
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylabel(metric, fontsize=11, fontweight='bold')
                ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
                ax.grid(True, axis='y', alpha=0.3)
                ax.set_axisbelow(True)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Main title
            title = 'Model Performance Comparison'
            if symbol:
                title += f' - {symbol}'
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(self.output_dir,
                                        f'model_comparison_{symbol or "all"}_{timestamp}.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Model comparison plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error creating model comparison plot: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def create_interactive_feature_importance(self, feature_importance: Dict[str, float],
                                             top_n: int = 20, symbol: str = None) -> str:
        """
        Create interactive feature importance plot using Plotly
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            top_n: Number of top features to display
            symbol: Stock symbol for title
            
        Returns:
            HTML string with interactive plot
        """
        try:
            if not PLOTLY_AVAILABLE:
                logger.warning("⚠️ Plotly not available for interactive plots")
                return ""
            
            logger.info(f"🎨 Creating interactive feature importance plot...")
            
            # Get top N features
            sorted_features = sorted(feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)[:top_n]
            
            features, importances = zip(*sorted_features)
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f'{imp:.0f}' for imp in importances],
                textposition='auto',
            ))
            
            title = f'LightGBM Feature Importance - Top {top_n} Features'
            if symbol:
                title += f' ({symbol})'
            
            fig.update_layout(
                title=title,
                xaxis_title='Importance Score (Gain)',
                yaxis_title='Features',
                height=max(600, top_n * 30),
                template='plotly_white',
                yaxis={'categoryorder': 'total ascending'}
            )
            
            # Save to HTML
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir,
                                    f'interactive_feature_importance_{symbol or "model"}_{timestamp}.html')
            
            fig.write_html(save_path)
            
            logger.info(f"✅ Interactive plot saved to {save_path}")
            return fig.to_html()
            
        except Exception as e:
            logger.error(f"❌ Error creating interactive plot: {e}")
            return ""


# Singleton instance
_lightgbm_visualizer_instance = None

def get_lightgbm_visualizer(output_dir: str = 'static/visualizations/') -> LightGBMVisualizer:
    """Get singleton instance of LightGBMVisualizer"""
    global _lightgbm_visualizer_instance
    if _lightgbm_visualizer_instance is None:
        _lightgbm_visualizer_instance = LightGBMVisualizer(output_dir)
    return _lightgbm_visualizer_instance


if __name__ == "__main__":
    # Demo usage
    logger.info("LightGBM Visualizer Module - Demo")
    visualizer = get_lightgbm_visualizer()
    logger.info("✅ Module ready for use")
