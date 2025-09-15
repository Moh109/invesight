"""
Visualization module for stock prediction analysis.
"""
import pandas as pd
import numpy as np
# matplotlib and seaborn imports removed - only used for style settings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Logging configuration removed - using main module's logging setup
logger = logging.getLogger(__name__)

# Style settings removed - using plotly templates instead

class StockVisualizer:
    """Class for creating various visualizations for stock analysis."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = {
            'price': '#1f77b4',
            'prediction': '#ff7f0e',
            'volume': '#2ca02c',
            'trend': '#d62728',
            'support': '#9467bd',
            'resistance': '#8c564b'
        }
    
    def plot_price_data(self, data: pd.DataFrame, symbol: str, 
                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive candlestick chart with volume.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            symbol (str): Stock symbol
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Stock Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.8)'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Price chart saved to {save_path}")
        
        return fig
    
    def plot_technical_indicators(self, data: pd.DataFrame, symbol: str,
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Plot technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} Price with Moving Averages',
                'RSI (Relative Strength Index)',
                'MACD',
                'Bollinger Bands'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], name='Close Price', line=dict(color='black')),
            row=1, col=1
        )
        
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='blue')),
                row=1, col=1
            )
        
        if 'EMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='red')),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal', line=dict(color='red')),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=data.index, y=data['MACD'] - data['MACD_signal'], name='MACD Histogram'),
                row=3, col=1
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['close'], name='Close Price', line=dict(color='black')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_middle'], name='BB Middle', line=dict(color='blue')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Technical indicators chart saved to {save_path}")
        
        return fig
    
    def plot_predictions(self, actual: pd.Series, predictions: np.ndarray, 
                        symbol: str, model_name: str,
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            actual (pd.Series): Actual values
            predictions (np.ndarray): Predicted values
            symbol (str): Stock symbol
            model_name (str): Name of the model used
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Create date range for predictions
        dates = actual.index[-len(predictions):]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual[-len(predictions):],
            mode='lines',
            name='Actual',
            line=dict(color=self.colors['price'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name=f'Predicted ({model_name})',
            line=dict(color=self.colors['prediction'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{symbol} Price Prediction - {model_name}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Predictions chart saved to {save_path}")
        
        return fig
    
    def plot_model_performance(self, performance: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None) -> go.Figure:
        """
        Plot model performance comparison.
        
        Args:
            performance (Dict[str, Dict[str, float]]): Model performance metrics
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        models = list(performance.keys())
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                          'R² (Higher is Better)', 'MAPE (Lower is Better)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, metric in enumerate(metrics):
            values = [performance[model][metric] for model in models]
            row, col = positions[i]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.upper(),
                    marker_color=px.colors.qualitative.Set3[i]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Model performance chart saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance: pd.DataFrame, 
                              model_name: str, top_n: int = 15,
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Plot feature importance.
        
        Args:
            importance (pd.DataFrame): Feature importance data
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Get top N features
        top_features = importance.head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker_color=px.colors.qualitative.Set3[0]
            )
        ])
        
        fig.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=500,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Feature importance chart saved to {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, 
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Plot correlation heatmap of features.
        
        Args:
            data (pd.DataFrame): Data with features
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        return fig
    
    def create_dashboard(self, data: pd.DataFrame, predictions: np.ndarray,
                        actual: pd.Series, symbol: str, model_name: str,
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            data (pd.DataFrame): Stock data
            predictions (np.ndarray): Predictions
            actual (pd.Series): Actual values
            symbol (str): Stock symbol
            model_name (str): Model name
            save_path (Optional[str]): Path to save the plot
        
        Returns:
            go.Figure: Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'{symbol} Price Chart',
                'Volume',
                'RSI',
                'MACD',
                'Bollinger Bands',
                'Predictions vs Actual'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price chart with volume
        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], name='Close Price', line=dict(color='black')),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume', opacity=0.3),
            row=1, col=1, secondary_y=True
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume'),
            row=1, col=2
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal'),
                row=2, col=2
            )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['close'], name='Close'),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(dash='dash')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_middle'], name='BB Middle'),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(dash='dash')),
                row=3, col=1
            )
        
        # Predictions
        dates = actual.index[-len(predictions):]
        fig.add_trace(
            go.Scatter(x=dates, y=actual[-len(predictions):], name='Actual'),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=predictions, name='Predicted'),
            row=3, col=2
        )
        
        fig.update_layout(
            title=f'{symbol} Stock Analysis Dashboard',
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
