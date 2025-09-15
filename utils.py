"""
Utility functions for the stock prediction project.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
import json

# Logging configuration removed - using main module's logging setup
logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating stock data."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """
        Validate that data contains required OHLCV columns.
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
        
        # Check for negative values in price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.error(f"Negative or zero values found in {col} column")
                return False
        
        # Check for negative volume
        if (data['volume'] < 0).any():
            logger.error("Negative volume values found")
            return False
        
        # Check for high-low consistency
        if not (data['high'] >= data['low']).all():
            logger.error("High values are not consistently >= Low values")
            return False
        
        # Check for open/close within high/low range
        if not ((data['open'] >= data['low']) & (data['open'] <= data['high'])).all():
            logger.error("Open values are not within high-low range")
            return False
        
        if not ((data['close'] >= data['low']) & (data['close'] <= data['high'])).all():
            logger.error("Close values are not within high-low range")
            return False
        
        logger.info("Data validation passed")
        return True
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality and return statistics.
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            Dict[str, Any]: Data quality statistics
        """
        quality_stats = {
            'total_records': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_records': data.duplicated().sum(),
            'date_range': {
                'start': data.index.min() if not data.empty else None,
                'end': data.index.max() if not data.empty else None
            },
            'price_range': {
                'min_close': data['close'].min() if 'close' in data.columns else None,
                'max_close': data['close'].max() if 'close' in data.columns else None,
                'avg_close': data['close'].mean() if 'close' in data.columns else None
            },
            'volume_stats': {
                'min_volume': data['volume'].min() if 'volume' in data.columns else None,
                'max_volume': data['volume'].max() if 'volume' in data.columns else None,
                'avg_volume': data['volume'].mean() if 'volume' in data.columns else None
            }
        }
        
        return quality_stats

class PerformanceCalculator:
    """Class for calculating various performance metrics."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate simple returns.
        
        Args:
            prices (pd.Series): Price series
        
        Returns:
            pd.Series: Returns series
        """
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate log returns.
        
        Args:
            prices (pd.Series): Price series
        
        Returns:
            pd.Series: Log returns series
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns (pd.Series): Returns series
            window (int): Rolling window size
        
        Returns:
            pd.Series: Volatility series
        """
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pd.Series): Returns series
            risk_free_rate (float): Risk-free rate
        
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices (pd.Series): Price series
        
        Returns:
            float: Maximum drawdown percentage
        """
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100

class DataProcessor:
    """Class for data processing utilities."""
    
    @staticmethod
    def resample_data(data: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            data (pd.DataFrame): Stock data
            frequency (str): Target frequency ('D', 'W', 'M', etc.)
        
        Returns:
            pd.DataFrame: Resampled data
        """
        resampled = data.resample(frequency).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Data resampled to {frequency} frequency: {len(resampled)} records")
        return resampled
    
    @staticmethod
    def remove_outliers(data: pd.DataFrame, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            data (pd.DataFrame): Stock data
            method (str): Method to use ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        df = data.copy()
        
        if method == 'iqr':
            for column in ['open', 'high', 'low', 'close']:
                if column in df.columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        elif method == 'zscore':
            for column in ['open', 'high', 'low', 'close']:
                if column in df.columns:
                    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                    df = df[z_scores < threshold]
        
        logger.info(f"Outliers removed: {len(data) - len(df)} records")
        return df
    
    @staticmethod
    def create_train_test_split(data: pd.DataFrame, test_size: float = 0.2,
                              date_split: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train-test split for time series data.
        
        Args:
            data (pd.DataFrame): Stock data
            test_size (float): Proportion of data for testing
            date_split (Optional[str]): Specific date to split on
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test data
        """
        if date_split:
            split_date = pd.to_datetime(date_split)
            train_data = data[data.index < split_date]
            test_data = data[data.index >= split_date]
        else:
            split_index = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:]
        
        logger.info(f"Train data: {len(train_data)} records, Test data: {len(test_data)} records")
        return train_data, test_data

class FileManager:
    """Class for file management utilities."""
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> None:
        """
        Save data to JSON file.
        
        Args:
            data (Dict[str, Any]): Data to save
            filepath (str): Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            filepath (str): Path to load file from
        
        Returns:
            Dict[str, Any]: Loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    
    @staticmethod
    def ensure_directory_exists(directory: str) -> None:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory (str): Directory path
        """
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

class ModelEvaluator:
    """Class for model evaluation utilities."""
    
    @staticmethod
    def calculate_prediction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate various accuracy metrics.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
        
        Returns:
            Dict[str, float]: Accuracy metrics
        """
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    @staticmethod
    def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
        
        Returns:
            float: Directional accuracy percentage
        """
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        correct_directions = np.sum(actual_direction == predicted_direction)
        total_predictions = len(actual_direction)
        
        return (correct_directions / total_predictions) * 100
