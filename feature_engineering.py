"""
Feature engineering module for stock prediction.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

# Logging configuration removed - using main module's logging setup
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for creating technical indicators and features for stock prediction."""
    
    def __init__(self, lookback_days: int = 14):
        """
        Initialize the feature engineer.
        
        Args:
            lookback_days (int): Number of days to look back for features
        """
        self.lookback_days = lookback_days
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the stock data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df['close'].ewm(span=5).mean()
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['close'])
        
        # MACD (Moving Average Convergence Divergence)
        macd_line, macd_signal = self._calculate_macd(df['close'])
        df['MACD'] = macd_line
        df['MACD_signal'] = macd_signal
        df['MACD_histogram'] = macd_line - macd_signal
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        df['BB_width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Price_change'] = df['close'].pct_change()
        df['Price_change_5d'] = df['close'].pct_change(periods=5)
        df['Price_change_10d'] = df['close'].pct_change(periods=10)
        
        # Volatility indicators
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        df['High_Low_ratio'] = df['high'] / df['low']
        df['Close_Open_ratio'] = df['close'] / df['open']
        
        # Momentum indicators
        df['Momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['Momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        
        logger.info("Technical indicators added successfully")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                 period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def create_lag_features(self, data: pd.DataFrame, 
                          target_column: str = 'close',
                          lag_days: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series prediction.
        
        Args:
            data (pd.DataFrame): Stock data
            target_column (str): Column to create lags for
            lag_days (List[int]): List of lag days to create
        
        Returns:
            pd.DataFrame: Data with lag features
        """
        if lag_days is None:
            lag_days = [1, 2, 3, 5, 10]
        
        df = data.copy()
        
        for lag in lag_days:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
            df[f'{target_column}_pct_change_lag_{lag}'] = df[target_column].pct_change(lag)
        
        logger.info(f"Created lag features for {len(lag_days)} periods")
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, 
                              target_column: str = 'close',
                              windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            data (pd.DataFrame): Stock data
            target_column (str): Column to create rolling features for
            windows (List[int]): List of window sizes
        
        Returns:
            pd.DataFrame: Data with rolling features
        """
        if windows is None:
            windows = [5, 10, 20]
        
        df = data.copy()
        
        for window in windows:
            df[f'{target_column}_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_std_{window}'] = df[target_column].rolling(window=window).std()
            df[f'{target_column}_min_{window}'] = df[target_column].rolling(window=window).min()
            df[f'{target_column}_max_{window}'] = df[target_column].rolling(window=window).max()
            df[f'{target_column}_skew_{window}'] = df[target_column].rolling(window=window).skew()
            df[f'{target_column}_kurt_{window}'] = df[target_column].rolling(window=window).kurt()
        
        logger.info(f"Created rolling features for {len(windows)} window sizes")
        return df
    
    def prepare_features(self, data: pd.DataFrame, 
                        target_column: str = 'close',
                        prediction_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            target_column (str): Target column for prediction
            prediction_days (int): Number of days ahead to predict
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Create target variable (future price)
        data = data.copy()
        data['target'] = data[target_column].shift(-prediction_days)
        
        # Select feature columns (exclude non-numeric and target columns)
        exclude_columns = ['symbol', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Create feature matrix
        features = data[feature_columns].copy()
        target = data['target'].copy()
        
        # Remove rows with NaN values
        valid_indices = ~(features.isnull().any(axis=1) | target.isnull())
        features = features[valid_indices]
        target = target[valid_indices]
        
        logger.info(f"Prepared {len(feature_columns)} features for {len(features)} samples")
        return features, target
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            data (pd.DataFrame): Raw stock data
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        df = data.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by date
        df = df.sort_index()
        
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill
        df = df.fillna(method='bfill')  # Backward fill for remaining NaN
        
        # Remove any remaining NaN values
        df = df.dropna()
        
        logger.info(f"Data cleaned: {len(df)} records remaining")
        return df
