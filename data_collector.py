"""
Data collection module for fetching stock data from various sources.
"""
import yfinance as yf
import pandas as pd
import requests
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

# Logging configuration removed - using main module's logging setup
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Class for collecting stock data from various sources."""
    
    def __init__(self):
        """Initialize the data collector."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_yahoo_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            logger.info(f"Fetching data for {symbol} from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def get_alpha_vantage_data(self, symbol: str, api_key: str, 
                              function: str = "TIME_SERIES_DAILY") -> pd.DataFrame:
        """
        Fetch stock data from Alpha Vantage API.
        
        Args:
            symbol (str): Stock symbol
            api_key (str): Alpha Vantage API key
            function (str): API function type
        
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            if not api_key:
                raise ValueError("Alpha Vantage API key is required")
            
            logger.info(f"Fetching data for {symbol} from Alpha Vantage...")
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            # Extract time series data
            time_series_key = list(data.keys())[1]  # Skip 'Meta Data' key
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            
            # Add symbol column
            df['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            raise
    
    def get_multiple_symbols(self, symbols: list, period: str = "1y", 
                           interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stock symbols.
        
        Args:
            symbols (list): List of stock symbols
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data_dict[symbol] = self.get_yahoo_data(symbol, period, interval)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    def save_data(self, data: pd.DataFrame, symbol: str, 
                  data_dir: str = "data") -> str:
        """
        Save stock data to CSV file.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            data_dir (str): Directory to save data
        
        Returns:
            str: Path to saved file
        """
        import os
        
        os.makedirs(data_dir, exist_ok=True)
        filename = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(data_dir, filename)
        
        data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        
        return filepath
