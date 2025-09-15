"""
Configuration settings for the stock prediction project.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for stock prediction settings."""
    
    # Data settings
    DEFAULT_SYMBOL = "AAPL"
    DEFAULT_PERIOD = "1y"
    DEFAULT_INTERVAL = "1d"
    
    # Model settings
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_STATE = 42
    PREDICTION_DAYS = 30
    
    # Feature engineering
    LOOKBACK_DAYS = 14
    TECHNICAL_INDICATORS = [
        'SMA_5', 'SMA_10', 'SMA_20',
        'EMA_5', 'EMA_10', 'EMA_20',
        'RSI', 'MACD', 'MACD_signal',
        'BB_upper', 'BB_middle', 'BB_lower',
        'Volume_SMA'
    ]
    
    # API settings
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
    # File paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    PLOTS_DIR = "plots"
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the project."""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)
