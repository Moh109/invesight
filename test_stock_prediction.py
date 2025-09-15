"""
Test script for the stock prediction project.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictor
from utils import DataValidator, PerformanceCalculator
from config import Config

class TestStockPrediction(unittest.TestCase):
    """Test cases for stock prediction components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.data_collector = StockDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.predictor = StockPredictor()
        self.validator = DataValidator()
        self.performance_calc = PerformanceCalculator()
        
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.random.rand(len(dates)) * 2,
            'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - np.random.rand(len(dates)) * 2,
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure high >= low and close is within range
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['close'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['close'])
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['open'])
        self.sample_data['low'] = np.minimum(self.sample_data['low'], self.sample_data['open'])
    
    def test_data_validation(self):
        """Test data validation functionality."""
        # Test valid data
        self.assertTrue(self.validator.validate_ohlcv_data(self.sample_data))
        
        # Test invalid data (missing column)
        invalid_data = self.sample_data.drop('close', axis=1)
        self.assertFalse(self.validator.validate_ohlcv_data(invalid_data))
        
        # Test invalid data (negative prices)
        invalid_data = self.sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = -1
        self.assertFalse(self.validator.validate_ohlcv_data(invalid_data))
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        # Test technical indicators
        data_with_indicators = self.feature_engineer.add_technical_indicators(self.sample_data)
        
        # Check if technical indicators were added
        expected_indicators = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD']
        for indicator in expected_indicators:
            self.assertIn(indicator, data_with_indicators.columns)
        
        # Test lag features
        data_with_lags = self.feature_engineer.create_lag_features(data_with_indicators)
        lag_columns = [col for col in data_with_lags.columns if 'lag' in col]
        self.assertGreater(len(lag_columns), 0)
        
        # Test rolling features
        data_with_rolling = self.feature_engineer.create_rolling_features(data_with_indicators)
        rolling_columns = [col for col in data_with_rolling.columns if 'mean_' in col or 'std_' in col]
        self.assertGreater(len(rolling_columns), 0)
    
    def test_model_training(self):
        """Test model training functionality."""
        # Prepare data
        data_with_indicators = self.feature_engineer.add_technical_indicators(self.sample_data)
        features, target = self.feature_engineer.prepare_features(data_with_indicators)
        
        # Split data
        X_train, X_test, y_train, y_test = self.predictor.prepare_data(features, target)
        
        # Train models
        trained_models = self.predictor.train_models(X_train, y_train)
        
        # Check if models were trained
        self.assertGreater(len(trained_models), 0)
        
        # Test prediction
        predictions = self.predictor.predict(X_test)
        self.assertEqual(len(predictions), len(y_test))
    
    def test_performance_calculation(self):
        """Test performance calculation functionality."""
        # Test returns calculation
        returns = self.performance_calc.calculate_returns(self.sample_data['close'])
        self.assertEqual(len(returns), len(self.sample_data) - 1)
        
        # Test volatility calculation
        volatility = self.performance_calc.calculate_volatility(returns)
        self.assertGreater(len(volatility), 0)
        
        # Test Sharpe ratio calculation
        sharpe_ratio = self.performance_calc.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe_ratio, float)
        
        # Test max drawdown calculation
        max_drawdown = self.performance_calc.calculate_max_drawdown(self.sample_data['close'])
        self.assertLessEqual(max_drawdown, 0)  # Max drawdown should be negative or zero
    
    def test_data_quality_check(self):
        """Test data quality checking functionality."""
        quality_stats = self.validator.check_data_quality(self.sample_data)
        
        # Check if quality stats contain expected keys
        expected_keys = ['total_records', 'missing_values', 'duplicate_records', 'date_range']
        for key in expected_keys:
            self.assertIn(key, quality_stats)
        
        # Check total records
        self.assertEqual(quality_stats['total_records'], len(self.sample_data))
    
    def test_configuration(self):
        """Test configuration settings."""
        # Test default values
        self.assertEqual(self.config.DEFAULT_SYMBOL, "AAPL")
        self.assertEqual(self.config.DEFAULT_PERIOD, "1y")
        self.assertEqual(self.config.TRAIN_TEST_SPLIT, 0.8)
        
        # Test directory creation
        self.config.create_directories()
        self.assertTrue(os.path.exists(self.config.DATA_DIR))
        self.assertTrue(os.path.exists(self.config.MODELS_DIR))
        self.assertTrue(os.path.exists(self.config.PLOTS_DIR))
    
    def test_feature_preparation(self):
        """Test feature preparation for ML."""
        # Add technical indicators
        data_with_indicators = self.feature_engineer.add_technical_indicators(self.sample_data)
        
        # Prepare features
        features, target = self.feature_engineer.prepare_features(data_with_indicators)
        
        # Check if features and target have same length
        self.assertEqual(len(features), len(target))
        
        # Check if features are numeric
        self.assertTrue(features.select_dtypes(include=[np.number]).shape[1] == features.shape[1])
        
        # Check if target is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(target))

def run_tests():
    """Run all tests."""
    print("Running Stock Prediction Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestStockPrediction)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
