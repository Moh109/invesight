"""
Main application for stock prediction.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import Config
from data_collector import StockDataCollector
from feature_engineering import FeatureEngineer
from models import StockPredictor
from visualization import StockVisualizer
from utils import DataValidator, PerformanceCalculator, FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictionApp:
    """Main application class for stock prediction."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = Config()
        self.config.create_directories()
        
        self.data_collector = StockDataCollector()
        self.feature_engineer = FeatureEngineer(lookback_days=self.config.LOOKBACK_DAYS)
        self.predictor = StockPredictor(random_state=self.config.RANDOM_STATE)
        self.visualizer = StockVisualizer()
        self.validator = DataValidator()
        self.performance_calc = PerformanceCalculator()
        self.file_manager = FileManager()
        
        logger.info("Stock Prediction App initialized")
    
    def run_full_pipeline(self, symbol: str, period: str = "1y", 
                         prediction_days: int = 30) -> Dict[str, Any]:
        """
        Run the complete stock prediction pipeline.
        
        Args:
            symbol (str): Stock symbol to analyze
            period (str): Data period to fetch
            prediction_days (int): Number of days to predict ahead
        
        Returns:
            Dict[str, Any]: Results dictionary
        """
        logger.info(f"Starting full pipeline for {symbol}")
        
        try:
            # Step 1: Collect data
            logger.info("Step 1: Collecting stock data...")
            raw_data = self.data_collector.get_yahoo_data(symbol, period)
            
            # Validate data
            if not self.validator.validate_ohlcv_data(raw_data):
                raise ValueError("Data validation failed")
            
            # Step 2: Feature engineering
            logger.info("Step 2: Engineering features...")
            data_with_features = self.feature_engineer.add_technical_indicators(raw_data)
            data_with_lags = self.feature_engineer.create_lag_features(
                data_with_features, target_column='close'
            )
            data_with_rolling = self.feature_engineer.create_rolling_features(
                data_with_lags, target_column='close'
            )
            
            # Clean data
            cleaned_data = self.feature_engineer.clean_data(data_with_rolling)
            
            # Step 3: Prepare features and target
            logger.info("Step 3: Preparing features for ML...")
            features, target = self.feature_engineer.prepare_features(
                cleaned_data, prediction_days=prediction_days
            )
            
            # Step 4: Train models
            logger.info("Step 4: Training machine learning models...")
            X_train, X_test, y_train, y_test = self.predictor.prepare_data(
                features, target, test_size=1-self.config.TRAIN_TEST_SPLIT
            )
            
            trained_models = self.predictor.train_models(X_train, y_train)
            
            # Step 5: Evaluate models
            logger.info("Step 5: Evaluating models...")
            performance = self.predictor.evaluate_models(X_test, y_test)
            
            # Get feature importance
            feature_importance = self.predictor.get_feature_importance(features.columns.tolist())
            
            # Step 6: Make predictions
            logger.info("Step 6: Making predictions...")
            best_model_name, best_model = self.predictor.get_best_model()
            predictions = self.predictor.predict(X_test, best_model_name)
            
            # Step 7: Create visualizations
            logger.info("Step 7: Creating visualizations...")
            self._create_visualizations(
                cleaned_data, predictions, y_test, symbol, best_model_name
            )
            
            # Step 8: Save results
            logger.info("Step 8: Saving results...")
            results = self._save_results(
                symbol, performance, feature_importance, best_model_name, predictions
            )
            
            logger.info(f"Pipeline completed successfully for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _create_visualizations(self, data: pd.DataFrame, predictions: np.ndarray,
                             actual: pd.Series, symbol: str, model_name: str) -> None:
        """Create and save visualizations."""
        try:
            # Price chart
            price_chart = self.visualizer.plot_price_data(data, symbol)
            price_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_price_chart.html")
            
            # Technical indicators
            tech_chart = self.visualizer.plot_technical_indicators(data, symbol)
            tech_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_technical_analysis.html")
            
            # Predictions
            pred_chart = self.visualizer.plot_predictions(actual, predictions, symbol, model_name)
            pred_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_predictions.html")
            
            # Model performance
            if hasattr(self.predictor, 'model_performance') and self.predictor.model_performance:
                perf_chart = self.visualizer.plot_model_performance(self.predictor.model_performance)
                perf_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_model_performance.html")
            
            # Feature importance
            if hasattr(self.predictor, 'feature_importance') and model_name in self.predictor.feature_importance:
                imp_chart = self.visualizer.plot_feature_importance(
                    self.predictor.feature_importance[model_name], model_name
                )
                imp_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_feature_importance.html")
            
            # Dashboard
            dashboard = self.visualizer.create_dashboard(
                data, predictions, actual, symbol, model_name
            )
            dashboard.write_html(f"{self.config.PLOTS_DIR}/{symbol}_dashboard.html")
            
            logger.info(f"Visualizations saved to {self.config.PLOTS_DIR}/")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _save_results(self, symbol: str, performance: Dict, feature_importance: Dict,
                     best_model_name: str, predictions: np.ndarray) -> Dict[str, Any]:
        """Save results to files."""
        try:
            # Save model performance
            self.file_manager.save_json(
                performance, f"{self.config.MODELS_DIR}/{symbol}_performance.json"
            )
            
            # Save feature importance
            for model_name, importance_df in feature_importance.items():
                importance_df.to_csv(
                    f"{self.config.MODELS_DIR}/{symbol}_{model_name}_feature_importance.csv",
                    index=False
                )
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'predictions': predictions,
                'timestamp': datetime.now()
            })
            predictions_df.to_csv(f"{self.config.DATA_DIR}/{symbol}_predictions.csv", index=False)
            
            # Save models
            self.predictor.save_models(self.config.MODELS_DIR)
            
            # Create summary
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'best_model': best_model_name,
                'performance': performance.get(best_model_name, {}),
                'total_features': len(feature_importance.get(best_model_name, [])),
                'prediction_days': self.config.PREDICTION_DAYS
            }
            
            self.file_manager.save_json(
                summary, f"{self.config.DATA_DIR}/{symbol}_summary.json"
            )
            
            logger.info(f"Results saved for {symbol}")
            return summary
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def quick_analysis(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Perform a quick analysis without full ML pipeline.
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info(f"Starting quick analysis for {symbol}")
        
        try:
            # Get data
            data = self.data_collector.get_yahoo_data(symbol, period)
            
            # Add basic technical indicators
            data_with_indicators = self.feature_engineer.add_technical_indicators(data)
            
            # Calculate performance metrics
            returns = self.performance_calc.calculate_returns(data['close'])
            volatility = self.performance_calc.calculate_volatility(returns)
            sharpe_ratio = self.performance_calc.calculate_sharpe_ratio(returns)
            max_drawdown = self.performance_calc.calculate_max_drawdown(data['close'])
            
            # Create basic visualizations
            price_chart = self.visualizer.plot_price_data(data, symbol)
            tech_chart = self.visualizer.plot_technical_indicators(data_with_indicators, symbol)
            
            # Save charts
            price_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_quick_analysis.html")
            tech_chart.write_html(f"{self.config.PLOTS_DIR}/{symbol}_quick_technical.html")
            
            # Prepare results
            results = {
                'symbol': symbol,
                'period': period,
                'total_days': len(data),
                'current_price': data['close'].iloc[-1],
                'price_change': data['close'].pct_change().iloc[-1] * 100,
                'volatility': volatility.iloc[-1] if not volatility.empty else None,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volume_avg': data['volume'].mean(),
                'charts_created': True
            }
            
            logger.info(f"Quick analysis completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Quick analysis failed: {str(e)}")
            raise

def main():
    """Main function to run the application."""
    print("=" * 60)
    print("STOCK PREDICTION APPLICATION")
    print("=" * 60)
    
    # Initialize app
    app = StockPredictionApp()
    
    # Get user input
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper().strip()
    if not symbol:
        symbol = "AAPL"  # Default
    
    print(f"\nAnalyzing {symbol}...")
    print("Choose analysis type:")
    print("1. Quick Analysis (Technical indicators only)")
    print("2. Full ML Pipeline (Complete prediction)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "1":
            # Quick analysis
            results = app.quick_analysis(symbol)
            print(f"\nQuick Analysis Results for {symbol}:")
            print(f"Current Price: ${results['current_price']:.2f}")
            print(f"Price Change: {results['price_change']:.2f}%")
            print(f"Volatility: {results['volatility']:.2f}%" if results['volatility'] else "N/A")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Charts saved to: {app.config.PLOTS_DIR}/")
            
        elif choice == "2":
            # Full pipeline
            results = app.run_full_pipeline(symbol)
            print(f"\nFull Pipeline Results for {symbol}:")
            print(f"Best Model: {results['best_model']}")
            if 'performance' in results and results['performance']:
                perf = results['performance']
                print(f"RMSE: {perf.get('rmse', 'N/A'):.4f}")
                print(f"R²: {perf.get('r2', 'N/A'):.4f}")
                print(f"MAPE: {perf.get('mape', 'N/A'):.2f}%")
            print(f"Total Features: {results['total_features']}")
            print(f"Results saved to: {app.config.DATA_DIR}/ and {app.config.MODELS_DIR}/")
            print(f"Charts saved to: {app.config.PLOTS_DIR}/")
            
        else:
            print("Invalid choice. Running quick analysis...")
            results = app.quick_analysis(symbol)
            print(f"Quick analysis completed for {symbol}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
