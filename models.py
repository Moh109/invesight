"""
Machine learning models for stock prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Logging configuration removed - using main module's logging setup
logger = logging.getLogger(__name__)

class StockPredictor:
    """Class for training and using machine learning models for stock prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the stock predictor.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            features (pd.DataFrame): Feature matrix
            target (pd.Series): Target values
            test_size (float): Proportion of data for testing
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=self.random_state, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        logger.info(f"Data prepared: Train set {X_train_scaled.shape}, Test set {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple machine learning models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
        
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        trained_models = {}
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"{name} trained successfully")
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
        
        Returns:
            Dict[str, Dict[str, float]]: Model performance metrics
        """
        performance = {}
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Calculate percentage errors
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                performance[name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape
                }
                
                logger.info(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
                continue
        
        self.model_performance = performance
        return performance
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names (List[str]): List of feature names
        
        Returns:
            Dict[str, pd.DataFrame]: Feature importance for each model
        """
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_dict[name] = importance
                logger.info(f"Feature importance calculated for {name}")
        
        self.feature_importance = importance_dict
        return importance_dict
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                            model_name: str = 'random_forest') -> Any:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            model_name (str): Name of the model to tune
        
        Returns:
            Any: Best model after tuning
        """
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=self.random_state)
        
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingRegressor(random_state=self.random_state)
        
        else:
            logger.warning(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.models[f'{model_name}_tuned'] = best_model
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return best_model
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X (np.ndarray): Features for prediction
            model_name (str): Name of the model to use (uses best model if None)
        
        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            # Use the best performing model
            if self.model_performance:
                best_model_name = min(self.model_performance.keys(), 
                                    key=lambda x: self.model_performance[x]['rmse'])
                model_name = best_model_name
            else:
                raise ValueError("No trained models available")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def save_models(self, models_dir: str = "models") -> None:
        """
        Save trained models to disk.
        
        Args:
            models_dir (str): Directory to save models
        """
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(models_dir, f"{name}.joblib")
            joblib.dump(model, filepath)
            logger.info(f"Model {name} saved to {filepath}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            filepath = os.path.join(models_dir, f"scaler_{name}.joblib")
            joblib.dump(scaler, filepath)
            logger.info(f"Scaler {name} saved to {filepath}")
    
    def load_models(self, models_dir: str = "models") -> None:
        """
        Load trained models from disk.
        
        Args:
            models_dir (str): Directory containing saved models
        """
        import os
        import glob
        
        # Load models
        model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
        for filepath in model_files:
            if not filepath.endswith("scaler_standard.joblib"):
                model_name = os.path.basename(filepath).replace('.joblib', '')
                self.models[model_name] = joblib.load(filepath)
                logger.info(f"Model {model_name} loaded from {filepath}")
        
        # Load scalers
        scaler_files = glob.glob(os.path.join(models_dir, "scaler_*.joblib"))
        for filepath in scaler_files:
            scaler_name = os.path.basename(filepath).replace('scaler_', '').replace('.joblib', '')
            self.scalers[scaler_name] = joblib.load(filepath)
            logger.info(f"Scaler {scaler_name} loaded from {filepath}")
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on RMSE.
        
        Returns:
            Tuple[str, Any]: Model name and model object
        """
        if not self.model_performance:
            raise ValueError("No model performance data available")
        
        best_model_name = min(self.model_performance.keys(), 
                            key=lambda x: self.model_performance[x]['rmse'])
        best_model = self.models[best_model_name]
        
        return best_model_name, best_model
