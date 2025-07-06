#!/usr/bin/env python3
"""
AI-OTS Signal Generation Model Training Framework
Implements ensemble-based signal generation with multiple trading strategies
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb

# Technical Analysis
import pandas_ta as ta

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """Configuration for signal generation models"""
    confidence_threshold: float = 0.6
    ensemble_size: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    lookback_period: int = 20
    prediction_horizon: int = 5
    rebalance_frequency: str = "daily"
    
@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_calibration: float
    sharpe_ratio: float
    max_drawdown: float

class TechnicalIndicatorEngine:
    """Technical indicator calculation engine"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        logger.info("Calculating technical indicators...")
        
        # Price-based indicators
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # Momentum indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
        df['macd_histogram'] = ta.macd(df['close'])['MACDh_12_26_9']
        
        # Volatility indicators
        bb = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Trend indicators
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Support/Resistance levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance1'] = 2 * df['pivot'] - df['low']
        df['support1'] = 2 * df['pivot'] - df['high']
        
        # Price patterns
        df['doji'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='doji')
        df['hammer'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='hammer')
        df['engulfing'] = ta.cdl_pattern(df['open'], df['high'], df['low'], df['close'], name='engulfing')
        
        # Custom indicators
        df['price_momentum'] = df['close'].pct_change(5)
        df['volume_momentum'] = df['volume'].pct_change(5)
        df['volatility'] = df['close'].rolling(20).std()
        df['price_position'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        return df.fillna(method='ffill').fillna(0)

class OptionsFlowAnalyzer:
    """Options flow analysis for signal generation"""
    
    def __init__(self):
        self.flow_indicators = {}
        
    def analyze_options_flow(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze options flow patterns"""
        logger.info("Analyzing options flow...")
        
        if options_data.empty:
            return pd.DataFrame()
            
        # Call/Put ratio
        calls = options_data[options_data['option_type'] == 'call']
        puts = options_data[options_data['option_type'] == 'put']
        
        flow_metrics = pd.DataFrame()
        flow_metrics['call_volume'] = calls.groupby('date')['volume'].sum()
        flow_metrics['put_volume'] = puts.groupby('date')['volume'].sum()
        flow_metrics['call_put_ratio'] = flow_metrics['call_volume'] / flow_metrics['put_volume']
        
        # Unusual activity detection
        flow_metrics['call_volume_ma'] = flow_metrics['call_volume'].rolling(20).mean()
        flow_metrics['put_volume_ma'] = flow_metrics['put_volume'].rolling(20).mean()
        flow_metrics['unusual_call_activity'] = flow_metrics['call_volume'] > (flow_metrics['call_volume_ma'] * 2)
        flow_metrics['unusual_put_activity'] = flow_metrics['put_volume'] > (flow_metrics['put_volume_ma'] * 2)
        
        # Options sentiment
        flow_metrics['options_sentiment'] = np.where(
            flow_metrics['call_put_ratio'] > 1.2, 1,  # Bullish
            np.where(flow_metrics['call_put_ratio'] < 0.8, -1, 0)  # Bearish
        )
        
        return flow_metrics.fillna(0)

class SignalGenerationModel:
    """Individual signal generation model"""
    
    def __init__(self, model_type: str, strategy_name: str, config: SignalConfig):
        self.model_type = model_type
        self.strategy_name = strategy_name
        self.config = config
        self.model = None
        self.scaler = RobustScaler()
        self.feature_importance = {}
        self.metrics = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for model training"""
        features = df.copy()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = features['close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
            
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'close_mean_{window}'] = features['close'].rolling(window).mean()
            features[f'close_std_{window}'] = features['close'].rolling(window).std()
            features[f'volume_mean_{window}'] = features['volume'].rolling(window).mean()
            
        # Price ratios
        features['close_to_sma20'] = features['close'] / features['sma_20']
        features['close_to_sma50'] = features['close'] / features['sma_50']
        features['sma20_to_sma50'] = features['sma_20'] / features['sma_50']
        
        # Momentum features
        features['price_acceleration'] = features['close'].diff().diff()
        features['volume_acceleration'] = features['volume'].diff().diff()
        
        return features.fillna(method='ffill').fillna(0)
        
    def create_targets(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable for training"""
        # Forward-looking returns
        future_returns = df['close'].shift(-self.config.prediction_horizon) / df['close'] - 1
        
        # Binary classification: 1 if return > threshold, 0 otherwise
        threshold = 0.02  # 2% return threshold
        targets = (future_returns > threshold).astype(int)
        
        return targets
        
    def train(self, df: pd.DataFrame) -> ModelMetrics:
        """Train the signal generation model"""
        logger.info(f"Training {self.strategy_name} model...")
        
        # Create features and targets
        features_df = self.create_features(df)
        targets = self.create_targets(df)
        
        # Select feature columns (exclude non-feature columns)
        feature_cols = [col for col in features_df.columns if col not in 
                       ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_cols].values
        y = targets.values
        
        # Remove rows with NaN targets
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Create model based on type
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='accuracy')
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        # Calculate metrics
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1_score=f1_score(y, y_pred, zero_division=0),
            confidence_calibration=np.mean(cv_scores),
            sharpe_ratio=0.0,  # Will be calculated in backtesting
            max_drawdown=0.0   # Will be calculated in backtesting
        )
        
        self.metrics = metrics
        logger.info(f"Model trained. Accuracy: {metrics.accuracy:.3f}, F1: {metrics.f1_score:.3f}")
        
        return metrics
        
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        features_df = self.create_features(df)
        feature_cols = [col for col in features_df.columns if col not in 
                       ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        X = features_df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Generate signals
        signals = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if prob >= self.config.confidence_threshold:
                signal = {
                    'strategy': self.strategy_name,
                    'signal_type': 'BUY' if pred == 1 else 'SELL',
                    'confidence': float(prob),
                    'timestamp': datetime.now(),
                    'features_used': len(feature_cols),
                    'model_type': self.model_type
                }
                signals.append(signal)
                
        return {
            'signals': signals,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'feature_importance': self.feature_importance
        }
        
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'strategy_name': self.strategy_name,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.metrics = model_data['metrics']
        self.feature_importance = model_data['feature_importance']
        self.strategy_name = model_data['strategy_name']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")

class EnsembleSignalGenerator:
    """Ensemble of signal generation models"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.models = []
        self.technical_engine = TechnicalIndicatorEngine()
        self.options_analyzer = OptionsFlowAnalyzer()
        
    def create_strategy_models(self) -> List[SignalGenerationModel]:
        """Create individual strategy models"""
        strategies = [
            ('momentum', 'random_forest'),
            ('mean_reversion', 'gradient_boosting'),
            ('breakout', 'xgboost'),
            ('volatility', 'lightgbm'),
            ('trend_following', 'random_forest'),
            ('support_resistance', 'gradient_boosting'),
            ('volume_analysis', 'xgboost'),
            ('pattern_recognition', 'lightgbm'),
            ('options_flow', 'random_forest'),
            ('multi_timeframe', 'gradient_boosting')
        ]
        
        models = []
        for strategy_name, model_type in strategies:
            model = SignalGenerationModel(model_type, strategy_name, self.config)
            models.append(model)
            
        return models
        
    def train_ensemble(self, market_data: pd.DataFrame, options_data: pd.DataFrame = None) -> Dict[str, ModelMetrics]:
        """Train ensemble of models"""
        logger.info("Training ensemble signal generation models...")
        
        # Prepare data
        df = self.technical_engine.calculate_indicators(market_data)
        
        if options_data is not None and not options_data.empty:
            options_flow = self.options_analyzer.analyze_options_flow(options_data)
            df = df.merge(options_flow, left_on='date', right_index=True, how='left')
        
        # Create and train models
        self.models = self.create_strategy_models()
        metrics_dict = {}
        
        for model in self.models:
            try:
                metrics = model.train(df)
                metrics_dict[model.strategy_name] = metrics
                logger.info(f"Trained {model.strategy_name}: Accuracy={metrics.accuracy:.3f}")
            except Exception as e:
                logger.error(f"Failed to train {model.strategy_name}: {e}")
                
        return metrics_dict
        
    def generate_signals(self, market_data: pd.DataFrame, options_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate ensemble signals"""
        if not self.models:
            raise ValueError("Ensemble not trained. Call train_ensemble() first.")
            
        # Prepare data
        df = self.technical_engine.calculate_indicators(market_data)
        
        if options_data is not None and not options_data.empty:
            options_flow = self.options_analyzer.analyze_options_flow(options_data)
            df = df.merge(options_flow, left_on='date', right_index=True, how='left')
        
        # Get predictions from all models
        all_signals = []
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            try:
                result = model.predict(df)
                all_signals.extend(result['signals'])
                all_predictions.append(result['predictions'])
                all_probabilities.append(result['probabilities'])
            except Exception as e:
                logger.error(f"Failed to generate signals from {model.strategy_name}: {e}")
                
        # Ensemble voting
        if all_probabilities:
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions = (ensemble_probabilities >= self.config.confidence_threshold).astype(int)
            
            # Generate ensemble signals
            ensemble_signals = []
            for i, (pred, prob) in enumerate(zip(ensemble_predictions, ensemble_probabilities)):
                if prob >= self.config.confidence_threshold:
                    signal = {
                        'strategy': 'ensemble',
                        'signal_type': 'BUY' if pred == 1 else 'SELL',
                        'confidence': float(prob),
                        'timestamp': datetime.now(),
                        'ensemble_size': len(self.models),
                        'individual_signals': len([s for s in all_signals if s.get('confidence', 0) >= self.config.confidence_threshold])
                    }
                    ensemble_signals.append(signal)
        else:
            ensemble_signals = []
            ensemble_probabilities = []
            
        return {
            'ensemble_signals': ensemble_signals,
            'individual_signals': all_signals,
            'ensemble_probabilities': ensemble_probabilities.tolist() if len(ensemble_probabilities) > 0 else [],
            'model_count': len(self.models),
            'timestamp': datetime.now().isoformat()
        }
        
    def save_ensemble(self, directory: str):
        """Save entire ensemble"""
        os.makedirs(directory, exist_ok=True)
        
        for i, model in enumerate(self.models):
            filepath = os.path.join(directory, f"{model.strategy_name}_{model.model_type}.pkl")
            model.save_model(filepath)
            
        # Save ensemble configuration
        config_path = os.path.join(directory, "ensemble_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'model_count': len(self.models),
                'strategies': [m.strategy_name for m in self.models],
                'model_types': [m.model_type for m in self.models],
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
            
        logger.info(f"Ensemble saved to {directory}")
        
    def load_ensemble(self, directory: str):
        """Load entire ensemble"""
        # Load configuration
        config_path = os.path.join(directory, "ensemble_config.json")
        with open(config_path, 'r') as f:
            ensemble_config = json.load(f)
            
        # Recreate config
        self.config = SignalConfig(**ensemble_config['config'])
        
        # Load models
        self.models = []
        for strategy, model_type in zip(ensemble_config['strategies'], ensemble_config['model_types']):
            model = SignalGenerationModel(model_type, strategy, self.config)
            filepath = os.path.join(directory, f"{strategy}_{model_type}.pkl")
            model.load_model(filepath)
            self.models.append(model)
            
        logger.info(f"Ensemble loaded from {directory}")

def main():
    """Main training function"""
    # Configuration
    config = SignalConfig(
        confidence_threshold=0.7,
        ensemble_size=10,
        validation_split=0.2,
        test_split=0.1,
        lookback_period=20,
        prediction_horizon=5
    )
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'symbol': 'AAPL',
        'open': 150 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 150 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 2,
        'low': 150 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 2,
        'close': 150 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Initialize ensemble
    ensemble = EnsembleSignalGenerator(config)
    
    # Train ensemble
    print("Training ensemble...")
    metrics = ensemble.train_ensemble(sample_data)
    
    # Print results
    print("\nTraining Results:")
    for strategy, metric in metrics.items():
        print(f"{strategy}: Accuracy={metric.accuracy:.3f}, F1={metric.f1_score:.3f}")
    
    # Generate signals
    print("\nGenerating signals...")
    signals = ensemble.generate_signals(sample_data.tail(50))
    
    print(f"Generated {len(signals['ensemble_signals'])} ensemble signals")
    print(f"Generated {len(signals['individual_signals'])} individual signals")
    
    # Save ensemble
    ensemble.save_ensemble("./ai-models/models/signal-generation/")
    print("Ensemble saved successfully")

if __name__ == "__main__":
    main()

