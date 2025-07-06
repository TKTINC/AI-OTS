#!/usr/bin/env python3
"""
AI-OTS Pattern Recognition Model Framework
Implements CNN-LSTM hybrid models for chart pattern recognition
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Pattern recognition will use traditional ML methods.")

# Traditional ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Image processing for chart patterns
try:
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    print("Imaging libraries not available. Chart visualization will be limited.")

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatternConfig:
    """Configuration for pattern recognition models"""
    sequence_length: int = 50
    prediction_horizon: int = 5
    pattern_types: List[str] = None
    confidence_threshold: float = 0.8
    image_size: Tuple[int, int] = (224, 224)
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.pattern_types is None:
            self.pattern_types = [
                'head_and_shoulders',
                'inverse_head_and_shoulders',
                'double_top',
                'double_bottom',
                'triangle_ascending',
                'triangle_descending',
                'triangle_symmetrical',
                'wedge_rising',
                'wedge_falling',
                'flag_bullish',
                'flag_bearish',
                'pennant',
                'cup_and_handle',
                'support_breakout',
                'resistance_breakout'
            ]

class ChartPatternDetector:
    """Traditional algorithmic pattern detection"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        
    def detect_head_and_shoulders(self, prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
        """Detect head and shoulders pattern"""
        if len(prices) < window * 3:
            return {'detected': False, 'confidence': 0.0}
            
        # Find local maxima
        peaks = []
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                peaks.append((i, prices[i]))
                
        if len(peaks) < 3:
            return {'detected': False, 'confidence': 0.0}
            
        # Check for head and shoulders pattern
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Head should be higher than shoulders
            if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                abs(left_shoulder[1] - right_shoulder[1]) / head[1] < 0.05):
                
                # Calculate neckline
                neckline_level = min(
                    min(prices[left_shoulder[0]:head[0]]),
                    min(prices[head[0]:right_shoulder[0]])
                )
                
                confidence = min(
                    (head[1] - left_shoulder[1]) / head[1],
                    (head[1] - right_shoulder[1]) / head[1]
                )
                
                return {
                    'detected': True,
                    'confidence': confidence,
                    'pattern_type': 'head_and_shoulders',
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': neckline_level,
                    'target_price': neckline_level - (head[1] - neckline_level)
                }
                
        return {'detected': False, 'confidence': 0.0}
        
    def detect_double_top(self, prices: np.ndarray, window: int = 15) -> Dict[str, Any]:
        """Detect double top pattern"""
        if len(prices) < window * 4:
            return {'detected': False, 'confidence': 0.0}
            
        peaks = []
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                peaks.append((i, prices[i]))
                
        if len(peaks) < 2:
            return {'detected': False, 'confidence': 0.0}
            
        # Check for double top pattern
        for i in range(len(peaks) - 1):
            first_peak = peaks[i]
            second_peak = peaks[i + 1]
            
            # Peaks should be at similar levels
            price_diff = abs(first_peak[1] - second_peak[1]) / max(first_peak[1], second_peak[1])
            
            if price_diff < 0.03:  # Within 3%
                # Find valley between peaks
                valley_start = first_peak[0]
                valley_end = second_peak[0]
                valley_price = min(prices[valley_start:valley_end])
                
                # Valley should be significantly lower
                valley_depth = (min(first_peak[1], second_peak[1]) - valley_price) / min(first_peak[1], second_peak[1])
                
                if valley_depth > 0.05:  # At least 5% drop
                    confidence = valley_depth * (1 - price_diff)
                    
                    return {
                        'detected': True,
                        'confidence': confidence,
                        'pattern_type': 'double_top',
                        'first_peak': first_peak,
                        'second_peak': second_peak,
                        'valley_price': valley_price,
                        'target_price': valley_price - (min(first_peak[1], second_peak[1]) - valley_price)
                    }
                    
        return {'detected': False, 'confidence': 0.0}
        
    def detect_triangle_pattern(self, prices: np.ndarray, window: int = 30) -> Dict[str, Any]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(prices) < window * 2:
            return {'detected': False, 'confidence': 0.0}
            
        # Find trend lines
        highs = []
        lows = []
        
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window//2:i+window//2+1]):
                highs.append((i, prices[i]))
            if prices[i] == min(prices[i-window//2:i+window//2+1]):
                lows.append((i, prices[i]))
                
        if len(highs) < 2 or len(lows) < 2:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate trend lines
        high_slope = (highs[-1][1] - highs[0][1]) / (highs[-1][0] - highs[0][0])
        low_slope = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0])
        
        # Determine pattern type
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            pattern_type = 'triangle_ascending'
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            pattern_type = 'triangle_descending'
        elif high_slope < -0.001 and low_slope > 0.001:
            pattern_type = 'triangle_symmetrical'
        else:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate convergence point
        if abs(high_slope - low_slope) > 0.001:
            convergence_x = (lows[0][1] - highs[0][1] + high_slope * highs[0][0] - low_slope * lows[0][0]) / (high_slope - low_slope)
            convergence_y = highs[0][1] + high_slope * (convergence_x - highs[0][0])
            
            confidence = min(1.0, abs(high_slope - low_slope) * 100)
            
            return {
                'detected': True,
                'confidence': confidence,
                'pattern_type': pattern_type,
                'high_trend_line': {'slope': high_slope, 'points': highs},
                'low_trend_line': {'slope': low_slope, 'points': lows},
                'convergence_point': (convergence_x, convergence_y)
            }
            
        return {'detected': False, 'confidence': 0.0}
        
    def detect_support_resistance_breakout(self, prices: np.ndarray, volume: np.ndarray, window: int = 20) -> Dict[str, Any]:
        """Detect support/resistance breakouts"""
        if len(prices) < window * 3:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate support and resistance levels
        recent_prices = prices[-window*2:]
        support_level = np.min(recent_prices)
        resistance_level = np.max(recent_prices)
        
        # Check for breakout
        current_price = prices[-1]
        previous_price = prices[-2]
        current_volume = volume[-1] if len(volume) > 0 else 1
        avg_volume = np.mean(volume[-window:]) if len(volume) >= window else 1
        
        volume_confirmation = current_volume > avg_volume * 1.5
        
        # Resistance breakout
        if (previous_price <= resistance_level and current_price > resistance_level and volume_confirmation):
            confidence = min(1.0, (current_price - resistance_level) / resistance_level * 10)
            return {
                'detected': True,
                'confidence': confidence,
                'pattern_type': 'resistance_breakout',
                'breakout_level': resistance_level,
                'breakout_direction': 'bullish',
                'volume_confirmation': volume_confirmation,
                'target_price': current_price + (current_price - support_level)
            }
            
        # Support breakout (breakdown)
        if (previous_price >= support_level and current_price < support_level and volume_confirmation):
            confidence = min(1.0, (support_level - current_price) / support_level * 10)
            return {
                'detected': True,
                'confidence': confidence,
                'pattern_type': 'support_breakout',
                'breakout_level': support_level,
                'breakout_direction': 'bearish',
                'volume_confirmation': volume_confirmation,
                'target_price': current_price - (resistance_level - current_price)
            }
            
        return {'detected': False, 'confidence': 0.0}
        
    def detect_all_patterns(self, prices: np.ndarray, volume: np.ndarray = None) -> List[Dict[str, Any]]:
        """Detect all supported patterns"""
        patterns = []
        
        # Head and shoulders
        hs_result = self.detect_head_and_shoulders(prices)
        if hs_result['detected']:
            patterns.append(hs_result)
            
        # Double top
        dt_result = self.detect_double_top(prices)
        if dt_result['detected']:
            patterns.append(dt_result)
            
        # Triangle patterns
        triangle_result = self.detect_triangle_pattern(prices)
        if triangle_result['detected']:
            patterns.append(triangle_result)
            
        # Support/Resistance breakouts
        if volume is not None:
            breakout_result = self.detect_support_resistance_breakout(prices, volume)
            if breakout_result['detected']:
                patterns.append(breakout_result)
                
        return patterns

class CNNLSTMPatternRecognizer:
    """CNN-LSTM hybrid model for pattern recognition"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for CNN-LSTM training"""
        # Normalize price data
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = self.scaler.fit_transform(data[price_cols].values)
        
        sequences = []
        labels = []
        
        for i in range(self.config.sequence_length, len(scaled_data)):
            sequences.append(scaled_data[i-self.config.sequence_length:i])
            
            # Create labels based on future price movement
            future_price = data.iloc[i]['close']
            current_price = data.iloc[i-1]['close']
            price_change = (future_price - current_price) / current_price
            
            # Classify into pattern categories based on price movement
            if price_change > 0.02:
                labels.append('bullish_breakout')
            elif price_change < -0.02:
                labels.append('bearish_breakout')
            elif abs(price_change) < 0.005:
                labels.append('consolidation')
            else:
                labels.append('neutral')
                
        return np.array(sequences), np.array(labels)
        
    def build_model(self, input_shape: Tuple[int, int], num_classes: int) -> Model:
        """Build CNN-LSTM hybrid model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN-LSTM model")
            
        # Input layer
        inputs = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(0.2)(conv3)
        
        # LSTM layers for sequence modeling
        lstm1 = LSTM(100, return_sequences=True)(conv3)
        lstm1 = Dropout(0.3)(lstm1)
        
        lstm2 = LSTM(50, return_sequences=False)(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        
        # Dense layers for classification
        dense1 = Dense(50, activation='relu')(lstm2)
        dense1 = Dropout(0.4)(dense1)
        
        outputs = Dense(num_classes, activation='softmax')(dense1)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the CNN-LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using fallback RandomForest model.")
            return self.train_fallback(data)
            
        logger.info("Training CNN-LSTM pattern recognition model...")
        
        # Prepare data
        X, y = self.prepare_sequences(data)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=self.config.validation_split, random_state=42
        )
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(input_shape, num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_pattern_model.h5', save_best_only=True)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test_classes, y_pred_classes, target_names=class_names, output_dict=True)
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'training_history': history.history,
            'model_type': 'CNN-LSTM',
            'num_classes': num_classes,
            'class_names': class_names.tolist()
        }
        
        logger.info(f"Model trained. Test accuracy: {test_accuracy:.3f}")
        return results
        
    def train_fallback(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback training using RandomForest"""
        logger.info("Training fallback RandomForest pattern recognition model...")
        
        # Prepare features
        features = []
        labels = []
        
        for i in range(self.config.sequence_length, len(data)):
            # Extract features from price sequence
            sequence = data.iloc[i-self.config.sequence_length:i]
            
            # Statistical features
            price_features = [
                sequence['close'].mean(),
                sequence['close'].std(),
                sequence['close'].max(),
                sequence['close'].min(),
                sequence['volume'].mean(),
                sequence['volume'].std(),
                (sequence['close'].iloc[-1] - sequence['close'].iloc[0]) / sequence['close'].iloc[0],
                sequence['close'].rolling(5).mean().iloc[-1],
                sequence['close'].rolling(10).mean().iloc[-1]
            ]
            
            features.append(price_features)
            
            # Create labels
            future_price = data.iloc[i]['close']
            current_price = data.iloc[i-1]['close']
            price_change = (future_price - current_price) / current_price
            
            if price_change > 0.02:
                labels.append('bullish_breakout')
            elif price_change < -0.02:
                labels.append('bearish_breakout')
            elif abs(price_change) < 0.005:
                labels.append('consolidation')
            else:
                labels.append('neutral')
                
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=self.config.validation_split, random_state=42
        )
        
        # Train RandomForest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        results = {
            'test_accuracy': accuracy,
            'classification_report': report,
            'model_type': 'RandomForest',
            'num_classes': len(class_names),
            'class_names': class_names.tolist()
        }
        
        logger.info(f"Fallback model trained. Test accuracy: {accuracy:.3f}")
        return results
        
    def predict_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict patterns in new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if TENSORFLOW_AVAILABLE and hasattr(self.model, 'predict'):
            return self._predict_cnn_lstm(data)
        else:
            return self._predict_fallback(data)
            
    def _predict_cnn_lstm(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict using CNN-LSTM model"""
        # Prepare sequences
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = self.scaler.transform(data[price_cols].values)
        
        predictions = []
        
        for i in range(self.config.sequence_length, len(scaled_data)):
            sequence = scaled_data[i-self.config.sequence_length:i].reshape(1, self.config.sequence_length, -1)
            
            # Get prediction
            pred_proba = self.model.predict(sequence, verbose=0)[0]
            pred_class_idx = np.argmax(pred_proba)
            pred_class = self.label_encoder.classes_[pred_class_idx]
            confidence = pred_proba[pred_class_idx]
            
            if confidence >= self.config.confidence_threshold:
                predictions.append({
                    'timestamp': data.iloc[i]['date'] if 'date' in data.columns else i,
                    'pattern_type': pred_class,
                    'confidence': float(confidence),
                    'model_type': 'CNN-LSTM',
                    'probabilities': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.label_encoder.classes_, pred_proba)
                    }
                })
                
        return predictions
        
    def _predict_fallback(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict using fallback model"""
        predictions = []
        
        for i in range(self.config.sequence_length, len(data)):
            # Extract features
            sequence = data.iloc[i-self.config.sequence_length:i]
            
            features = np.array([[
                sequence['close'].mean(),
                sequence['close'].std(),
                sequence['close'].max(),
                sequence['close'].min(),
                sequence['volume'].mean(),
                sequence['volume'].std(),
                (sequence['close'].iloc[-1] - sequence['close'].iloc[0]) / sequence['close'].iloc[0],
                sequence['close'].rolling(5).mean().iloc[-1],
                sequence['close'].rolling(10).mean().iloc[-1]
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction
            pred_proba = self.model.predict_proba(features_scaled)[0]
            pred_class_idx = np.argmax(pred_proba)
            pred_class = self.label_encoder.classes_[pred_class_idx]
            confidence = pred_proba[pred_class_idx]
            
            if confidence >= self.config.confidence_threshold:
                predictions.append({
                    'timestamp': data.iloc[i]['date'] if 'date' in data.columns else i,
                    'pattern_type': pred_class,
                    'confidence': float(confidence),
                    'model_type': 'RandomForest',
                    'probabilities': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.label_encoder.classes_, pred_proba)
                    }
                })
                
        return predictions
        
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'config': self.config.__dict__,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'model_type': 'CNN-LSTM' if TENSORFLOW_AVAILABLE else 'RandomForest'
        }
        
        if TENSORFLOW_AVAILABLE and hasattr(self.model, 'save'):
            # Save TensorFlow model
            self.model.save(f"{filepath}_model.h5")
            
            # Save other components
            import pickle
            with open(f"{filepath}_components.pkl", 'wb') as f:
                pickle.dump(model_data, f)
        else:
            # Save everything together for sklearn model
            model_data['model'] = self.model
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
        logger.info(f"Model saved to {filepath}")

class PatternRecognitionEnsemble:
    """Ensemble combining algorithmic and ML pattern recognition"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.algorithmic_detector = ChartPatternDetector(config)
        self.ml_recognizer = CNNLSTMPatternRecognizer(config)
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the ML component of the ensemble"""
        logger.info("Training pattern recognition ensemble...")
        
        # Train ML model
        ml_results = self.ml_recognizer.train(data)
        
        return {
            'ml_model_results': ml_results,
            'algorithmic_patterns': len(self.config.pattern_types),
            'ensemble_ready': True
        }
        
    def detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns using both algorithmic and ML approaches"""
        results = {
            'algorithmic_patterns': [],
            'ml_patterns': [],
            'ensemble_patterns': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Algorithmic detection
        if len(data) >= 50:
            prices = data['close'].values
            volume = data['volume'].values if 'volume' in data.columns else None
            
            algorithmic_patterns = self.algorithmic_detector.detect_all_patterns(prices, volume)
            results['algorithmic_patterns'] = algorithmic_patterns
            
        # ML detection
        if self.ml_recognizer.model is not None and len(data) >= self.config.sequence_length:
            ml_patterns = self.ml_recognizer.predict_patterns(data)
            results['ml_patterns'] = ml_patterns
            
        # Ensemble combination
        ensemble_patterns = self._combine_predictions(
            results['algorithmic_patterns'], 
            results['ml_patterns']
        )
        results['ensemble_patterns'] = ensemble_patterns
        
        return results
        
    def _combine_predictions(self, algorithmic: List[Dict], ml: List[Dict]) -> List[Dict]:
        """Combine algorithmic and ML predictions"""
        ensemble = []
        
        # Add high-confidence algorithmic patterns
        for pattern in algorithmic:
            if pattern.get('confidence', 0) >= 0.7:
                pattern['source'] = 'algorithmic'
                pattern['ensemble_confidence'] = pattern['confidence'] * 0.8  # Slight discount
                ensemble.append(pattern)
                
        # Add high-confidence ML patterns
        for pattern in ml:
            if pattern.get('confidence', 0) >= self.config.confidence_threshold:
                pattern['source'] = 'ml'
                pattern['ensemble_confidence'] = pattern['confidence'] * 0.9  # Slight discount
                ensemble.append(pattern)
                
        # Sort by ensemble confidence
        ensemble.sort(key=lambda x: x.get('ensemble_confidence', 0), reverse=True)
        
        return ensemble

def main():
    """Main training and testing function"""
    # Configuration
    config = PatternConfig(
        sequence_length=50,
        prediction_horizon=5,
        confidence_threshold=0.8,
        validation_split=0.2
    )
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate more realistic price data with patterns
    base_price = 150
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # Add some trend and noise
        trend = 0.001 * np.sin(i / 50)  # Long-term trend
        noise = np.random.randn() * 0.02  # Daily noise
        
        # Occasional pattern-like movements
        if i % 100 == 0:  # Every 100 days, create a pattern
            pattern_move = np.random.choice([-0.1, 0.1])  # Big move up or down
            prices.append(prices[-1] * (1 + trend + noise + pattern_move))
        else:
            prices.append(prices[-1] * (1 + trend + noise))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'symbol': 'AAPL',
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Initialize ensemble
    ensemble = PatternRecognitionEnsemble(config)
    
    # Train ensemble
    print("Training pattern recognition ensemble...")
    results = ensemble.train(sample_data)
    
    print(f"Training completed. ML Model Type: {results['ml_model_results']['model_type']}")
    print(f"Test Accuracy: {results['ml_model_results']['test_accuracy']:.3f}")
    
    # Detect patterns
    print("\nDetecting patterns...")
    pattern_results = ensemble.detect_patterns(sample_data.tail(100))
    
    print(f"Algorithmic patterns detected: {len(pattern_results['algorithmic_patterns'])}")
    print(f"ML patterns detected: {len(pattern_results['ml_patterns'])}")
    print(f"Ensemble patterns: {len(pattern_results['ensemble_patterns'])}")
    
    # Display some results
    if pattern_results['ensemble_patterns']:
        print("\nTop ensemble patterns:")
        for i, pattern in enumerate(pattern_results['ensemble_patterns'][:3]):
            print(f"{i+1}. {pattern.get('pattern_type', 'Unknown')} - "
                  f"Confidence: {pattern.get('ensemble_confidence', 0):.3f} - "
                  f"Source: {pattern.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()

