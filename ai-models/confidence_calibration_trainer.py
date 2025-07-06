#!/usr/bin/env python3
"""
AI-OTS Confidence Calibration Framework
Implements advanced confidence calibration for trading signal reliability
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries for calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import brier_score_loss, log_loss
import joblib

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration"""
    calibration_method: str = 'isotonic'  # 'isotonic', 'platt', 'beta', 'temperature'
    validation_split: float = 0.3
    n_bins: int = 10
    target_reliability: float = 0.95
    min_samples_per_bin: int = 50
    cross_validation_folds: int = 5
    
@dataclass
class CalibrationMetrics:
    """Metrics for calibration performance"""
    reliability_score: float
    brier_score: float
    log_loss_score: float
    ece_score: float  # Expected Calibration Error
    mce_score: float  # Maximum Calibration Error
    calibration_slope: float
    calibration_intercept: float

class ReliabilityDiagram:
    """Generate and analyze reliability diagrams"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        
    def compute_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve data"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy='uniform'
        )
        return fraction_of_positives, mean_predicted_value
        
    def plot_reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                title: str = "Reliability Diagram", save_path: str = None) -> Dict[str, float]:
        """Plot reliability diagram and return calibration metrics"""
        fraction_pos, mean_pred = self.compute_calibration_curve(y_true, y_prob)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Main reliability plot
        plt.subplot(2, 2, 1)
        plt.plot(mean_pred, fraction_pos, "s-", label="AI-OTS Model", markersize=8)
        plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Histogram of predictions
        plt.subplot(2, 2, 2)
        plt.hist(y_prob, bins=20, alpha=0.7, density=True)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title("Distribution of Predictions")
        plt.grid(True, alpha=0.3)
        
        # Calibration error by bin
        plt.subplot(2, 2, 3)
        calibration_errors = np.abs(fraction_pos - mean_pred)
        plt.bar(range(len(calibration_errors)), calibration_errors, alpha=0.7)
        plt.xlabel("Bin")
        plt.ylabel("Calibration Error")
        plt.title("Calibration Error by Bin")
        plt.grid(True, alpha=0.3)
        
        # ROC-like curve for calibration
        plt.subplot(2, 2, 4)
        sorted_indices = np.argsort(y_prob)
        cumulative_positives = np.cumsum(y_true[sorted_indices])
        total_positives = np.sum(y_true)
        cumulative_rate = cumulative_positives / total_positives if total_positives > 0 else np.zeros_like(cumulative_positives)
        
        plt.plot(np.linspace(0, 1, len(cumulative_rate)), cumulative_rate, label="Cumulative Positive Rate")
        plt.plot([0, 1], [0, 1], "k:", label="Random")
        plt.xlabel("Fraction of Samples")
        plt.ylabel("Cumulative Positive Rate")
        plt.title("Cumulative Gain")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        # Calculate metrics
        metrics = self.calculate_calibration_metrics(y_true, y_prob, fraction_pos, mean_pred)
        return metrics
        
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    fraction_pos: np.ndarray, mean_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive calibration metrics"""
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
        
        # Brier Score
        brier = brier_score_loss(y_true, y_prob)
        
        # Log Loss
        log_loss_val = log_loss(y_true, y_prob)
        
        # Reliability Score (correlation between predicted and actual)
        reliability = np.corrcoef(mean_pred, fraction_pos)[0, 1] if len(mean_pred) > 1 else 0.0
        
        # Calibration slope and intercept
        if len(mean_pred) > 1:
            slope, intercept, _, _, _ = stats.linregress(mean_pred, fraction_pos)
        else:
            slope, intercept = 0.0, 0.0
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'log_loss': log_loss_val,
            'reliability': reliability,
            'calibration_slope': slope,
            'calibration_intercept': intercept
        }

class IsotonicCalibrator:
    """Isotonic regression calibration"""
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'IsotonicCalibrator':
        """Fit isotonic calibration"""
        self.calibrator.fit(y_prob, y_true)
        self.is_fitted = True
        return self
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        return self.calibrator.predict(y_prob)

class PlattCalibrator:
    """Platt scaling calibration using logistic regression"""
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'PlattCalibrator':
        """Fit Platt scaling"""
        # Reshape for sklearn
        X = y_prob.reshape(-1, 1)
        self.calibrator.fit(X, y_true)
        self.is_fitted = True
        return self
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt scaling"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        X = y_prob.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]

class BetaCalibrator:
    """Beta calibration for improved reliability"""
    
    def __init__(self):
        self.a = 1.0
        self.b = 1.0
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'BetaCalibrator':
        """Fit beta calibration parameters"""
        
        def negative_log_likelihood(params):
            a, b = params
            if a <= 0 or b <= 0:
                return np.inf
                
            # Beta distribution likelihood
            calibrated_probs = self._beta_calibration(y_prob, a, b)
            
            # Avoid log(0)
            calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)
            
            # Negative log likelihood
            nll = -np.sum(y_true * np.log(calibrated_probs) + 
                         (1 - y_true) * np.log(1 - calibrated_probs))
            return nll
        
        # Optimize parameters
        result = minimize(negative_log_likelihood, [1.0, 1.0], 
                         method='L-BFGS-B', bounds=[(0.1, 10), (0.1, 10)])
        
        if result.success:
            self.a, self.b = result.x
        else:
            logger.warning("Beta calibration optimization failed, using default parameters")
            
        self.is_fitted = True
        return self
        
    def _beta_calibration(self, y_prob: np.ndarray, a: float, b: float) -> np.ndarray:
        """Apply beta calibration transformation"""
        return y_prob ** a / (y_prob ** a + (1 - y_prob) ** b)
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply beta calibration"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        return self._beta_calibration(y_prob, self.a, self.b)

class TemperatureScaling:
    """Temperature scaling calibration"""
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, logits: np.ndarray) -> 'TemperatureScaling':
        """Fit temperature scaling parameter"""
        
        def negative_log_likelihood(temperature):
            if temperature <= 0:
                return np.inf
                
            # Apply temperature scaling
            scaled_logits = logits / temperature
            probs = self._sigmoid(scaled_logits)
            
            # Avoid log(0)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            
            # Negative log likelihood
            nll = -np.sum(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            return nll
        
        # Optimize temperature
        result = minimize(negative_log_likelihood, [1.0], 
                         method='L-BFGS-B', bounds=[(0.1, 10)])
        
        if result.success:
            self.temperature = result.x[0]
        else:
            logger.warning("Temperature scaling optimization failed, using default temperature")
            
        self.is_fitted = True
        return self
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        scaled_logits = logits / self.temperature
        return self._sigmoid(scaled_logits)

class ConfidenceCalibrationFramework:
    """Main framework for confidence calibration"""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.calibrator = None
        self.reliability_diagram = ReliabilityDiagram(config.n_bins)
        self.calibration_metrics = None
        
    def create_calibrator(self, method: str):
        """Create calibrator based on method"""
        if method == 'isotonic':
            return IsotonicCalibrator()
        elif method == 'platt':
            return PlattCalibrator()
        elif method == 'beta':
            return BetaCalibrator()
        elif method == 'temperature':
            return TemperatureScaling()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
    def fit_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       logits: np.ndarray = None) -> CalibrationMetrics:
        """Fit calibration model"""
        logger.info(f"Fitting {self.config.calibration_method} calibration...")
        
        # Split data for calibration
        if len(y_true) < 100:
            logger.warning("Small dataset for calibration. Results may be unreliable.")
            
        X_cal, X_test, y_cal, y_test = train_test_split(
            y_prob, y_true, test_size=self.config.validation_split, random_state=42
        )
        
        # Create and fit calibrator
        self.calibrator = self.create_calibrator(self.config.calibration_method)
        
        if self.config.calibration_method == 'temperature' and logits is not None:
            logits_cal, logits_test = train_test_split(
                logits, test_size=self.config.validation_split, random_state=42
            )
            self.calibrator.fit(y_cal, logits_cal)
            calibrated_probs = self.calibrator.predict_proba(logits_test)
        else:
            self.calibrator.fit(y_cal, X_cal)
            calibrated_probs = self.calibrator.predict_proba(X_test)
        
        # Calculate metrics
        metrics_dict = self.reliability_diagram.calculate_calibration_metrics(
            y_test, calibrated_probs, 
            *self.reliability_diagram.compute_calibration_curve(y_test, calibrated_probs)
        )
        
        self.calibration_metrics = CalibrationMetrics(
            reliability_score=metrics_dict['reliability'],
            brier_score=metrics_dict['brier_score'],
            log_loss_score=metrics_dict['log_loss'],
            ece_score=metrics_dict['ece'],
            mce_score=metrics_dict['mce'],
            calibration_slope=metrics_dict['calibration_slope'],
            calibration_intercept=metrics_dict['calibration_intercept']
        )
        
        logger.info(f"Calibration fitted. ECE: {self.calibration_metrics.ece_score:.4f}, "
                   f"Brier: {self.calibration_metrics.brier_score:.4f}")
        
        return self.calibration_metrics
        
    def calibrate_predictions(self, y_prob: np.ndarray, logits: np.ndarray = None) -> np.ndarray:
        """Apply calibration to new predictions"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit_calibration first.")
            
        if self.config.calibration_method == 'temperature' and logits is not None:
            return self.calibrator.predict_proba(logits)
        else:
            return self.calibrator.predict_proba(y_prob)
            
    def evaluate_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, 
                           title: str = "Calibration Evaluation") -> Dict[str, Any]:
        """Evaluate calibration performance"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit_calibration first.")
            
        # Apply calibration
        calibrated_probs = self.calibrate_predictions(y_prob)
        
        # Generate reliability diagram
        metrics = self.reliability_diagram.plot_reliability_diagram(
            y_true, calibrated_probs, title=title
        )
        
        # Compare before and after
        uncalibrated_metrics = self.reliability_diagram.calculate_calibration_metrics(
            y_true, y_prob,
            *self.reliability_diagram.compute_calibration_curve(y_true, y_prob)
        )
        
        return {
            'uncalibrated_metrics': uncalibrated_metrics,
            'calibrated_metrics': metrics,
            'improvement': {
                'ece_improvement': uncalibrated_metrics['ece'] - metrics['ece'],
                'brier_improvement': uncalibrated_metrics['brier_score'] - metrics['brier_score'],
                'reliability_improvement': metrics['reliability'] - uncalibrated_metrics['reliability']
            }
        }
        
    def cross_validate_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Cross-validate calibration performance"""
        logger.info("Cross-validating calibration...")
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
        
        ece_scores = []
        brier_scores = []
        reliability_scores = []
        
        for train_idx, val_idx in kf.split(y_prob):
            # Split data
            y_train, y_val = y_true[train_idx], y_true[val_idx]
            prob_train, prob_val = y_prob[train_idx], y_prob[val_idx]
            
            # Fit calibrator
            calibrator = self.create_calibrator(self.config.calibration_method)
            calibrator.fit(y_train, prob_train)
            
            # Calibrate validation set
            calibrated_val = calibrator.predict_proba(prob_val)
            
            # Calculate metrics
            metrics = self.reliability_diagram.calculate_calibration_metrics(
                y_val, calibrated_val,
                *self.reliability_diagram.compute_calibration_curve(y_val, calibrated_val)
            )
            
            ece_scores.append(metrics['ece'])
            brier_scores.append(metrics['brier_score'])
            reliability_scores.append(metrics['reliability'])
            
        return {
            'mean_ece': np.mean(ece_scores),
            'std_ece': np.std(ece_scores),
            'mean_brier': np.mean(brier_scores),
            'std_brier': np.std(brier_scores),
            'mean_reliability': np.mean(reliability_scores),
            'std_reliability': np.std(reliability_scores)
        }
        
    def save_calibrator(self, filepath: str):
        """Save calibration model"""
        if self.calibrator is None:
            raise ValueError("No calibrator to save")
            
        calibration_data = {
            'calibrator': self.calibrator,
            'config': self.config,
            'metrics': self.calibration_metrics,
            'method': self.config.calibration_method,
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(calibration_data, filepath)
        logger.info(f"Calibrator saved to {filepath}")
        
    def load_calibrator(self, filepath: str):
        """Load calibration model"""
        calibration_data = joblib.load(filepath)
        
        self.calibrator = calibration_data['calibrator']
        self.config = calibration_data['config']
        self.calibration_metrics = calibration_data['metrics']
        
        logger.info(f"Calibrator loaded from {filepath}")

def compare_calibration_methods(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Dict]:
    """Compare different calibration methods"""
    methods = ['isotonic', 'platt', 'beta']
    results = {}
    
    for method in methods:
        logger.info(f"Testing {method} calibration...")
        
        config = CalibrationConfig(calibration_method=method)
        framework = ConfidenceCalibrationFramework(config)
        
        try:
            metrics = framework.fit_calibration(y_true, y_prob)
            cv_results = framework.cross_validate_calibration(y_true, y_prob)
            
            results[method] = {
                'metrics': metrics.__dict__,
                'cv_results': cv_results,
                'success': True
            }
        except Exception as e:
            logger.error(f"Failed to test {method}: {e}")
            results[method] = {'success': False, 'error': str(e)}
            
    return results

def main():
    """Main calibration testing function"""
    # Generate sample data with miscalibrated predictions
    np.random.seed(42)
    n_samples = 1000
    
    # True probabilities
    true_probs = np.random.beta(2, 5, n_samples)
    
    # Generate outcomes
    y_true = np.random.binomial(1, true_probs)
    
    # Generate miscalibrated predictions (overconfident)
    y_prob = np.clip(true_probs * 1.5, 0, 1)  # Make overconfident
    
    print("Testing confidence calibration framework...")
    print(f"Dataset size: {n_samples}")
    print(f"Positive rate: {np.mean(y_true):.3f}")
    
    # Compare calibration methods
    results = compare_calibration_methods(y_true, y_prob)
    
    print("\nCalibration Method Comparison:")
    for method, result in results.items():
        if result['success']:
            ece = result['metrics']['ece_score']
            brier = result['metrics']['brier_score']
            reliability = result['metrics']['reliability_score']
            print(f"{method.capitalize()}: ECE={ece:.4f}, Brier={brier:.4f}, Reliability={reliability:.4f}")
        else:
            print(f"{method.capitalize()}: Failed - {result['error']}")
    
    # Detailed analysis with best method
    best_method = min([m for m in results.keys() if results[m]['success']], 
                     key=lambda m: results[m]['metrics']['ece_score'])
    
    print(f"\nDetailed analysis with best method: {best_method}")
    
    config = CalibrationConfig(calibration_method=best_method)
    framework = ConfidenceCalibrationFramework(config)
    
    # Fit and evaluate
    metrics = framework.fit_calibration(y_true, y_prob)
    evaluation = framework.evaluate_calibration(y_true, y_prob, 
                                              title=f"{best_method.capitalize()} Calibration")
    
    print(f"ECE improvement: {evaluation['improvement']['ece_improvement']:.4f}")
    print(f"Brier improvement: {evaluation['improvement']['brier_improvement']:.4f}")
    print(f"Reliability improvement: {evaluation['improvement']['reliability_improvement']:.4f}")
    
    # Save calibrator
    framework.save_calibrator("./ai-models/models/confidence-calibration/calibrator.pkl")
    print("Calibrator saved successfully")

if __name__ == "__main__":
    main()

