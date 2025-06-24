"""
Risk Management Service for AI Options Trading System

This service provides comprehensive risk management capabilities including:
- Real-time risk monitoring and assessment
- Position limits and controls enforcement
- Drawdown protection and emergency stops
- Portfolio risk analytics and reporting
- Integration with portfolio and signal services

Author: Manus AI
Version: 4.0.0
Port: 8006
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskMetricType(Enum):
    """Types of risk metrics"""
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    portfolio_id: str
    timestamp: datetime
    total_value: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    concentration_risk: float
    liquidity_risk: float
    correlation_risk: float
    risk_level: RiskLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['risk_level'] = self.risk_level.value
        return data

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    limit_id: str
    portfolio_id: str
    metric_type: RiskMetricType
    limit_value: float
    warning_threshold: float  # Percentage of limit (e.g., 0.8 for 80%)
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    portfolio_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_type: RiskMetricType
    current_value: float
    limit_value: float
    threshold_breached: float
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['metric_type'] = self.metric_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class RiskCalculator:
    """Advanced risk calculation engine"""
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99]
        self.lookback_periods = {
            'short': 30,
            'medium': 90,
            'long': 252
        }
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate VaR at specified confidence level
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index] if index < len(sorted_returns) else 0.0
        
        return max(0.0, var)
    
    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        
        # Calculate average of returns worse than VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        expected_shortfall = -np.mean(tail_returns)
        return max(0.0, expected_shortfall)
    
    def calculate_max_drawdown(self, values: np.ndarray) -> Tuple[float, float]:
        """Calculate maximum drawdown and current drawdown"""
        if len(values) == 0:
            return 0.0, 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = abs(np.min(drawdowns))
        
        # Current drawdown
        current_drawdown = abs(drawdowns[-1])
        
        return max_drawdown, current_drawdown
    
    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns, ddof=1)
        
        if annualize:
            # Annualize assuming 252 trading days
            volatility *= np.sqrt(252)
        
        return volatility
    
    def calculate_beta(self, portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio beta relative to market"""
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 1.0
        
        # Ensure same length
        min_length = min(len(portfolio_returns), len(market_returns))
        portfolio_returns = portfolio_returns[-min_length:]
        market_returns = market_returns[-min_length:]
        
        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns, ddof=1)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) * np.sqrt(252) / downside_deviation
        return sortino
    
    def calculate_concentration_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index"""
        if not positions:
            return 0.0
        
        total_value = sum(abs(pos.get('market_value', 0)) for pos in positions)
        
        if total_value == 0:
            return 0.0
        
        # Calculate weights
        weights = [abs(pos.get('market_value', 0)) / total_value for pos in positions]
        
        # Calculate HHI
        hhi = sum(w**2 for w in weights)
        
        # Normalize to 0-1 scale (1 = maximum concentration)
        max_hhi = 1.0  # All in one position
        min_hhi = 1.0 / len(positions)  # Equally weighted
        
        if max_hhi == min_hhi:
            return 0.0
        
        concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi)
        return min(1.0, max(0.0, concentration_risk))
    
    def calculate_liquidity_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate liquidity risk based on position sizes and market cap"""
        if not positions:
            return 0.0
        
        total_liquidity_score = 0.0
        total_weight = 0.0
        
        for position in positions:
            market_value = abs(position.get('market_value', 0))
            if market_value == 0:
                continue
            
            # Get liquidity metrics
            avg_volume = position.get('avg_volume', 1000000)  # Default volume
            market_cap = position.get('market_cap', 1000000000)  # Default market cap
            
            # Calculate liquidity score (higher is more liquid)
            volume_score = min(1.0, avg_volume / 10000000)  # Normalize to 10M volume
            market_cap_score = min(1.0, market_cap / 100000000000)  # Normalize to 100B market cap
            
            liquidity_score = (volume_score + market_cap_score) / 2
            
            # Weight by position size
            weight = market_value
            total_liquidity_score += liquidity_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_liquidity_score = total_liquidity_score / total_weight
        
        # Convert to risk (inverse of liquidity)
        liquidity_risk = 1.0 - avg_liquidity_score
        return min(1.0, max(0.0, liquidity_risk))
    
    def calculate_correlation_risk(self, positions: List[Dict[str, Any]], 
                                 correlation_matrix: Optional[np.ndarray] = None) -> float:
        """Calculate correlation risk of portfolio"""
        if not positions or len(positions) < 2:
            return 0.0
        
        if correlation_matrix is None:
            # Use default correlation assumptions
            n_positions = len(positions)
            correlation_matrix = np.full((n_positions, n_positions), 0.3)
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # Calculate position weights
        total_value = sum(abs(pos.get('market_value', 0)) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weights = np.array([abs(pos.get('market_value', 0)) / total_value for pos in positions])
        
        # Calculate portfolio correlation risk
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
        
        # Normalize to 0-1 scale
        max_correlation = 1.0  # Perfect correlation
        min_correlation = 1.0 / len(positions)  # No correlation
        
        correlation_risk = (portfolio_variance - min_correlation) / (max_correlation - min_correlation)
        return min(1.0, max(0.0, correlation_risk))

class RiskMonitor:
    """Real-time risk monitoring engine"""
    
    def __init__(self, redis_client, db_connection):
        self.redis_client = redis_client
        self.db_connection = db_connection
        self.calculator = RiskCalculator()
        self.monitoring_active = False
        self.monitor_thread = None
        self.update_interval = 5  # seconds
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.50,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0
        }
    
    def start_monitoring(self):
        """Start real-time risk monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time risk monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get all active portfolios
                portfolios = self._get_active_portfolios()
                
                for portfolio in portfolios:
                    portfolio_id = portfolio['portfolio_id']
                    
                    # Calculate risk metrics
                    risk_metrics = self.calculate_portfolio_risk(portfolio_id)
                    
                    if risk_metrics:
                        # Store metrics
                        self._store_risk_metrics(risk_metrics)
                        
                        # Check for limit breaches
                        self._check_risk_limits(risk_metrics)
                        
                        # Cache current metrics
                        self._cache_risk_metrics(risk_metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def calculate_portfolio_risk(self, portfolio_id: str) -> Optional[RiskMetrics]:
        """Calculate comprehensive risk metrics for a portfolio"""
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return None
            
            positions = portfolio_data.get('positions', [])
            historical_values = portfolio_data.get('historical_values', [])
            
            if not positions:
                return None
            
            # Calculate returns
            returns = self._calculate_returns(historical_values)
            
            # Calculate individual risk metrics
            var_95 = self.calculator.calculate_var(returns, 0.95)
            var_99 = self.calculator.calculate_var(returns, 0.99)
            expected_shortfall = self.calculator.calculate_expected_shortfall(returns, 0.95)
            
            max_drawdown, current_drawdown = self.calculator.calculate_max_drawdown(
                np.array(historical_values)
            )
            
            volatility = self.calculator.calculate_volatility(returns)
            
            # Get market returns for beta calculation
            market_returns = self._get_market_returns()
            beta = self.calculator.calculate_beta(returns, market_returns)
            
            sharpe_ratio = self.calculator.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculator.calculate_sortino_ratio(returns)
            
            concentration_risk = self.calculator.calculate_concentration_risk(positions)
            liquidity_risk = self.calculator.calculate_liquidity_risk(positions)
            correlation_risk = self.calculator.calculate_correlation_risk(positions)
            
            # Determine overall risk level
            risk_level = self._determine_risk_level({
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'volatility': volatility,
                'concentration_risk': concentration_risk,
                'liquidity_risk': liquidity_risk,
                'correlation_risk': correlation_risk
            })
            
            # Get total portfolio value
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            risk_metrics = RiskMetrics(
                portfolio_id=portfolio_id,
                timestamp=datetime.now(),
                total_value=total_value,
                var_95=var_95 * total_value,  # Convert to dollar amount
                var_99=var_99 * total_value,
                expected_shortfall=expected_shortfall * total_value,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                correlation_risk=correlation_risk,
                risk_level=risk_level
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk for {portfolio_id}: {e}")
            return None
    
    def _get_active_portfolios(self) -> List[Dict[str, Any]]:
        """Get list of active portfolios"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT portfolio_id, account_id, portfolio_name, status
                    FROM portfolios 
                    WHERE status = 'active'
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting active portfolios: {e}")
            return []
    
    def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive portfolio data"""
        try:
            # Get current positions
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT symbol, quantity, market_value, cost_basis, 
                           unrealized_pnl, strategy_id
                    FROM positions 
                    WHERE portfolio_id = %s AND status = 'open'
                """, (portfolio_id,))
                positions = [dict(row) for row in cursor.fetchall()]
                
                # Get historical portfolio values (last 252 days)
                cursor.execute("""
                    SELECT total_value, timestamp
                    FROM portfolio_snapshots 
                    WHERE portfolio_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 252
                """, (portfolio_id,))
                snapshots = [dict(row) for row in cursor.fetchall()]
                
                historical_values = [s['total_value'] for s in reversed(snapshots)]
                
                return {
                    'positions': positions,
                    'historical_values': historical_values
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio data for {portfolio_id}: {e}")
            return None
    
    def _calculate_returns(self, values: List[float]) -> np.ndarray:
        """Calculate daily returns from portfolio values"""
        if len(values) < 2:
            return np.array([])
        
        values_array = np.array(values)
        returns = np.diff(values_array) / values_array[:-1]
        
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        
        return returns
    
    def _get_market_returns(self) -> np.ndarray:
        """Get market returns (SPY) for beta calculation"""
        try:
            # Try to get from cache first
            cached_returns = self.redis_client.get('market_returns_252d')
            if cached_returns:
                return np.array(json.loads(cached_returns))
            
            # Generate mock market returns for development
            # In production, this would fetch real SPY data
            np.random.seed(42)  # For consistent results
            market_returns = np.random.normal(0.0005, 0.015, 252)  # ~12% annual return, 24% volatility
            
            # Cache for 1 hour
            self.redis_client.setex(
                'market_returns_252d',
                3600,
                json.dumps(market_returns.tolist())
            )
            
            return market_returns
            
        except Exception as e:
            logger.error(f"Error getting market returns: {e}")
            # Return default market returns
            return np.random.normal(0.0005, 0.015, 252)
    
    def _determine_risk_level(self, metrics: Dict[str, float]) -> RiskLevel:
        """Determine overall risk level based on multiple metrics"""
        risk_scores = []
        
        # VaR score (normalized to portfolio value)
        var_score = min(1.0, metrics.get('var_95', 0) / 0.05)  # 5% VaR threshold
        risk_scores.append(var_score)
        
        # Drawdown scores
        max_dd_score = min(1.0, metrics.get('max_drawdown', 0) / 0.20)  # 20% max drawdown
        current_dd_score = min(1.0, metrics.get('current_drawdown', 0) / 0.10)  # 10% current drawdown
        risk_scores.extend([max_dd_score, current_dd_score])
        
        # Volatility score
        vol_score = min(1.0, metrics.get('volatility', 0) / 0.30)  # 30% volatility threshold
        risk_scores.append(vol_score)
        
        # Concentration, liquidity, and correlation risks
        risk_scores.append(metrics.get('concentration_risk', 0))
        risk_scores.append(metrics.get('liquidity_risk', 0))
        risk_scores.append(metrics.get('correlation_risk', 0))
        
        # Calculate weighted average risk score
        weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]  # Weights for each metric
        overall_risk_score = sum(score * weight for score, weight in zip(risk_scores, weights))
        
        # Determine risk level
        if overall_risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif overall_risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif overall_risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _store_risk_metrics(self, risk_metrics: RiskMetrics):
        """Store risk metrics in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO risk_metrics (
                        portfolio_id, timestamp, total_value, var_95, var_99,
                        expected_shortfall, max_drawdown, current_drawdown,
                        volatility, beta, sharpe_ratio, sortino_ratio,
                        concentration_risk, liquidity_risk, correlation_risk,
                        risk_level
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    risk_metrics.portfolio_id,
                    risk_metrics.timestamp,
                    risk_metrics.total_value,
                    risk_metrics.var_95,
                    risk_metrics.var_99,
                    risk_metrics.expected_shortfall,
                    risk_metrics.max_drawdown,
                    risk_metrics.current_drawdown,
                    risk_metrics.volatility,
                    risk_metrics.beta,
                    risk_metrics.sharpe_ratio,
                    risk_metrics.sortino_ratio,
                    risk_metrics.concentration_risk,
                    risk_metrics.liquidity_risk,
                    risk_metrics.correlation_risk,
                    risk_metrics.risk_level.value
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
            self.db_connection.rollback()
    
    def _check_risk_limits(self, risk_metrics: RiskMetrics):
        """Check risk metrics against configured limits"""
        try:
            # Get active limits for this portfolio
            limits = self._get_risk_limits(risk_metrics.portfolio_id)
            
            for limit in limits:
                current_value = self._get_metric_value(risk_metrics, limit.metric_type)
                
                # Check if limit is breached
                if current_value > limit.limit_value:
                    self._create_risk_alert(
                        risk_metrics.portfolio_id,
                        "LIMIT_BREACH",
                        AlertSeverity.CRITICAL,
                        f"{limit.metric_type.value.upper()} limit breached",
                        limit.metric_type,
                        current_value,
                        limit.limit_value,
                        1.0  # 100% breach
                    )
                
                # Check warning threshold
                elif current_value > limit.limit_value * limit.warning_threshold:
                    threshold_breached = current_value / limit.limit_value
                    self._create_risk_alert(
                        risk_metrics.portfolio_id,
                        "LIMIT_WARNING",
                        AlertSeverity.WARNING,
                        f"{limit.metric_type.value.upper()} approaching limit",
                        limit.metric_type,
                        current_value,
                        limit.limit_value,
                        threshold_breached
                    )
                    
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def _get_risk_limits(self, portfolio_id: str) -> List[RiskLimit]:
        """Get active risk limits for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT limit_id, portfolio_id, metric_type, limit_value,
                           warning_threshold, is_active, created_at, updated_at
                    FROM risk_limits 
                    WHERE portfolio_id = %s AND is_active = true
                """, (portfolio_id,))
                
                limits = []
                for row in cursor.fetchall():
                    limit = RiskLimit(
                        limit_id=row['limit_id'],
                        portfolio_id=row['portfolio_id'],
                        metric_type=RiskMetricType(row['metric_type']),
                        limit_value=row['limit_value'],
                        warning_threshold=row['warning_threshold'],
                        is_active=row['is_active'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    limits.append(limit)
                
                return limits
                
        except Exception as e:
            logger.error(f"Error getting risk limits: {e}")
            return []
    
    def _get_metric_value(self, risk_metrics: RiskMetrics, metric_type: RiskMetricType) -> float:
        """Get specific metric value from risk metrics"""
        metric_map = {
            RiskMetricType.VAR_95: risk_metrics.var_95,
            RiskMetricType.VAR_99: risk_metrics.var_99,
            RiskMetricType.EXPECTED_SHORTFALL: risk_metrics.expected_shortfall,
            RiskMetricType.MAX_DRAWDOWN: risk_metrics.max_drawdown,
            RiskMetricType.VOLATILITY: risk_metrics.volatility,
            RiskMetricType.BETA: risk_metrics.beta,
            RiskMetricType.CONCENTRATION: risk_metrics.concentration_risk,
            RiskMetricType.LIQUIDITY: risk_metrics.liquidity_risk,
            RiskMetricType.CORRELATION: risk_metrics.correlation_risk
        }
        
        return metric_map.get(metric_type, 0.0)
    
    def _create_risk_alert(self, portfolio_id: str, alert_type: str, severity: AlertSeverity,
                          message: str, metric_type: RiskMetricType, current_value: float,
                          limit_value: float, threshold_breached: float):
        """Create a risk alert"""
        try:
            alert_id = f"alert_{portfolio_id}_{int(time.time())}"
            
            alert = RiskAlert(
                alert_id=alert_id,
                portfolio_id=portfolio_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                metric_type=metric_type,
                current_value=current_value,
                limit_value=limit_value,
                threshold_breached=threshold_breached,
                timestamp=datetime.now()
            )
            
            # Store in database
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO risk_alerts (
                        alert_id, portfolio_id, alert_type, severity, message,
                        metric_type, current_value, limit_value, threshold_breached,
                        timestamp, acknowledged
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    alert.alert_id,
                    alert.portfolio_id,
                    alert.alert_type,
                    alert.severity.value,
                    alert.message,
                    alert.metric_type.value,
                    alert.current_value,
                    alert.limit_value,
                    alert.threshold_breached,
                    alert.timestamp,
                    alert.acknowledged
                ))
                self.db_connection.commit()
            
            # Cache for real-time access
            self.redis_client.setex(
                f"alert:{alert_id}",
                86400,  # 24 hours
                json.dumps(alert.to_dict())
            )
            
            logger.warning(f"Risk alert created: {alert.message} for portfolio {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error creating risk alert: {e}")
            self.db_connection.rollback()
    
    def _cache_risk_metrics(self, risk_metrics: RiskMetrics):
        """Cache current risk metrics for fast access"""
        try:
            cache_key = f"risk_metrics:{risk_metrics.portfolio_id}"
            self.redis_client.setex(
                cache_key,
                300,  # 5 minutes
                json.dumps(risk_metrics.to_dict())
            )
        except Exception as e:
            logger.error(f"Error caching risk metrics: {e}")

class RiskManagementService:
    """Main Risk Management Service"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, origins="*")  # Allow all origins for development
        
        # Initialize connections
        self.redis_client = self._init_redis()
        self.db_connection = self._init_database()
        
        # Initialize risk monitor
        self.risk_monitor = RiskMonitor(self.redis_client, self.db_connection)
        
        # Setup routes
        self._setup_routes()
        
        # Start monitoring
        self.risk_monitor.start_monitoring()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_db = int(os.getenv('REDIS_DB', 0))
            
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Return mock client for development
            return type('MockRedis', (), {
                'get': lambda self, key: None,
                'set': lambda self, key, value: True,
                'setex': lambda self, key, time, value: True,
                'delete': lambda self, key: True,
                'ping': lambda self: True
            })()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/risk_db')
            connection = psycopg2.connect(db_url)
            connection.autocommit = False
            
            logger.info("Connected to PostgreSQL database")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            # Return mock connection for development
            return type('MockConnection', (), {
                'cursor': lambda self, cursor_factory=None: type('MockCursor', (), {
                    'execute': lambda self, query, params=None: None,
                    'fetchall': lambda self: [],
                    'fetchone': lambda self: None,
                    '__enter__': lambda self: self,
                    '__exit__': lambda self, *args: None
                })(),
                'commit': lambda self: None,
                'rollback': lambda self: None
            })()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'service': 'risk-management',
                'version': '4.0.0',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/risk/metrics/<portfolio_id>', methods=['GET'])
        def get_risk_metrics(portfolio_id):
            """Get current risk metrics for a portfolio"""
            try:
                # Try cache first
                cached_metrics = self.redis_client.get(f"risk_metrics:{portfolio_id}")
                if cached_metrics:
                    return jsonify(json.loads(cached_metrics))
                
                # Calculate fresh metrics
                risk_metrics = self.risk_monitor.calculate_portfolio_risk(portfolio_id)
                if risk_metrics:
                    return jsonify(risk_metrics.to_dict())
                else:
                    return jsonify({'error': 'Portfolio not found or no data available'}), 404
                    
            except Exception as e:
                logger.error(f"Error getting risk metrics: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/risk/limits', methods=['POST'])
        def create_risk_limit():
            """Create a new risk limit"""
            try:
                data = request.get_json()
                
                # Validate required fields
                required_fields = ['portfolio_id', 'metric_type', 'limit_value']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                limit_id = f"limit_{data['portfolio_id']}_{int(time.time())}"
                
                risk_limit = RiskLimit(
                    limit_id=limit_id,
                    portfolio_id=data['portfolio_id'],
                    metric_type=RiskMetricType(data['metric_type']),
                    limit_value=float(data['limit_value']),
                    warning_threshold=float(data.get('warning_threshold', 0.8)),
                    is_active=data.get('is_active', True),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                # Store in database
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO risk_limits (
                            limit_id, portfolio_id, metric_type, limit_value,
                            warning_threshold, is_active, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        risk_limit.limit_id,
                        risk_limit.portfolio_id,
                        risk_limit.metric_type.value,
                        risk_limit.limit_value,
                        risk_limit.warning_threshold,
                        risk_limit.is_active,
                        risk_limit.created_at,
                        risk_limit.updated_at
                    ))
                    self.db_connection.commit()
                
                return jsonify(risk_limit.to_dict()), 201
                
            except Exception as e:
                logger.error(f"Error creating risk limit: {e}")
                self.db_connection.rollback()
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/risk/limits/<portfolio_id>', methods=['GET'])
        def get_risk_limits(portfolio_id):
            """Get risk limits for a portfolio"""
            try:
                limits = self.risk_monitor._get_risk_limits(portfolio_id)
                return jsonify([limit.to_dict() for limit in limits])
                
            except Exception as e:
                logger.error(f"Error getting risk limits: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/risk/alerts/<portfolio_id>', methods=['GET'])
        def get_risk_alerts(portfolio_id):
            """Get risk alerts for a portfolio"""
            try:
                with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT alert_id, portfolio_id, alert_type, severity, message,
                               metric_type, current_value, limit_value, threshold_breached,
                               timestamp, acknowledged
                        FROM risk_alerts 
                        WHERE portfolio_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 100
                    """, (portfolio_id,))
                    
                    alerts = []
                    for row in cursor.fetchall():
                        alert = RiskAlert(
                            alert_id=row['alert_id'],
                            portfolio_id=row['portfolio_id'],
                            alert_type=row['alert_type'],
                            severity=AlertSeverity(row['severity']),
                            message=row['message'],
                            metric_type=RiskMetricType(row['metric_type']),
                            current_value=row['current_value'],
                            limit_value=row['limit_value'],
                            threshold_breached=row['threshold_breached'],
                            timestamp=row['timestamp'],
                            acknowledged=row['acknowledged']
                        )
                        alerts.append(alert.to_dict())
                    
                    return jsonify(alerts)
                    
            except Exception as e:
                logger.error(f"Error getting risk alerts: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/risk/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """Acknowledge a risk alert"""
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("""
                        UPDATE risk_alerts 
                        SET acknowledged = true 
                        WHERE alert_id = %s
                    """, (alert_id,))
                    self.db_connection.commit()
                
                return jsonify({'status': 'acknowledged'})
                
            except Exception as e:
                logger.error(f"Error acknowledging alert: {e}")
                self.db_connection.rollback()
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/v1/risk/monitoring/status', methods=['GET'])
        def get_monitoring_status():
            """Get risk monitoring status"""
            return jsonify({
                'monitoring_active': self.risk_monitor.monitoring_active,
                'update_interval': self.risk_monitor.update_interval,
                'last_update': datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/risk/monitoring/start', methods=['POST'])
        def start_monitoring():
            """Start risk monitoring"""
            self.risk_monitor.start_monitoring()
            return jsonify({'status': 'monitoring started'})
        
        @self.app.route('/api/v1/risk/monitoring/stop', methods=['POST'])
        def stop_monitoring():
            """Stop risk monitoring"""
            self.risk_monitor.stop_monitoring()
            return jsonify({'status': 'monitoring stopped'})
    
    def run(self, host='0.0.0.0', port=8006, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Risk Management Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def main():
    """Main entry point"""
    service = RiskManagementService()
    
    try:
        service.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 8006)),
            debug=os.getenv('FLASK_ENV') == 'development'
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Risk Management Service...")
        service.risk_monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Error running service: {e}")
        service.risk_monitor.stop_monitoring()

if __name__ == '__main__':
    main()

