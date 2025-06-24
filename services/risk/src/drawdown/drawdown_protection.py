"""
Drawdown Protection System
Advanced drawdown monitoring and emergency stop mechanisms

This module provides comprehensive drawdown protection including:
- Real-time drawdown monitoring
- Multi-level drawdown thresholds
- Automated emergency stops
- Portfolio protection mechanisms
- Recovery tracking and analysis

Author: Manus AI
Version: 4.0.0
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
import uuid
import math

import numpy as np
import pandas as pd
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"           # < 5% drawdown
    WARNING = "warning"         # 5-10% drawdown
    MODERATE = "moderate"       # 10-15% drawdown
    SEVERE = "severe"          # 15-20% drawdown
    CRITICAL = "critical"      # > 20% drawdown

class ProtectionAction(Enum):
    """Protection actions for drawdown events"""
    MONITOR = "monitor"                    # Monitor only
    ALERT = "alert"                       # Generate alerts
    REDUCE_RISK = "reduce_risk"           # Reduce portfolio risk
    HALT_NEW_TRADES = "halt_new_trades"   # Stop new trades
    REDUCE_POSITIONS = "reduce_positions" # Reduce position sizes
    EMERGENCY_STOP = "emergency_stop"     # Emergency stop all trading
    LIQUIDATE = "liquidate"               # Liquidate all positions

class DrawdownType(Enum):
    """Types of drawdown measurements"""
    ABSOLUTE = "absolute"       # Absolute dollar drawdown
    PERCENTAGE = "percentage"   # Percentage drawdown
    DAILY = "daily"            # Daily drawdown
    WEEKLY = "weekly"          # Weekly drawdown
    MONTHLY = "monthly"        # Monthly drawdown
    ROLLING = "rolling"        # Rolling period drawdown

@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown metrics"""
    portfolio_id: str
    timestamp: datetime
    current_value: float
    peak_value: float
    trough_value: float
    current_drawdown: float
    current_drawdown_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_duration: int  # Days in current drawdown
    max_drawdown_duration: int  # Days in worst drawdown
    recovery_factor: float
    drawdown_level: DrawdownLevel
    days_to_recovery: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['drawdown_level'] = self.drawdown_level.value
        return data

@dataclass
class DrawdownThreshold:
    """Drawdown threshold configuration"""
    threshold_id: str
    portfolio_id: str
    drawdown_type: DrawdownType
    threshold_value: float
    threshold_percentage: float
    protection_action: ProtectionAction
    is_active: bool
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['drawdown_type'] = self.drawdown_type.value
        data['protection_action'] = self.protection_action.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class DrawdownEvent:
    """Drawdown event record"""
    event_id: str
    portfolio_id: str
    event_type: str
    drawdown_level: DrawdownLevel
    peak_value: float
    trough_value: float
    drawdown_amount: float
    drawdown_percentage: float
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    recovery_days: Optional[int]
    protection_action: ProtectionAction
    action_taken: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['drawdown_level'] = self.drawdown_level.value
        data['protection_action'] = self.protection_action.value
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat() if self.end_date else None
        return data

class DrawdownCalculator:
    """Advanced drawdown calculation engine"""
    
    def __init__(self):
        self.lookback_periods = {
            'short': 30,
            'medium': 90,
            'long': 252
        }
    
    def calculate_drawdown_metrics(self, portfolio_values: List[float], 
                                 timestamps: List[datetime]) -> DrawdownMetrics:
        """Calculate comprehensive drawdown metrics"""
        if len(portfolio_values) < 2:
            return self._empty_metrics()
        
        values = np.array(portfolio_values)
        
        # Calculate running maximum (peaks)
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = values - running_max
        drawdown_percentages = drawdowns / running_max * 100
        
        # Current metrics
        current_value = values[-1]
        peak_value = running_max[-1]
        current_drawdown = drawdowns[-1]
        current_drawdown_pct = drawdown_percentages[-1]
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        max_drawdown_pct = np.min(drawdown_percentages)
        
        # Find trough value
        trough_idx = np.argmin(drawdowns)
        trough_value = values[trough_idx]
        
        # Calculate drawdown duration
        drawdown_duration = self._calculate_current_drawdown_duration(
            drawdowns, timestamps
        )
        
        # Calculate maximum drawdown duration
        max_drawdown_duration = self._calculate_max_drawdown_duration(
            drawdowns, timestamps
        )
        
        # Calculate recovery factor
        recovery_factor = self._calculate_recovery_factor(values, drawdowns)
        
        # Determine drawdown level
        drawdown_level = self._determine_drawdown_level(abs(current_drawdown_pct))
        
        # Estimate days to recovery
        days_to_recovery = self._estimate_recovery_days(
            current_drawdown_pct, values, timestamps
        )
        
        return DrawdownMetrics(
            portfolio_id="",  # Will be set by caller
            timestamp=timestamps[-1] if timestamps else datetime.now(),
            current_value=current_value,
            peak_value=peak_value,
            trough_value=trough_value,
            current_drawdown=abs(current_drawdown),
            current_drawdown_pct=abs(current_drawdown_pct),
            max_drawdown=abs(max_drawdown),
            max_drawdown_pct=abs(max_drawdown_pct),
            drawdown_duration=drawdown_duration,
            max_drawdown_duration=max_drawdown_duration,
            recovery_factor=recovery_factor,
            drawdown_level=drawdown_level,
            days_to_recovery=days_to_recovery
        )
    
    def _empty_metrics(self) -> DrawdownMetrics:
        """Return empty metrics for insufficient data"""
        return DrawdownMetrics(
            portfolio_id="",
            timestamp=datetime.now(),
            current_value=0.0,
            peak_value=0.0,
            trough_value=0.0,
            current_drawdown=0.0,
            current_drawdown_pct=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            drawdown_duration=0,
            max_drawdown_duration=0,
            recovery_factor=0.0,
            drawdown_level=DrawdownLevel.NORMAL
        )
    
    def _calculate_current_drawdown_duration(self, drawdowns: np.ndarray, 
                                           timestamps: List[datetime]) -> int:
        """Calculate duration of current drawdown in days"""
        if len(drawdowns) == 0 or drawdowns[-1] >= 0:
            return 0
        
        # Find start of current drawdown
        for i in range(len(drawdowns) - 1, -1, -1):
            if drawdowns[i] >= 0:
                start_idx = i + 1
                break
        else:
            start_idx = 0
        
        if start_idx >= len(timestamps):
            return 0
        
        start_date = timestamps[start_idx]
        end_date = timestamps[-1]
        
        return (end_date - start_date).days
    
    def _calculate_max_drawdown_duration(self, drawdowns: np.ndarray, 
                                       timestamps: List[datetime]) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(drawdowns) == 0:
            return 0
        
        max_duration = 0
        current_duration = 0
        start_idx = None
        
        for i, dd in enumerate(drawdowns):
            if dd < 0:
                if start_idx is None:
                    start_idx = i
                current_duration = i - start_idx + 1
            else:
                if start_idx is not None:
                    max_duration = max(max_duration, current_duration)
                    start_idx = None
                    current_duration = 0
        
        # Handle case where drawdown continues to end
        if start_idx is not None:
            max_duration = max(max_duration, current_duration)
        
        # Convert to days
        if max_duration > 0 and len(timestamps) > max_duration:
            days = (timestamps[max_duration - 1] - timestamps[0]).days
            return max(1, days)
        
        return max_duration
    
    def _calculate_recovery_factor(self, values: np.ndarray, 
                                 drawdowns: np.ndarray) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if len(values) < 2:
            return 0.0
        
        total_return = (values[-1] / values[0] - 1) * 100
        max_drawdown_pct = abs(np.min(drawdowns / np.maximum.accumulate(values) * 100))
        
        if max_drawdown_pct == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_drawdown_pct
    
    def _determine_drawdown_level(self, drawdown_pct: float) -> DrawdownLevel:
        """Determine drawdown severity level"""
        if drawdown_pct < 5.0:
            return DrawdownLevel.NORMAL
        elif drawdown_pct < 10.0:
            return DrawdownLevel.WARNING
        elif drawdown_pct < 15.0:
            return DrawdownLevel.MODERATE
        elif drawdown_pct < 20.0:
            return DrawdownLevel.SEVERE
        else:
            return DrawdownLevel.CRITICAL
    
    def _estimate_recovery_days(self, current_drawdown_pct: float, 
                              values: np.ndarray, timestamps: List[datetime]) -> Optional[int]:
        """Estimate days to recovery based on historical performance"""
        if current_drawdown_pct >= 0 or len(values) < 30:
            return None
        
        # Calculate historical daily returns
        returns = np.diff(values) / values[:-1]
        
        # Calculate average positive return
        positive_returns = returns[returns > 0]
        if len(positive_returns) == 0:
            return None
        
        avg_positive_return = np.mean(positive_returns)
        
        # Estimate days needed to recover
        recovery_needed = abs(current_drawdown_pct) / 100
        
        if avg_positive_return <= 0:
            return None
        
        estimated_days = math.ceil(recovery_needed / avg_positive_return)
        
        # Cap at reasonable maximum
        return min(estimated_days, 365)

class DrawdownMonitor:
    """Real-time drawdown monitoring system"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.calculator = DrawdownCalculator()
        self.monitoring_active = False
        self.monitor_thread = None
        self.update_interval = 30  # seconds
    
    def start_monitoring(self):
        """Start drawdown monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Drawdown monitoring started")
    
    def stop_monitoring(self):
        """Stop drawdown monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Drawdown monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get all active portfolios
                portfolios = self._get_active_portfolios()
                
                for portfolio in portfolios:
                    portfolio_id = portfolio['portfolio_id']
                    
                    # Calculate drawdown metrics
                    metrics = self.calculate_portfolio_drawdown(portfolio_id)
                    
                    if metrics:
                        # Store metrics
                        self._store_drawdown_metrics(metrics)
                        
                        # Check thresholds
                        self._check_drawdown_thresholds(metrics)
                        
                        # Cache metrics
                        self._cache_drawdown_metrics(metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in drawdown monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def calculate_portfolio_drawdown(self, portfolio_id: str) -> Optional[DrawdownMetrics]:
        """Calculate drawdown metrics for a portfolio"""
        try:
            # Get portfolio value history
            values, timestamps = self._get_portfolio_history(portfolio_id)
            
            if len(values) < 2:
                return None
            
            # Calculate metrics
            metrics = self.calculator.calculate_drawdown_metrics(values, timestamps)
            metrics.portfolio_id = portfolio_id
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio drawdown: {e}")
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
    
    def _get_portfolio_history(self, portfolio_id: str, days: int = 252) -> Tuple[List[float], List[datetime]]:
        """Get portfolio value history"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT total_value, timestamp
                    FROM portfolio_snapshots 
                    WHERE portfolio_id = %s 
                    AND timestamp >= %s
                    ORDER BY timestamp ASC
                """, (portfolio_id, datetime.now() - timedelta(days=days)))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return [], []
                
                values = [float(row['total_value']) for row in rows]
                timestamps = [row['timestamp'] for row in rows]
                
                return values, timestamps
                
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return [], []
    
    def _store_drawdown_metrics(self, metrics: DrawdownMetrics):
        """Store drawdown metrics in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO drawdown_metrics (
                        portfolio_id, timestamp, current_value, peak_value, trough_value,
                        current_drawdown, current_drawdown_pct, max_drawdown, max_drawdown_pct,
                        drawdown_duration, max_drawdown_duration, recovery_factor,
                        drawdown_level, days_to_recovery
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    metrics.portfolio_id,
                    metrics.timestamp,
                    metrics.current_value,
                    metrics.peak_value,
                    metrics.trough_value,
                    metrics.current_drawdown,
                    metrics.current_drawdown_pct,
                    metrics.max_drawdown,
                    metrics.max_drawdown_pct,
                    metrics.drawdown_duration,
                    metrics.max_drawdown_duration,
                    metrics.recovery_factor,
                    metrics.drawdown_level.value,
                    metrics.days_to_recovery
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing drawdown metrics: {e}")
            self.db_connection.rollback()
    
    def _check_drawdown_thresholds(self, metrics: DrawdownMetrics):
        """Check drawdown metrics against configured thresholds"""
        try:
            # Get active thresholds
            thresholds = self._get_drawdown_thresholds(metrics.portfolio_id)
            
            for threshold in thresholds:
                if self._is_threshold_breached(metrics, threshold):
                    self._trigger_protection_action(metrics, threshold)
                    
        except Exception as e:
            logger.error(f"Error checking drawdown thresholds: {e}")
    
    def _get_drawdown_thresholds(self, portfolio_id: str) -> List[DrawdownThreshold]:
        """Get active drawdown thresholds for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT threshold_id, portfolio_id, drawdown_type, threshold_value,
                           threshold_percentage, protection_action, is_active,
                           created_at, updated_at, description
                    FROM drawdown_thresholds 
                    WHERE portfolio_id = %s AND is_active = true
                """, (portfolio_id,))
                
                thresholds = []
                for row in cursor.fetchall():
                    threshold = DrawdownThreshold(
                        threshold_id=row['threshold_id'],
                        portfolio_id=row['portfolio_id'],
                        drawdown_type=DrawdownType(row['drawdown_type']),
                        threshold_value=row['threshold_value'],
                        threshold_percentage=row['threshold_percentage'],
                        protection_action=ProtectionAction(row['protection_action']),
                        is_active=row['is_active'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        description=row['description']
                    )
                    thresholds.append(threshold)
                
                return thresholds
                
        except Exception as e:
            logger.error(f"Error getting drawdown thresholds: {e}")
            return []
    
    def _is_threshold_breached(self, metrics: DrawdownMetrics, 
                             threshold: DrawdownThreshold) -> bool:
        """Check if a threshold is breached"""
        if threshold.drawdown_type == DrawdownType.PERCENTAGE:
            return metrics.current_drawdown_pct >= threshold.threshold_percentage
        elif threshold.drawdown_type == DrawdownType.ABSOLUTE:
            return metrics.current_drawdown >= threshold.threshold_value
        else:
            # For other types, use percentage
            return metrics.current_drawdown_pct >= threshold.threshold_percentage
    
    def _trigger_protection_action(self, metrics: DrawdownMetrics, 
                                 threshold: DrawdownThreshold):
        """Trigger protection action for threshold breach"""
        try:
            event_id = str(uuid.uuid4())
            
            # Create drawdown event
            event = DrawdownEvent(
                event_id=event_id,
                portfolio_id=metrics.portfolio_id,
                event_type="THRESHOLD_BREACH",
                drawdown_level=metrics.drawdown_level,
                peak_value=metrics.peak_value,
                trough_value=metrics.trough_value,
                drawdown_amount=metrics.current_drawdown,
                drawdown_percentage=metrics.current_drawdown_pct,
                start_date=datetime.now() - timedelta(days=metrics.drawdown_duration),
                end_date=None,
                duration_days=metrics.drawdown_duration,
                recovery_days=None,
                protection_action=threshold.protection_action,
                action_taken=""
            )
            
            # Execute protection action
            action_result = self._execute_protection_action(event, threshold)
            event.action_taken = action_result
            
            # Store event
            self._store_drawdown_event(event)
            
            logger.warning(f"Drawdown protection triggered: {threshold.protection_action.value} "
                         f"for portfolio {metrics.portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error triggering protection action: {e}")
    
    def _execute_protection_action(self, event: DrawdownEvent, 
                                 threshold: DrawdownThreshold) -> str:
        """Execute the specified protection action"""
        try:
            if threshold.protection_action == ProtectionAction.MONITOR:
                return "MONITORING_ENHANCED"
            
            elif threshold.protection_action == ProtectionAction.ALERT:
                return self._generate_drawdown_alert(event)
            
            elif threshold.protection_action == ProtectionAction.REDUCE_RISK:
                return self._reduce_portfolio_risk(event)
            
            elif threshold.protection_action == ProtectionAction.HALT_NEW_TRADES:
                return self._halt_new_trades(event)
            
            elif threshold.protection_action == ProtectionAction.REDUCE_POSITIONS:
                return self._reduce_positions(event)
            
            elif threshold.protection_action == ProtectionAction.EMERGENCY_STOP:
                return self._emergency_stop(event)
            
            elif threshold.protection_action == ProtectionAction.LIQUIDATE:
                return self._liquidate_portfolio(event)
            
            else:
                return "UNKNOWN_ACTION"
                
        except Exception as e:
            logger.error(f"Error executing protection action: {e}")
            return "ACTION_FAILED"
    
    def _generate_drawdown_alert(self, event: DrawdownEvent) -> str:
        """Generate drawdown alert"""
        try:
            alert_data = {
                'type': 'DRAWDOWN_ALERT',
                'portfolio_id': event.portfolio_id,
                'drawdown_level': event.drawdown_level.value,
                'drawdown_percentage': event.drawdown_percentage,
                'drawdown_amount': event.drawdown_amount,
                'duration_days': event.duration_days,
                'timestamp': event.start_date.isoformat()
            }
            
            # Cache alert
            alert_key = f"alert:drawdown:{event.event_id}"
            self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))
            
            return "ALERT_GENERATED"
            
        except Exception as e:
            logger.error(f"Error generating drawdown alert: {e}")
            return "ALERT_FAILED"
    
    def _reduce_portfolio_risk(self, event: DrawdownEvent) -> str:
        """Reduce portfolio risk by reducing position sizes"""
        try:
            # Reduce all positions by 25%
            reduction_percentage = 0.25
            
            risk_reduction_data = {
                'portfolio_id': event.portfolio_id,
                'action': 'REDUCE_RISK',
                'reduction_percentage': reduction_percentage,
                'reason': f"Drawdown protection: {event.event_id}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Queue risk reduction
            reduction_key = f"risk_reduction:{event.event_id}"
            self.redis_client.setex(reduction_key, 3600, json.dumps(risk_reduction_data))
            
            return f"RISK_REDUCTION_QUEUED:{reduction_percentage}"
            
        except Exception as e:
            logger.error(f"Error reducing portfolio risk: {e}")
            return "RISK_REDUCTION_FAILED"
    
    def _halt_new_trades(self, event: DrawdownEvent) -> str:
        """Halt new trades for the portfolio"""
        try:
            halt_data = {
                'halted': True,
                'reason': f"Drawdown protection: {event.drawdown_level.value}",
                'event_id': event.event_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Set halt flag for 24 hours
            halt_key = f"halt_trades:{event.portfolio_id}"
            self.redis_client.setex(halt_key, 86400, json.dumps(halt_data))
            
            return "NEW_TRADES_HALTED"
            
        except Exception as e:
            logger.error(f"Error halting new trades: {e}")
            return "HALT_FAILED"
    
    def _reduce_positions(self, event: DrawdownEvent) -> str:
        """Reduce position sizes"""
        try:
            # Reduce positions by 50%
            reduction_percentage = 0.50
            
            reduction_data = {
                'portfolio_id': event.portfolio_id,
                'action': 'REDUCE_POSITIONS',
                'reduction_percentage': reduction_percentage,
                'reason': f"Drawdown protection: {event.event_id}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Queue position reduction
            reduction_key = f"position_reduction:{event.event_id}"
            self.redis_client.setex(reduction_key, 3600, json.dumps(reduction_data))
            
            return f"POSITION_REDUCTION_QUEUED:{reduction_percentage}"
            
        except Exception as e:
            logger.error(f"Error reducing positions: {e}")
            return "POSITION_REDUCTION_FAILED"
    
    def _emergency_stop(self, event: DrawdownEvent) -> str:
        """Emergency stop all trading"""
        try:
            stop_data = {
                'stopped': True,
                'reason': f"Emergency stop due to {event.drawdown_level.value} drawdown",
                'event_id': event.event_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Set emergency stop for 24 hours
            stop_key = f"emergency_stop:{event.portfolio_id}"
            self.redis_client.setex(stop_key, 86400, json.dumps(stop_data))
            
            return "EMERGENCY_STOP_ACTIVATED"
            
        except Exception as e:
            logger.error(f"Error activating emergency stop: {e}")
            return "EMERGENCY_STOP_FAILED"
    
    def _liquidate_portfolio(self, event: DrawdownEvent) -> str:
        """Liquidate entire portfolio"""
        try:
            liquidation_data = {
                'portfolio_id': event.portfolio_id,
                'action': 'LIQUIDATE_ALL',
                'reason': f"Critical drawdown protection: {event.event_id}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Queue liquidation
            liquidation_key = f"liquidation:{event.event_id}"
            self.redis_client.setex(liquidation_key, 3600, json.dumps(liquidation_data))
            
            return "LIQUIDATION_QUEUED"
            
        except Exception as e:
            logger.error(f"Error liquidating portfolio: {e}")
            return "LIQUIDATION_FAILED"
    
    def _store_drawdown_event(self, event: DrawdownEvent):
        """Store drawdown event in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO drawdown_events (
                        event_id, portfolio_id, event_type, drawdown_level,
                        peak_value, trough_value, drawdown_amount, drawdown_percentage,
                        start_date, end_date, duration_days, recovery_days,
                        protection_action, action_taken, is_active
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    event.event_id,
                    event.portfolio_id,
                    event.event_type,
                    event.drawdown_level.value,
                    event.peak_value,
                    event.trough_value,
                    event.drawdown_amount,
                    event.drawdown_percentage,
                    event.start_date,
                    event.end_date,
                    event.duration_days,
                    event.recovery_days,
                    event.protection_action.value,
                    event.action_taken,
                    event.is_active
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing drawdown event: {e}")
            self.db_connection.rollback()
    
    def _cache_drawdown_metrics(self, metrics: DrawdownMetrics):
        """Cache drawdown metrics for fast access"""
        try:
            cache_key = f"drawdown_metrics:{metrics.portfolio_id}"
            self.redis_client.setex(
                cache_key,
                300,  # 5 minutes
                json.dumps(metrics.to_dict())
            )
        except Exception as e:
            logger.error(f"Error caching drawdown metrics: {e}")

class DrawdownProtectionManager:
    """Main drawdown protection management system"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.monitor = DrawdownMonitor(db_connection, redis_client)
    
    def start_protection(self):
        """Start drawdown protection monitoring"""
        self.monitor.start_monitoring()
    
    def stop_protection(self):
        """Stop drawdown protection monitoring"""
        self.monitor.stop_monitoring()
    
    def create_drawdown_threshold(self, threshold_data: Dict[str, Any]) -> DrawdownThreshold:
        """Create a new drawdown threshold"""
        try:
            threshold_id = str(uuid.uuid4())
            
            threshold = DrawdownThreshold(
                threshold_id=threshold_id,
                portfolio_id=threshold_data['portfolio_id'],
                drawdown_type=DrawdownType(threshold_data['drawdown_type']),
                threshold_value=float(threshold_data.get('threshold_value', 0.0)),
                threshold_percentage=float(threshold_data['threshold_percentage']),
                protection_action=ProtectionAction(threshold_data['protection_action']),
                is_active=threshold_data.get('is_active', True),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                description=threshold_data.get('description')
            )
            
            # Store in database
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO drawdown_thresholds (
                        threshold_id, portfolio_id, drawdown_type, threshold_value,
                        threshold_percentage, protection_action, is_active,
                        created_at, updated_at, description
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    threshold.threshold_id,
                    threshold.portfolio_id,
                    threshold.drawdown_type.value,
                    threshold.threshold_value,
                    threshold.threshold_percentage,
                    threshold.protection_action.value,
                    threshold.is_active,
                    threshold.created_at,
                    threshold.updated_at,
                    threshold.description
                ))
                self.db_connection.commit()
            
            return threshold
            
        except Exception as e:
            logger.error(f"Error creating drawdown threshold: {e}")
            self.db_connection.rollback()
            raise
    
    def get_drawdown_metrics(self, portfolio_id: str) -> Optional[DrawdownMetrics]:
        """Get current drawdown metrics for a portfolio"""
        try:
            # Try cache first
            cache_key = f"drawdown_metrics:{portfolio_id}"
            cached_metrics = self.redis_client.get(cache_key)
            if cached_metrics:
                data = json.loads(cached_metrics)
                return DrawdownMetrics(
                    portfolio_id=data['portfolio_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    current_value=data['current_value'],
                    peak_value=data['peak_value'],
                    trough_value=data['trough_value'],
                    current_drawdown=data['current_drawdown'],
                    current_drawdown_pct=data['current_drawdown_pct'],
                    max_drawdown=data['max_drawdown'],
                    max_drawdown_pct=data['max_drawdown_pct'],
                    drawdown_duration=data['drawdown_duration'],
                    max_drawdown_duration=data['max_drawdown_duration'],
                    recovery_factor=data['recovery_factor'],
                    drawdown_level=DrawdownLevel(data['drawdown_level']),
                    days_to_recovery=data.get('days_to_recovery')
                )
            
            # Calculate fresh metrics
            return self.monitor.calculate_portfolio_drawdown(portfolio_id)
            
        except Exception as e:
            logger.error(f"Error getting drawdown metrics: {e}")
            return None
    
    def get_drawdown_thresholds(self, portfolio_id: str) -> List[DrawdownThreshold]:
        """Get drawdown thresholds for a portfolio"""
        return self.monitor._get_drawdown_thresholds(portfolio_id)
    
    def get_drawdown_events(self, portfolio_id: str, days: int = 30) -> List[DrawdownEvent]:
        """Get recent drawdown events for a portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT event_id, portfolio_id, event_type, drawdown_level,
                           peak_value, trough_value, drawdown_amount, drawdown_percentage,
                           start_date, end_date, duration_days, recovery_days,
                           protection_action, action_taken, is_active
                    FROM drawdown_events 
                    WHERE portfolio_id = %s 
                    AND start_date >= %s
                    ORDER BY start_date DESC
                """, (portfolio_id, datetime.now() - timedelta(days=days)))
                
                events = []
                for row in cursor.fetchall():
                    event = DrawdownEvent(
                        event_id=row['event_id'],
                        portfolio_id=row['portfolio_id'],
                        event_type=row['event_type'],
                        drawdown_level=DrawdownLevel(row['drawdown_level']),
                        peak_value=row['peak_value'],
                        trough_value=row['trough_value'],
                        drawdown_amount=row['drawdown_amount'],
                        drawdown_percentage=row['drawdown_percentage'],
                        start_date=row['start_date'],
                        end_date=row['end_date'],
                        duration_days=row['duration_days'],
                        recovery_days=row['recovery_days'],
                        protection_action=ProtectionAction(row['protection_action']),
                        action_taken=row['action_taken'],
                        is_active=row['is_active']
                    )
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Error getting drawdown events: {e}")
            return []
    
    def is_portfolio_protected(self, portfolio_id: str) -> Tuple[bool, str]:
        """Check if portfolio has active protection measures"""
        try:
            # Check emergency stop
            stop_key = f"emergency_stop:{portfolio_id}"
            stop_data = self.redis_client.get(stop_key)
            if stop_data:
                return True, "Emergency stop active"
            
            # Check trade halt
            halt_key = f"halt_trades:{portfolio_id}"
            halt_data = self.redis_client.get(halt_key)
            if halt_data:
                return True, "New trades halted"
            
            return False, "No active protection"
            
        except Exception as e:
            logger.error(f"Error checking portfolio protection: {e}")
            return False, "Error checking protection status"

