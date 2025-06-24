"""
Position Limits and Controls System
Advanced position limit management with automated enforcement

This module provides comprehensive position limit management including:
- Dynamic position sizing limits
- Real-time position monitoring
- Automated limit enforcement
- Risk-based position controls
- Emergency position management

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

import numpy as np
import pandas as pd
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import requests

logger = logging.getLogger(__name__)

class LimitType(Enum):
    """Types of position limits"""
    POSITION_SIZE = "position_size"           # Maximum position size per symbol
    PORTFOLIO_EXPOSURE = "portfolio_exposure" # Maximum portfolio exposure
    SECTOR_EXPOSURE = "sector_exposure"       # Maximum sector exposure
    STRATEGY_EXPOSURE = "strategy_exposure"   # Maximum strategy exposure
    DAILY_LOSS = "daily_loss"                # Maximum daily loss
    CONCENTRATION = "concentration"           # Maximum concentration per symbol
    LEVERAGE = "leverage"                     # Maximum leverage ratio
    CORRELATION = "correlation"               # Maximum correlation exposure
    LIQUIDITY = "liquidity"                   # Minimum liquidity requirements
    VOLATILITY = "volatility"                 # Maximum volatility exposure

class LimitScope(Enum):
    """Scope of position limits"""
    SYMBOL = "symbol"           # Per symbol limit
    SECTOR = "sector"           # Per sector limit
    STRATEGY = "strategy"       # Per strategy limit
    PORTFOLIO = "portfolio"     # Portfolio-wide limit
    ACCOUNT = "account"         # Account-wide limit

class EnforcementAction(Enum):
    """Actions to take when limits are breached"""
    ALERT_ONLY = "alert_only"           # Generate alert only
    BLOCK_NEW = "block_new"             # Block new positions
    REDUCE_POSITION = "reduce_position" # Automatically reduce position
    CLOSE_POSITION = "close_position"   # Automatically close position
    EMERGENCY_STOP = "emergency_stop"   # Emergency stop all trading

class LimitStatus(Enum):
    """Status of position limits"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BREACHED = "breached"
    WARNING = "warning"
    SUSPENDED = "suspended"

@dataclass
class PositionLimit:
    """Position limit configuration"""
    limit_id: str
    portfolio_id: str
    limit_type: LimitType
    limit_scope: LimitScope
    scope_value: str  # Symbol, sector, strategy, etc.
    limit_value: float
    warning_threshold: float  # Percentage of limit (e.g., 0.8 for 80%)
    enforcement_action: EnforcementAction
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['limit_type'] = self.limit_type.value
        data['limit_scope'] = self.limit_scope.value
        data['enforcement_action'] = self.enforcement_action.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class PositionExposure:
    """Current position exposure data"""
    portfolio_id: str
    scope_type: LimitScope
    scope_value: str
    current_exposure: float
    exposure_percentage: float  # Percentage of portfolio
    position_count: int
    largest_position: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['scope_type'] = self.scope_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class LimitViolation:
    """Position limit violation record"""
    violation_id: str
    limit_id: str
    portfolio_id: str
    limit_type: LimitType
    scope_value: str
    current_value: float
    limit_value: float
    violation_percentage: float
    enforcement_action: EnforcementAction
    action_taken: str
    timestamp: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['limit_type'] = self.limit_type.value
        data['enforcement_action'] = self.enforcement_action.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class PositionAnalyzer:
    """Analyzes current positions against limits"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
    
    def analyze_portfolio_exposure(self, portfolio_id: str) -> List[PositionExposure]:
        """Analyze current portfolio exposure across different dimensions"""
        try:
            positions = self._get_portfolio_positions(portfolio_id)
            if not positions:
                return []
            
            exposures = []
            
            # Calculate symbol-level exposure
            symbol_exposures = self._calculate_symbol_exposure(portfolio_id, positions)
            exposures.extend(symbol_exposures)
            
            # Calculate sector-level exposure
            sector_exposures = self._calculate_sector_exposure(portfolio_id, positions)
            exposures.extend(sector_exposures)
            
            # Calculate strategy-level exposure
            strategy_exposures = self._calculate_strategy_exposure(portfolio_id, positions)
            exposures.extend(strategy_exposures)
            
            # Calculate portfolio-level exposure
            portfolio_exposure = self._calculate_portfolio_exposure(portfolio_id, positions)
            exposures.append(portfolio_exposure)
            
            return exposures
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio exposure: {e}")
            return []
    
    def _get_portfolio_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get current portfolio positions"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT p.symbol, p.quantity, p.market_value, p.cost_basis,
                           p.unrealized_pnl, p.strategy_id, p.position_type,
                           s.sector, s.market_cap, s.avg_volume
                    FROM positions p
                    LEFT JOIN symbols s ON p.symbol = s.symbol
                    WHERE p.portfolio_id = %s AND p.status = 'open'
                """, (portfolio_id,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {e}")
            return []
    
    def _calculate_symbol_exposure(self, portfolio_id: str, positions: List[Dict[str, Any]]) -> List[PositionExposure]:
        """Calculate exposure by symbol"""
        symbol_exposures = {}
        total_portfolio_value = sum(abs(pos['market_value']) for pos in positions)
        
        for position in positions:
            symbol = position['symbol']
            market_value = abs(position['market_value'])
            
            if symbol not in symbol_exposures:
                symbol_exposures[symbol] = {
                    'exposure': 0.0,
                    'count': 0,
                    'largest': 0.0
                }
            
            symbol_exposures[symbol]['exposure'] += market_value
            symbol_exposures[symbol]['count'] += 1
            symbol_exposures[symbol]['largest'] = max(
                symbol_exposures[symbol]['largest'], 
                market_value
            )
        
        exposures = []
        for symbol, data in symbol_exposures.items():
            exposure_pct = (data['exposure'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            exposure = PositionExposure(
                portfolio_id=portfolio_id,
                scope_type=LimitScope.SYMBOL,
                scope_value=symbol,
                current_exposure=data['exposure'],
                exposure_percentage=exposure_pct,
                position_count=data['count'],
                largest_position=data['largest'],
                timestamp=datetime.now()
            )
            exposures.append(exposure)
        
        return exposures
    
    def _calculate_sector_exposure(self, portfolio_id: str, positions: List[Dict[str, Any]]) -> List[PositionExposure]:
        """Calculate exposure by sector"""
        sector_exposures = {}
        total_portfolio_value = sum(abs(pos['market_value']) for pos in positions)
        
        for position in positions:
            sector = position.get('sector', 'Unknown')
            market_value = abs(position['market_value'])
            
            if sector not in sector_exposures:
                sector_exposures[sector] = {
                    'exposure': 0.0,
                    'count': 0,
                    'largest': 0.0
                }
            
            sector_exposures[sector]['exposure'] += market_value
            sector_exposures[sector]['count'] += 1
            sector_exposures[sector]['largest'] = max(
                sector_exposures[sector]['largest'], 
                market_value
            )
        
        exposures = []
        for sector, data in sector_exposures.items():
            exposure_pct = (data['exposure'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            exposure = PositionExposure(
                portfolio_id=portfolio_id,
                scope_type=LimitScope.SECTOR,
                scope_value=sector,
                current_exposure=data['exposure'],
                exposure_percentage=exposure_pct,
                position_count=data['count'],
                largest_position=data['largest'],
                timestamp=datetime.now()
            )
            exposures.append(exposure)
        
        return exposures
    
    def _calculate_strategy_exposure(self, portfolio_id: str, positions: List[Dict[str, Any]]) -> List[PositionExposure]:
        """Calculate exposure by strategy"""
        strategy_exposures = {}
        total_portfolio_value = sum(abs(pos['market_value']) for pos in positions)
        
        for position in positions:
            strategy = position.get('strategy_id', 'Unknown')
            market_value = abs(position['market_value'])
            
            if strategy not in strategy_exposures:
                strategy_exposures[strategy] = {
                    'exposure': 0.0,
                    'count': 0,
                    'largest': 0.0
                }
            
            strategy_exposures[strategy]['exposure'] += market_value
            strategy_exposures[strategy]['count'] += 1
            strategy_exposures[strategy]['largest'] = max(
                strategy_exposures[strategy]['largest'], 
                market_value
            )
        
        exposures = []
        for strategy, data in strategy_exposures.items():
            exposure_pct = (data['exposure'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            exposure = PositionExposure(
                portfolio_id=portfolio_id,
                scope_type=LimitScope.STRATEGY,
                scope_value=strategy,
                current_exposure=data['exposure'],
                exposure_percentage=exposure_pct,
                position_count=data['count'],
                largest_position=data['largest'],
                timestamp=datetime.now()
            )
            exposures.append(exposure)
        
        return exposures
    
    def _calculate_portfolio_exposure(self, portfolio_id: str, positions: List[Dict[str, Any]]) -> PositionExposure:
        """Calculate total portfolio exposure"""
        total_exposure = sum(abs(pos['market_value']) for pos in positions)
        largest_position = max((abs(pos['market_value']) for pos in positions), default=0.0)
        
        return PositionExposure(
            portfolio_id=portfolio_id,
            scope_type=LimitScope.PORTFOLIO,
            scope_value="TOTAL",
            current_exposure=total_exposure,
            exposure_percentage=100.0,
            position_count=len(positions),
            largest_position=largest_position,
            timestamp=datetime.now()
        )

class LimitEnforcer:
    """Enforces position limits with automated actions"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.analyzer = PositionAnalyzer(db_connection, redis_client)
    
    def check_position_limits(self, portfolio_id: str) -> List[LimitViolation]:
        """Check all position limits for a portfolio"""
        try:
            violations = []
            
            # Get active limits for portfolio
            limits = self._get_active_limits(portfolio_id)
            if not limits:
                return violations
            
            # Get current exposures
            exposures = self.analyzer.analyze_portfolio_exposure(portfolio_id)
            exposure_map = {
                (exp.scope_type, exp.scope_value): exp for exp in exposures
            }
            
            # Check each limit
            for limit in limits:
                exposure_key = (limit.limit_scope, limit.scope_value)
                current_exposure = exposure_map.get(exposure_key)
                
                if current_exposure:
                    violation = self._check_limit_violation(limit, current_exposure)
                    if violation:
                        violations.append(violation)
                        
                        # Take enforcement action
                        self._take_enforcement_action(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return []
    
    def _get_active_limits(self, portfolio_id: str) -> List[PositionLimit]:
        """Get active position limits for portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT limit_id, portfolio_id, limit_type, limit_scope, scope_value,
                           limit_value, warning_threshold, enforcement_action, is_active,
                           created_at, updated_at, created_by, description
                    FROM position_limits 
                    WHERE portfolio_id = %s AND is_active = true
                """, (portfolio_id,))
                
                limits = []
                for row in cursor.fetchall():
                    limit = PositionLimit(
                        limit_id=row['limit_id'],
                        portfolio_id=row['portfolio_id'],
                        limit_type=LimitType(row['limit_type']),
                        limit_scope=LimitScope(row['limit_scope']),
                        scope_value=row['scope_value'],
                        limit_value=row['limit_value'],
                        warning_threshold=row['warning_threshold'],
                        enforcement_action=EnforcementAction(row['enforcement_action']),
                        is_active=row['is_active'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        created_by=row['created_by'],
                        description=row['description']
                    )
                    limits.append(limit)
                
                return limits
                
        except Exception as e:
            logger.error(f"Error getting active limits: {e}")
            return []
    
    def _check_limit_violation(self, limit: PositionLimit, exposure: PositionExposure) -> Optional[LimitViolation]:
        """Check if a limit is violated"""
        try:
            current_value = exposure.current_exposure
            limit_value = limit.limit_value
            
            # Convert percentage limits to absolute values if needed
            if limit.limit_type in [LimitType.PORTFOLIO_EXPOSURE, LimitType.SECTOR_EXPOSURE, 
                                  LimitType.CONCENTRATION] and limit_value <= 1.0:
                # Assume percentage limit, convert to absolute
                portfolio_value = self._get_portfolio_value(limit.portfolio_id)
                limit_value = limit_value * portfolio_value
            
            # Check if limit is breached
            if current_value > limit_value:
                violation_percentage = (current_value / limit_value) - 1.0
                
                violation = LimitViolation(
                    violation_id=str(uuid.uuid4()),
                    limit_id=limit.limit_id,
                    portfolio_id=limit.portfolio_id,
                    limit_type=limit.limit_type,
                    scope_value=limit.scope_value,
                    current_value=current_value,
                    limit_value=limit_value,
                    violation_percentage=violation_percentage,
                    enforcement_action=limit.enforcement_action,
                    action_taken="",  # Will be filled by enforcement action
                    timestamp=datetime.now()
                )
                
                return violation
            
            # Check warning threshold
            warning_threshold = limit_value * limit.warning_threshold
            if current_value > warning_threshold:
                # Create warning violation
                violation_percentage = (current_value / limit_value)
                
                violation = LimitViolation(
                    violation_id=str(uuid.uuid4()),
                    limit_id=limit.limit_id,
                    portfolio_id=limit.portfolio_id,
                    limit_type=limit.limit_type,
                    scope_value=limit.scope_value,
                    current_value=current_value,
                    limit_value=limit_value,
                    violation_percentage=violation_percentage,
                    enforcement_action=EnforcementAction.ALERT_ONLY,  # Warning only
                    action_taken="WARNING_GENERATED",
                    timestamp=datetime.now()
                )
                
                return violation
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking limit violation: {e}")
            return None
    
    def _get_portfolio_value(self, portfolio_id: str) -> float:
        """Get total portfolio value"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT SUM(ABS(market_value)) as total_value
                    FROM positions 
                    WHERE portfolio_id = %s AND status = 'open'
                """, (portfolio_id,))
                
                result = cursor.fetchone()
                return result[0] if result and result[0] else 0.0
                
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    def _take_enforcement_action(self, violation: LimitViolation):
        """Take enforcement action for limit violation"""
        try:
            action_taken = ""
            
            if violation.enforcement_action == EnforcementAction.ALERT_ONLY:
                action_taken = self._generate_alert(violation)
            
            elif violation.enforcement_action == EnforcementAction.BLOCK_NEW:
                action_taken = self._block_new_positions(violation)
            
            elif violation.enforcement_action == EnforcementAction.REDUCE_POSITION:
                action_taken = self._reduce_position(violation)
            
            elif violation.enforcement_action == EnforcementAction.CLOSE_POSITION:
                action_taken = self._close_position(violation)
            
            elif violation.enforcement_action == EnforcementAction.EMERGENCY_STOP:
                action_taken = self._emergency_stop(violation)
            
            # Update violation record with action taken
            violation.action_taken = action_taken
            
            # Store violation in database
            self._store_violation(violation)
            
            logger.warning(f"Limit violation: {violation.limit_type.value} for {violation.scope_value}, "
                         f"Action: {action_taken}")
            
        except Exception as e:
            logger.error(f"Error taking enforcement action: {e}")
    
    def _generate_alert(self, violation: LimitViolation) -> str:
        """Generate alert for limit violation"""
        try:
            alert_data = {
                'type': 'POSITION_LIMIT_VIOLATION',
                'portfolio_id': violation.portfolio_id,
                'limit_type': violation.limit_type.value,
                'scope_value': violation.scope_value,
                'current_value': violation.current_value,
                'limit_value': violation.limit_value,
                'violation_percentage': violation.violation_percentage,
                'timestamp': violation.timestamp.isoformat()
            }
            
            # Cache alert for real-time access
            alert_key = f"alert:limit_violation:{violation.violation_id}"
            self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))
            
            return "ALERT_GENERATED"
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            return "ALERT_FAILED"
    
    def _block_new_positions(self, violation: LimitViolation) -> str:
        """Block new positions for the violated scope"""
        try:
            # Set blocking flag in Redis
            block_key = f"block_positions:{violation.portfolio_id}:{violation.scope_value}"
            block_data = {
                'blocked': True,
                'reason': f"{violation.limit_type.value} limit exceeded",
                'violation_id': violation.violation_id,
                'timestamp': violation.timestamp.isoformat()
            }
            
            # Block for 24 hours
            self.redis_client.setex(block_key, 86400, json.dumps(block_data))
            
            return "NEW_POSITIONS_BLOCKED"
            
        except Exception as e:
            logger.error(f"Error blocking new positions: {e}")
            return "BLOCKING_FAILED"
    
    def _reduce_position(self, violation: LimitViolation) -> str:
        """Automatically reduce position to comply with limit"""
        try:
            # Calculate required reduction
            excess_amount = violation.current_value - violation.limit_value
            reduction_percentage = excess_amount / violation.current_value
            
            # Get positions to reduce
            positions_to_reduce = self._get_positions_for_scope(
                violation.portfolio_id, 
                violation.scope_value
            )
            
            if not positions_to_reduce:
                return "NO_POSITIONS_TO_REDUCE"
            
            # Create reduction orders
            orders_created = 0
            for position in positions_to_reduce:
                reduction_quantity = position['quantity'] * reduction_percentage
                
                if abs(reduction_quantity) >= 1:  # Only reduce if meaningful
                    order_data = {
                        'portfolio_id': violation.portfolio_id,
                        'symbol': position['symbol'],
                        'quantity': -reduction_quantity,  # Opposite sign to reduce
                        'order_type': 'MARKET',
                        'reason': f"Limit violation reduction: {violation.violation_id}"
                    }
                    
                    # Queue order for execution
                    order_key = f"reduction_order:{violation.violation_id}:{position['symbol']}"
                    self.redis_client.setex(order_key, 3600, json.dumps(order_data))
                    orders_created += 1
            
            return f"REDUCTION_ORDERS_CREATED:{orders_created}"
            
        except Exception as e:
            logger.error(f"Error reducing position: {e}")
            return "REDUCTION_FAILED"
    
    def _close_position(self, violation: LimitViolation) -> str:
        """Automatically close positions for the violated scope"""
        try:
            # Get positions to close
            positions_to_close = self._get_positions_for_scope(
                violation.portfolio_id, 
                violation.scope_value
            )
            
            if not positions_to_close:
                return "NO_POSITIONS_TO_CLOSE"
            
            # Create closing orders
            orders_created = 0
            for position in positions_to_close:
                order_data = {
                    'portfolio_id': violation.portfolio_id,
                    'symbol': position['symbol'],
                    'quantity': -position['quantity'],  # Close entire position
                    'order_type': 'MARKET',
                    'reason': f"Limit violation closure: {violation.violation_id}"
                }
                
                # Queue order for execution
                order_key = f"close_order:{violation.violation_id}:{position['symbol']}"
                self.redis_client.setex(order_key, 3600, json.dumps(order_data))
                orders_created += 1
            
            return f"CLOSE_ORDERS_CREATED:{orders_created}"
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return "CLOSURE_FAILED"
    
    def _emergency_stop(self, violation: LimitViolation) -> str:
        """Emergency stop all trading for the portfolio"""
        try:
            # Set emergency stop flag
            stop_key = f"emergency_stop:{violation.portfolio_id}"
            stop_data = {
                'stopped': True,
                'reason': f"Emergency stop due to {violation.limit_type.value} violation",
                'violation_id': violation.violation_id,
                'timestamp': violation.timestamp.isoformat()
            }
            
            # Emergency stop for 24 hours
            self.redis_client.setex(stop_key, 86400, json.dumps(stop_data))
            
            # Also close all positions
            self._close_all_positions(violation.portfolio_id, violation.violation_id)
            
            return "EMERGENCY_STOP_ACTIVATED"
            
        except Exception as e:
            logger.error(f"Error activating emergency stop: {e}")
            return "EMERGENCY_STOP_FAILED"
    
    def _get_positions_for_scope(self, portfolio_id: str, scope_value: str) -> List[Dict[str, Any]]:
        """Get positions for a specific scope"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Determine query based on scope
                if scope_value == "TOTAL":
                    # All positions
                    cursor.execute("""
                        SELECT symbol, quantity, market_value, position_type
                        FROM positions 
                        WHERE portfolio_id = %s AND status = 'open'
                    """, (portfolio_id,))
                else:
                    # Specific symbol, sector, or strategy
                    cursor.execute("""
                        SELECT p.symbol, p.quantity, p.market_value, p.position_type
                        FROM positions p
                        LEFT JOIN symbols s ON p.symbol = s.symbol
                        WHERE p.portfolio_id = %s AND p.status = 'open'
                        AND (p.symbol = %s OR s.sector = %s OR p.strategy_id = %s)
                    """, (portfolio_id, scope_value, scope_value, scope_value))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting positions for scope: {e}")
            return []
    
    def _close_all_positions(self, portfolio_id: str, violation_id: str):
        """Close all positions in a portfolio"""
        try:
            positions = self._get_positions_for_scope(portfolio_id, "TOTAL")
            
            for position in positions:
                order_data = {
                    'portfolio_id': portfolio_id,
                    'symbol': position['symbol'],
                    'quantity': -position['quantity'],
                    'order_type': 'MARKET',
                    'reason': f"Emergency stop closure: {violation_id}"
                }
                
                order_key = f"emergency_close:{violation_id}:{position['symbol']}"
                self.redis_client.setex(order_key, 3600, json.dumps(order_data))
                
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
    
    def _store_violation(self, violation: LimitViolation):
        """Store violation record in database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO limit_violations (
                        violation_id, limit_id, portfolio_id, limit_type, scope_value,
                        current_value, limit_value, violation_percentage, enforcement_action,
                        action_taken, timestamp, resolved
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    violation.violation_id,
                    violation.limit_id,
                    violation.portfolio_id,
                    violation.limit_type.value,
                    violation.scope_value,
                    violation.current_value,
                    violation.limit_value,
                    violation.violation_percentage,
                    violation.enforcement_action.value,
                    violation.action_taken,
                    violation.timestamp,
                    violation.resolved
                ))
                self.db_connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing violation: {e}")
            self.db_connection.rollback()

class PositionLimitManager:
    """Main position limit management system"""
    
    def __init__(self, db_connection, redis_client):
        self.db_connection = db_connection
        self.redis_client = redis_client
        self.analyzer = PositionAnalyzer(db_connection, redis_client)
        self.enforcer = LimitEnforcer(db_connection, redis_client)
        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval = 10  # seconds
    
    def start_monitoring(self):
        """Start position limit monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Position limit monitoring started")
    
    def stop_monitoring(self):
        """Stop position limit monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Position limit monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Get all active portfolios
                portfolios = self._get_active_portfolios()
                
                for portfolio in portfolios:
                    portfolio_id = portfolio['portfolio_id']
                    
                    # Check position limits
                    violations = self.enforcer.check_position_limits(portfolio_id)
                    
                    if violations:
                        logger.warning(f"Found {len(violations)} limit violations for portfolio {portfolio_id}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in position limit monitoring loop: {e}")
                time.sleep(self.check_interval)
    
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
    
    def create_position_limit(self, limit_data: Dict[str, Any]) -> PositionLimit:
        """Create a new position limit"""
        try:
            limit_id = str(uuid.uuid4())
            
            position_limit = PositionLimit(
                limit_id=limit_id,
                portfolio_id=limit_data['portfolio_id'],
                limit_type=LimitType(limit_data['limit_type']),
                limit_scope=LimitScope(limit_data['limit_scope']),
                scope_value=limit_data['scope_value'],
                limit_value=float(limit_data['limit_value']),
                warning_threshold=float(limit_data.get('warning_threshold', 0.8)),
                enforcement_action=EnforcementAction(limit_data['enforcement_action']),
                is_active=limit_data.get('is_active', True),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by=limit_data.get('created_by', 'system'),
                description=limit_data.get('description')
            )
            
            # Store in database
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO position_limits (
                        limit_id, portfolio_id, limit_type, limit_scope, scope_value,
                        limit_value, warning_threshold, enforcement_action, is_active,
                        created_at, updated_at, created_by, description
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    position_limit.limit_id,
                    position_limit.portfolio_id,
                    position_limit.limit_type.value,
                    position_limit.limit_scope.value,
                    position_limit.scope_value,
                    position_limit.limit_value,
                    position_limit.warning_threshold,
                    position_limit.enforcement_action.value,
                    position_limit.is_active,
                    position_limit.created_at,
                    position_limit.updated_at,
                    position_limit.created_by,
                    position_limit.description
                ))
                self.db_connection.commit()
            
            return position_limit
            
        except Exception as e:
            logger.error(f"Error creating position limit: {e}")
            self.db_connection.rollback()
            raise
    
    def get_position_limits(self, portfolio_id: str) -> List[PositionLimit]:
        """Get position limits for a portfolio"""
        return self.enforcer._get_active_limits(portfolio_id)
    
    def get_position_exposures(self, portfolio_id: str) -> List[PositionExposure]:
        """Get current position exposures for a portfolio"""
        return self.analyzer.analyze_portfolio_exposure(portfolio_id)
    
    def get_limit_violations(self, portfolio_id: str, limit_days: int = 30) -> List[LimitViolation]:
        """Get recent limit violations for a portfolio"""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT violation_id, limit_id, portfolio_id, limit_type, scope_value,
                           current_value, limit_value, violation_percentage, enforcement_action,
                           action_taken, timestamp, resolved
                    FROM limit_violations 
                    WHERE portfolio_id = %s 
                    AND timestamp >= %s
                    ORDER BY timestamp DESC
                """, (portfolio_id, datetime.now() - timedelta(days=limit_days)))
                
                violations = []
                for row in cursor.fetchall():
                    violation = LimitViolation(
                        violation_id=row['violation_id'],
                        limit_id=row['limit_id'],
                        portfolio_id=row['portfolio_id'],
                        limit_type=LimitType(row['limit_type']),
                        scope_value=row['scope_value'],
                        current_value=row['current_value'],
                        limit_value=row['limit_value'],
                        violation_percentage=row['violation_percentage'],
                        enforcement_action=EnforcementAction(row['enforcement_action']),
                        action_taken=row['action_taken'],
                        timestamp=row['timestamp'],
                        resolved=row['resolved']
                    )
                    violations.append(violation)
                
                return violations
                
        except Exception as e:
            logger.error(f"Error getting limit violations: {e}")
            return []
    
    def check_position_allowed(self, portfolio_id: str, symbol: str, quantity: float) -> Tuple[bool, str]:
        """Check if a new position is allowed given current limits"""
        try:
            # Check if positions are blocked
            block_key = f"block_positions:{portfolio_id}:{symbol}"
            blocked_data = self.redis_client.get(block_key)
            if blocked_data:
                return False, "New positions blocked due to limit violation"
            
            # Check emergency stop
            stop_key = f"emergency_stop:{portfolio_id}"
            stop_data = self.redis_client.get(stop_key)
            if stop_data:
                return False, "Emergency stop active - no new positions allowed"
            
            # Simulate the new position and check limits
            current_exposures = self.analyzer.analyze_portfolio_exposure(portfolio_id)
            
            # Get symbol price for position value calculation
            symbol_price = self._get_symbol_price(symbol)
            new_position_value = abs(quantity * symbol_price)
            
            # Check against symbol limits
            symbol_exposure = next(
                (exp for exp in current_exposures 
                 if exp.scope_type == LimitScope.SYMBOL and exp.scope_value == symbol), 
                None
            )
            
            if symbol_exposure:
                projected_exposure = symbol_exposure.current_exposure + new_position_value
                
                # Get symbol limits
                symbol_limits = [
                    limit for limit in self.enforcer._get_active_limits(portfolio_id)
                    if limit.limit_scope == LimitScope.SYMBOL and limit.scope_value == symbol
                ]
                
                for limit in symbol_limits:
                    if projected_exposure > limit.limit_value:
                        return False, f"Would exceed {limit.limit_type.value} limit for {symbol}"
            
            return True, "Position allowed"
            
        except Exception as e:
            logger.error(f"Error checking position allowed: {e}")
            return False, "Error checking position limits"
    
    def _get_symbol_price(self, symbol: str) -> float:
        """Get current symbol price"""
        try:
            # Try cache first
            price_key = f"price:{symbol}"
            cached_price = self.redis_client.get(price_key)
            if cached_price:
                return float(cached_price)
            
            # Mock price for development
            # In production, this would fetch real market data
            mock_prices = {
                'AAPL': 150.0,
                'MSFT': 300.0,
                'GOOGL': 2500.0,
                'TSLA': 200.0,
                'SPY': 400.0,
                'QQQ': 350.0
            }
            
            price = mock_prices.get(symbol, 100.0)
            
            # Cache for 1 minute
            self.redis_client.setex(price_key, 60, str(price))
            
            return price
            
        except Exception as e:
            logger.error(f"Error getting symbol price: {e}")
            return 100.0  # Default price

