"""
Signal History Tracking and Performance Analytics System for AI Options Trading System
Comprehensive tracking, analysis, and reporting of signal performance and outcomes
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
import logging
import json
import math
from contextlib import contextmanager
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

logger = logging.getLogger(__name__)

class SignalStatus(Enum):
    """Signal execution status"""
    PENDING = "pending"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PARTIAL = "partial"

class SignalOutcome(Enum):
    """Signal outcome classification"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"

class PerformanceMetric(Enum):
    """Performance metrics"""
    WIN_RATE = "win_rate"
    AVERAGE_RETURN = "average_return"
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    LARGEST_WIN = "largest_win"
    LARGEST_LOSS = "largest_loss"

@dataclass
class SignalExecution:
    """Signal execution record"""
    signal_id: str
    execution_id: str
    user_id: str
    symbol: str
    strategy_name: str
    signal_type: str
    
    # Execution details
    executed_at: datetime
    execution_price: float
    position_size: float
    commission: float = 0.0
    slippage: float = 0.0
    
    # Original signal data
    original_entry_price: float = 0.0
    original_target_price: float = 0.0
    original_stop_loss: float = 0.0
    original_confidence: float = 0.0
    original_expected_return: float = 0.0
    
    # Current status
    status: SignalStatus = SignalStatus.EXECUTED
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Final outcome (when closed)
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    actual_return: Optional[float] = None
    outcome: SignalOutcome = SignalOutcome.PENDING
    
    # Performance tracking
    max_favorable_excursion: float = 0.0  # Best unrealized profit
    max_adverse_excursion: float = 0.0    # Worst unrealized loss
    hold_time_hours: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    period_start: datetime
    period_end: datetime
    total_signals: int
    executed_signals: int
    
    # Win/Loss metrics
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    
    # Return metrics
    total_return: float
    average_return: float
    average_winning_return: float
    average_losing_return: float
    best_trade: float
    worst_trade: float
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # Days
    volatility: float
    
    # Efficiency metrics
    profit_factor: float  # Gross profit / Gross loss
    expectancy: float     # Average win * win_rate - Average loss * loss_rate
    recovery_factor: float # Total return / Max drawdown
    
    # Strategy breakdown
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    symbol_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    monthly_performance: Dict[str, float] = field(default_factory=dict)
    
    # Additional metrics
    average_hold_time: float = 0.0  # Hours
    commission_total: float = 0.0
    slippage_total: float = 0.0

class SignalTracker:
    """Signal history tracking and performance analytics"""
    
    def __init__(self, db_path: str = "signal_history.db"):
        self.db_path = db_path
        self._init_database()
        
        # Performance cache
        self.performance_cache = {}
        self.cache_expiry = {}
        
    def track_signal_generation(self, signal_data: Dict[str, Any]) -> bool:
        """Track when a signal is generated"""
        try:
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO signal_history (
                        signal_id, symbol, strategy_name, signal_type, priority,
                        confidence, expected_return, max_risk, entry_price,
                        target_price, stop_loss, time_horizon, reasoning,
                        technical_factors, options_factors, risk_factors,
                        score_data, generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data.get("signal_id"),
                    signal_data.get("symbol"),
                    signal_data.get("strategy_name"),
                    signal_data.get("signal_type"),
                    signal_data.get("priority", 2),
                    signal_data.get("confidence", 0.0),
                    signal_data.get("expected_return", 0.0),
                    signal_data.get("max_risk", 0.0),
                    signal_data.get("entry_price", 0.0),
                    signal_data.get("target_price", 0.0),
                    signal_data.get("stop_loss", 0.0),
                    signal_data.get("time_horizon", 0),
                    signal_data.get("reasoning", ""),
                    json.dumps(signal_data.get("technical_factors", {})),
                    json.dumps(signal_data.get("options_factors", {})),
                    json.dumps(signal_data.get("risk_factors", {})),
                    json.dumps(signal_data.get("score", {})),
                    datetime.now(timezone.utc)
                ))
                conn.commit()
                
            logger.info(f"Signal {signal_data.get('signal_id')} tracked in history")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking signal generation: {e}")
            return False
    
    def track_signal_execution(self, execution: SignalExecution) -> bool:
        """Track when a signal is executed"""
        try:
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO signal_executions (
                        execution_id, signal_id, user_id, symbol, strategy_name,
                        signal_type, executed_at, execution_price, position_size,
                        commission, slippage, original_entry_price, original_target_price,
                        original_stop_loss, original_confidence, original_expected_return,
                        status, current_price, unrealized_pnl, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    execution.execution_id,
                    execution.signal_id,
                    execution.user_id,
                    execution.symbol,
                    execution.strategy_name,
                    execution.signal_type,
                    execution.executed_at,
                    execution.execution_price,
                    execution.position_size,
                    execution.commission,
                    execution.slippage,
                    execution.original_entry_price,
                    execution.original_target_price,
                    execution.original_stop_loss,
                    execution.original_confidence,
                    execution.original_expected_return,
                    execution.status.value,
                    execution.current_price,
                    execution.unrealized_pnl,
                    execution.created_at
                ))
                conn.commit()
                
            logger.info(f"Signal execution {execution.execution_id} tracked")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking signal execution: {e}")
            return False
    
    def update_signal_position(self, execution_id: str, current_price: float,
                             unrealized_pnl: float) -> bool:
        """Update current position status"""
        try:
            with self._get_db_connection() as conn:
                # Get current execution data
                cursor = conn.execute('''
                    SELECT max_favorable_excursion, max_adverse_excursion
                    FROM signal_executions WHERE execution_id = ?
                ''', (execution_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                current_mfe, current_mae = result
                
                # Update MFE and MAE
                new_mfe = max(current_mfe, unrealized_pnl) if unrealized_pnl > 0 else current_mfe
                new_mae = min(current_mae, unrealized_pnl) if unrealized_pnl < 0 else current_mae
                
                # Update position
                conn.execute('''
                    UPDATE signal_executions SET
                        current_price = ?,
                        unrealized_pnl = ?,
                        max_favorable_excursion = ?,
                        max_adverse_excursion = ?,
                        updated_at = ?
                    WHERE execution_id = ?
                ''', (
                    current_price,
                    unrealized_pnl,
                    new_mfe,
                    new_mae,
                    datetime.now(timezone.utc),
                    execution_id
                ))
                conn.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal position: {e}")
            return False
    
    def close_signal_position(self, execution_id: str, exit_price: float,
                            realized_pnl: float, outcome: SignalOutcome) -> bool:
        """Close a signal position and record final outcome"""
        try:
            with self._get_db_connection() as conn:
                # Get execution data to calculate metrics
                cursor = conn.execute('''
                    SELECT executed_at, position_size FROM signal_executions
                    WHERE execution_id = ?
                ''', (execution_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                executed_at, position_size = result
                executed_at = datetime.fromisoformat(executed_at.replace('Z', '+00:00'))
                
                # Calculate hold time
                closed_at = datetime.now(timezone.utc)
                hold_time_hours = (closed_at - executed_at).total_seconds() / 3600
                
                # Calculate actual return
                actual_return = realized_pnl / (position_size * abs(exit_price)) if position_size > 0 else 0
                
                # Update execution record
                conn.execute('''
                    UPDATE signal_executions SET
                        status = ?,
                        closed_at = ?,
                        exit_price = ?,
                        realized_pnl = ?,
                        actual_return = ?,
                        outcome = ?,
                        hold_time_hours = ?,
                        updated_at = ?
                    WHERE execution_id = ?
                ''', (
                    SignalStatus.EXECUTED.value,
                    closed_at,
                    exit_price,
                    realized_pnl,
                    actual_return,
                    outcome.value,
                    hold_time_hours,
                    datetime.now(timezone.utc),
                    execution_id
                ))
                conn.commit()
                
            # Clear performance cache
            self._clear_performance_cache()
            
            logger.info(f"Signal position {execution_id} closed with {outcome.value} outcome")
            return True
            
        except Exception as e:
            logger.error(f"Error closing signal position: {e}")
            return False
    
    def get_signal_history(self, symbol: str = None, strategy: str = None,
                          start_date: datetime = None, end_date: datetime = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get signal history with optional filters"""
        try:
            query = "SELECT * FROM signal_history WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy_name = ?"
                params.append(strategy)
            
            if start_date:
                query += " AND generated_at >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND generated_at <= ?"
                params.append(end_date)
            
            query += " ORDER BY generated_at DESC LIMIT ?"
            params.append(limit)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                signals = []
                for row in cursor.fetchall():
                    signal = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    for field in ['technical_factors', 'options_factors', 'risk_factors', 'score_data']:
                        if signal.get(field):
                            signal[field] = json.loads(signal[field])
                    
                    signals.append(signal)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting signal history: {e}")
            return []
    
    def get_execution_history(self, user_id: str = None, symbol: str = None,
                            strategy: str = None, start_date: datetime = None,
                            end_date: datetime = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history with optional filters"""
        try:
            query = "SELECT * FROM signal_executions WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy_name = ?"
                params.append(strategy)
            
            if start_date:
                query += " AND executed_at >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND executed_at <= ?"
                params.append(end_date)
            
            query += " ORDER BY executed_at DESC LIMIT ?"
            params.append(limit)
            
            with self._get_db_connection() as conn:
                cursor = conn.execute(query, params)
                columns = [description[0] for description in cursor.description]
                
                executions = []
                for row in cursor.fetchall():
                    execution = dict(zip(columns, row))
                    executions.append(execution)
                
                return executions
                
        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []
    
    def generate_performance_report(self, user_id: str = None, start_date: datetime = None,
                                  end_date: datetime = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now(timezone.utc)
            if not start_date:
                start_date = end_date - timedelta(days=30)  # Last 30 days
            
            # Check cache first
            cache_key = f"{user_id}_{start_date.isoformat()}_{end_date.isoformat()}"
            if cache_key in self.performance_cache:
                if datetime.now(timezone.utc) < self.cache_expiry.get(cache_key, datetime.min):
                    return self.performance_cache[cache_key]
            
            # Get execution data
            executions = self.get_execution_history(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            if not executions:
                return self._create_empty_report(start_date, end_date)
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(executions)
            
            # Filter closed positions for performance calculation
            closed_df = df[df['outcome'].notna() & (df['outcome'] != 'pending')]
            
            if closed_df.empty:
                return self._create_empty_report(start_date, end_date)
            
            # Calculate basic metrics
            total_signals = len(df)
            executed_signals = len(closed_df)
            
            # Win/Loss metrics
            winning_trades = len(closed_df[closed_df['outcome'] == 'win'])
            losing_trades = len(closed_df[closed_df['outcome'] == 'loss'])
            breakeven_trades = len(closed_df[closed_df['outcome'] == 'breakeven'])
            win_rate = winning_trades / executed_signals if executed_signals > 0 else 0
            
            # Return metrics
            returns = closed_df['actual_return'].fillna(0).astype(float)
            total_return = returns.sum()
            average_return = returns.mean()
            
            winning_returns = returns[closed_df['outcome'] == 'win']
            losing_returns = returns[closed_df['outcome'] == 'loss']
            
            average_winning_return = winning_returns.mean() if len(winning_returns) > 0 else 0
            average_losing_return = losing_returns.mean() if len(losing_returns) > 0 else 0
            
            best_trade = returns.max() if len(returns) > 0 else 0
            worst_trade = returns.min() if len(returns) > 0 else 0
            
            # Risk metrics
            volatility = returns.std() if len(returns) > 1 else 0
            sharpe_ratio = (average_return / volatility) if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Calculate max drawdown duration
            max_dd_duration = self._calculate_max_drawdown_duration(cumulative_returns)
            
            # Efficiency metrics
            gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
            gross_loss = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            expectancy = (average_winning_return * win_rate) - (abs(average_losing_return) * (1 - win_rate))
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
            
            # Strategy and symbol breakdown
            strategy_performance = self._calculate_strategy_performance(closed_df)
            symbol_performance = self._calculate_symbol_performance(closed_df)
            monthly_performance = self._calculate_monthly_performance(closed_df)
            
            # Additional metrics
            hold_times = closed_df['hold_time_hours'].fillna(0).astype(float)
            average_hold_time = hold_times.mean() if len(hold_times) > 0 else 0
            
            commission_total = closed_df['commission'].fillna(0).astype(float).sum()
            slippage_total = closed_df['slippage'].fillna(0).astype(float).sum()
            
            # Create report
            report = PerformanceReport(
                period_start=start_date,
                period_end=end_date,
                total_signals=total_signals,
                executed_signals=executed_signals,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                breakeven_trades=breakeven_trades,
                win_rate=win_rate,
                total_return=total_return,
                average_return=average_return,
                average_winning_return=average_winning_return,
                average_losing_return=average_losing_return,
                best_trade=best_trade,
                worst_trade=worst_trade,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                volatility=volatility,
                profit_factor=profit_factor,
                expectancy=expectancy,
                recovery_factor=recovery_factor,
                strategy_performance=strategy_performance,
                symbol_performance=symbol_performance,
                monthly_performance=monthly_performance,
                average_hold_time=average_hold_time,
                commission_total=commission_total,
                slippage_total=slippage_total
            )
            
            # Cache the report
            self.performance_cache[cache_key] = report
            self.cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=15)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return self._create_empty_report(start_date or datetime.now(timezone.utc), 
                                           end_date or datetime.now(timezone.utc))
    
    def get_strategy_performance(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            executions = self.get_execution_history(
                strategy=strategy_name,
                start_date=start_date,
                end_date=end_date
            )
            
            if not executions:
                return {"error": "No executions found for strategy"}
            
            df = pd.DataFrame(executions)
            closed_df = df[df['outcome'].notna() & (df['outcome'] != 'pending')]
            
            if closed_df.empty:
                return {"error": "No closed positions found for strategy"}
            
            returns = closed_df['actual_return'].fillna(0).astype(float)
            
            return {
                "strategy_name": strategy_name,
                "period_days": days,
                "total_signals": len(df),
                "executed_signals": len(closed_df),
                "win_rate": len(closed_df[closed_df['outcome'] == 'win']) / len(closed_df),
                "average_return": returns.mean(),
                "total_return": returns.sum(),
                "best_trade": returns.max(),
                "worst_trade": returns.min(),
                "volatility": returns.std(),
                "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {"error": str(e)}
    
    def get_signal_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """Get signal accuracy metrics"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get signals and their outcomes
            with self._get_db_connection() as conn:
                cursor = conn.execute('''
                    SELECT 
                        sh.signal_id,
                        sh.symbol,
                        sh.strategy_name,
                        sh.confidence,
                        sh.expected_return,
                        se.actual_return,
                        se.outcome
                    FROM signal_history sh
                    LEFT JOIN signal_executions se ON sh.signal_id = se.signal_id
                    WHERE sh.generated_at >= ? AND sh.generated_at <= ?
                    AND se.outcome IS NOT NULL AND se.outcome != 'pending'
                ''', (start_date, end_date))
                
                results = cursor.fetchall()
            
            if not results:
                return {"error": "No completed signals found"}
            
            df = pd.DataFrame(results, columns=[
                'signal_id', 'symbol', 'strategy_name', 'confidence',
                'expected_return', 'actual_return', 'outcome'
            ])
            
            # Calculate accuracy metrics
            total_signals = len(df)
            correct_predictions = len(df[
                ((df['expected_return'] > 0) & (df['actual_return'] > 0)) |
                ((df['expected_return'] < 0) & (df['actual_return'] < 0))
            ])
            
            accuracy = correct_predictions / total_signals if total_signals > 0 else 0
            
            # Confidence calibration
            confidence_bins = pd.cut(df['confidence'], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
            calibration = df.groupby(confidence_bins).agg({
                'outcome': lambda x: (x == 'win').mean(),
                'confidence': 'mean'
            }).to_dict()
            
            return {
                "period_days": days,
                "total_signals": total_signals,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "confidence_calibration": calibration
            }
            
        except Exception as e:
            logger.error(f"Error getting signal accuracy: {e}")
            return {"error": str(e)}
    
    def generate_performance_charts(self, user_id: str = None, days: int = 30) -> Dict[str, str]:
        """Generate performance visualization charts"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            executions = self.get_execution_history(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not executions:
                return {"error": "No execution data found"}
            
            df = pd.DataFrame(executions)
            closed_df = df[df['outcome'].notna() & (df['outcome'] != 'pending')]
            
            if closed_df.empty:
                return {"error": "No closed positions found"}
            
            charts = {}
            
            # 1. Cumulative returns chart
            closed_df['executed_at'] = pd.to_datetime(closed_df['executed_at'])
            closed_df = closed_df.sort_values('executed_at')
            closed_df['cumulative_return'] = (1 + closed_df['actual_return'].fillna(0)).cumprod() - 1
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=closed_df['executed_at'],
                y=closed_df['cumulative_return'] * 100,
                mode='lines',
                name='Cumulative Return',
                line=dict(color='blue', width=2)
            ))
            fig1.update_layout(
                title='Cumulative Returns Over Time',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                template='plotly_white'
            )
            charts['cumulative_returns'] = self._fig_to_base64(fig1)
            
            # 2. Win/Loss distribution
            outcome_counts = closed_df['outcome'].value_counts()
            fig2 = go.Figure(data=[go.Pie(
                labels=outcome_counts.index,
                values=outcome_counts.values,
                hole=0.3
            )])
            fig2.update_layout(title='Win/Loss Distribution')
            charts['win_loss_distribution'] = self._fig_to_base64(fig2)
            
            # 3. Strategy performance comparison
            strategy_perf = closed_df.groupby('strategy_name').agg({
                'actual_return': ['mean', 'count'],
                'outcome': lambda x: (x == 'win').mean()
            }).round(4)
            
            strategy_perf.columns = ['avg_return', 'count', 'win_rate']
            strategy_perf = strategy_perf.reset_index()
            
            fig3 = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Average Return by Strategy', 'Win Rate by Strategy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig3.add_trace(
                go.Bar(x=strategy_perf['strategy_name'], y=strategy_perf['avg_return'] * 100,
                       name='Avg Return %'),
                row=1, col=1
            )
            
            fig3.add_trace(
                go.Bar(x=strategy_perf['strategy_name'], y=strategy_perf['win_rate'] * 100,
                       name='Win Rate %'),
                row=1, col=2
            )
            
            fig3.update_layout(title='Strategy Performance Comparison')
            charts['strategy_performance'] = self._fig_to_base64(fig3)
            
            # 4. Return distribution histogram
            fig4 = go.Figure(data=[go.Histogram(
                x=closed_df['actual_return'] * 100,
                nbinsx=20,
                name='Return Distribution'
            )])
            fig4.update_layout(
                title='Return Distribution',
                xaxis_title='Return (%)',
                yaxis_title='Frequency'
            )
            charts['return_distribution'] = self._fig_to_base64(fig4)
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
            return {"error": str(e)}
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with self._get_db_connection() as conn:
                # Signal history table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signal_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        priority INTEGER DEFAULT 2,
                        confidence REAL DEFAULT 0.0,
                        expected_return REAL DEFAULT 0.0,
                        max_risk REAL DEFAULT 0.0,
                        entry_price REAL DEFAULT 0.0,
                        target_price REAL DEFAULT 0.0,
                        stop_loss REAL DEFAULT 0.0,
                        time_horizon INTEGER DEFAULT 0,
                        reasoning TEXT,
                        technical_factors TEXT,
                        options_factors TEXT,
                        risk_factors TEXT,
                        score_data TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Signal executions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signal_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT UNIQUE NOT NULL,
                        signal_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        executed_at TIMESTAMP NOT NULL,
                        execution_price REAL NOT NULL,
                        position_size REAL NOT NULL,
                        commission REAL DEFAULT 0.0,
                        slippage REAL DEFAULT 0.0,
                        original_entry_price REAL DEFAULT 0.0,
                        original_target_price REAL DEFAULT 0.0,
                        original_stop_loss REAL DEFAULT 0.0,
                        original_confidence REAL DEFAULT 0.0,
                        original_expected_return REAL DEFAULT 0.0,
                        status TEXT DEFAULT 'executed',
                        current_price REAL DEFAULT 0.0,
                        unrealized_pnl REAL DEFAULT 0.0,
                        closed_at TIMESTAMP,
                        exit_price REAL,
                        realized_pnl REAL,
                        actual_return REAL,
                        outcome TEXT,
                        max_favorable_excursion REAL DEFAULT 0.0,
                        max_adverse_excursion REAL DEFAULT 0.0,
                        hold_time_hours REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (signal_id) REFERENCES signal_history (signal_id)
                    )
                ''')
                
                # Create indexes for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_history_symbol ON signal_history (symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_history_strategy ON signal_history (strategy_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_history_generated_at ON signal_history (generated_at)')
                
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_executions_user_id ON signal_executions (user_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_executions_symbol ON signal_executions (symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_executions_strategy ON signal_executions (strategy_name)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_executions_executed_at ON signal_executions (executed_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_signal_executions_outcome ON signal_executions (outcome)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_empty_report(self, start_date: datetime, end_date: datetime) -> PerformanceReport:
        """Create empty performance report"""
        return PerformanceReport(
            period_start=start_date,
            period_end=end_date,
            total_signals=0,
            executed_signals=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            total_return=0.0,
            average_return=0.0,
            average_winning_return=0.0,
            average_losing_return=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            volatility=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            recovery_factor=0.0
        )
    
    def _calculate_strategy_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate performance by strategy"""
        try:
            strategy_perf = {}
            
            for strategy in df['strategy_name'].unique():
                strategy_df = df[df['strategy_name'] == strategy]
                returns = strategy_df['actual_return'].fillna(0).astype(float)
                
                strategy_perf[strategy] = {
                    "total_trades": len(strategy_df),
                    "win_rate": len(strategy_df[strategy_df['outcome'] == 'win']) / len(strategy_df),
                    "average_return": returns.mean(),
                    "total_return": returns.sum(),
                    "volatility": returns.std(),
                    "sharpe_ratio": returns.mean() / returns.std() if returns.std() > 0 else 0
                }
            
            return strategy_perf
            
        except Exception as e:
            logger.error(f"Error calculating strategy performance: {e}")
            return {}
    
    def _calculate_symbol_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate performance by symbol"""
        try:
            symbol_perf = {}
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                returns = symbol_df['actual_return'].fillna(0).astype(float)
                
                symbol_perf[symbol] = {
                    "total_trades": len(symbol_df),
                    "win_rate": len(symbol_df[symbol_df['outcome'] == 'win']) / len(symbol_df),
                    "average_return": returns.mean(),
                    "total_return": returns.sum()
                }
            
            return symbol_perf
            
        except Exception as e:
            logger.error(f"Error calculating symbol performance: {e}")
            return {}
    
    def _calculate_monthly_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly performance"""
        try:
            df['executed_at'] = pd.to_datetime(df['executed_at'])
            df['month'] = df['executed_at'].dt.to_period('M')
            
            monthly_perf = df.groupby('month')['actual_return'].sum().to_dict()
            
            # Convert period keys to strings
            return {str(k): v for k, v in monthly_perf.items()}
            
        except Exception as e:
            logger.error(f"Error calculating monthly performance: {e}")
            return {}
    
    def _calculate_max_drawdown_duration(self, cumulative_returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        try:
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find periods of drawdown
            in_drawdown = drawdown < 0
            drawdown_periods = []
            start = None
            
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start is None:
                    start = i
                elif not is_dd and start is not None:
                    drawdown_periods.append(i - start)
                    start = None
            
            # Handle case where drawdown continues to end
            if start is not None:
                drawdown_periods.append(len(in_drawdown) - start)
            
            return max(drawdown_periods) if drawdown_periods else 0
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown duration: {e}")
            return 0
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string"""
        try:
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    def _clear_performance_cache(self):
        """Clear performance cache"""
        self.performance_cache.clear()
        self.cache_expiry.clear()

# Factory function
def create_signal_tracker(db_path: str = "signal_history.db") -> SignalTracker:
    """Create signal tracker"""
    return SignalTracker(db_path)

if __name__ == "__main__":
    # Example usage
    tracker = create_signal_tracker()
    
    # Example signal tracking
    signal_data = {
        "signal_id": "test_001",
        "symbol": "AAPL",
        "strategy_name": "Momentum Breakout",
        "signal_type": "BUY_CALL",
        "confidence": 0.75,
        "expected_return": 0.08,
        "max_risk": 0.03,
        "entry_price": 150.0,
        "target_price": 162.0,
        "stop_loss": 145.5,
        "time_horizon": 48,
        "reasoning": "Strong breakout with volume confirmation"
    }
    
    tracker.track_signal_generation(signal_data)
    
    # Example execution tracking
    execution = SignalExecution(
        signal_id="test_001",
        execution_id="exec_001",
        user_id="user_001",
        symbol="AAPL",
        strategy_name="Momentum Breakout",
        signal_type="BUY_CALL",
        executed_at=datetime.now(timezone.utc),
        execution_price=150.5,
        position_size=1000,
        original_entry_price=150.0,
        original_confidence=0.75,
        original_expected_return=0.08
    )
    
    tracker.track_signal_execution(execution)
    
    # Generate performance report
    report = tracker.generate_performance_report()
    print(f"Win Rate: {report.win_rate:.1%}")
    print(f"Average Return: {report.average_return:.1%}")
    print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")

