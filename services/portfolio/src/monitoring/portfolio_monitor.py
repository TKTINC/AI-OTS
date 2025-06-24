"""
Real-time Portfolio Monitoring System
Advanced P&L tracking and performance analytics with real-time updates
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import time
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metrics for tracking"""
    TOTAL_RETURN = "total_return"
    UNREALIZED_PNL = "unrealized_pnl"
    REALIZED_PNL = "realized_pnl"
    DAILY_PNL = "daily_pnl"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"
    VAR_95 = "var_95"
    CALMAR_RATIO = "calmar_ratio"

class MonitoringStatus(Enum):
    """Portfolio monitoring status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class PositionSnapshot:
    """Real-time position snapshot"""
    symbol: str
    quantity: int
    market_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    
    # Intraday tracking
    day_change: float
    day_change_percent: float
    high_of_day: float
    low_of_day: float
    
    # Risk metrics
    position_var: float
    beta: float
    volatility: float
    
    # Attribution
    strategy_id: Optional[str] = None
    sector: Optional[str] = None
    entry_date: Optional[datetime] = None
    
    # Real-time data
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume: int = 0

@dataclass
class PortfolioSnapshot:
    """Real-time portfolio snapshot"""
    portfolio_id: str
    timestamp: datetime
    
    # Portfolio values
    total_value: float
    cash_balance: float
    buying_power: float
    total_cost_basis: float
    
    # P&L metrics
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    daily_pnl: float
    
    # Performance metrics
    total_return_percent: float
    daily_return_percent: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    var_95: float
    
    # Risk metrics
    portfolio_var: float
    concentration_risk: float
    sector_exposure: Dict[str, float]
    strategy_exposure: Dict[str, float]
    
    # Positions
    positions: List[PositionSnapshot]
    position_count: int
    
    # Market data
    market_status: str = "open"
    last_market_update: Optional[datetime] = None

@dataclass
class PerformanceAnalytics:
    """Comprehensive performance analytics"""
    portfolio_id: str
    analysis_period: str  # "1D", "1W", "1M", "3M", "1Y", "YTD", "ITD"
    start_date: datetime
    end_date: datetime
    
    # Return metrics
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    var_95: float
    var_99: float
    expected_shortfall: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    tracking_error: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Drawdown analysis
    drawdown_periods: List[Dict[str, Any]]
    recovery_periods: List[Dict[str, Any]]
    
    # Monthly/daily returns
    period_returns: List[float]
    return_distribution: Dict[str, float]

class PortfolioMonitor:
    """
    Real-time portfolio monitoring system
    Tracks P&L, performance metrics, and risk in real-time
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Monitoring parameters
        self.update_frequency = self.config.get('update_frequency', 1.0)  # seconds
        self.history_retention = self.config.get('history_retention', 30)  # days
        self.performance_lookback = self.config.get('performance_lookback', 252)  # trading days
        
        # Data storage
        self.portfolio_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.position_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self.performance_cache: Dict[str, PerformanceAnalytics] = {}
        
        # Real-time tracking
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.monitoring_status: Dict[str, MonitoringStatus] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        
        # Market data integration
        self.market_data_callbacks = []
        self.last_market_update = {}
        
        # Performance calculation cache
        self.calculation_cache = {}
        self.cache_expiry = {}
        
        logger.info("Portfolio Monitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for portfolio monitoring"""
        return {
            'update_frequency': 1.0,  # Update every second
            'history_retention': 30,  # Keep 30 days of history
            'performance_lookback': 252,  # 1 year for performance calculations
            'cache_duration': 60,  # Cache performance calculations for 60 seconds
            'risk_free_rate': 0.02,  # 2% risk-free rate
            'benchmark_symbol': 'SPY',  # S&P 500 benchmark
            'max_drawdown_threshold': 0.10,  # 10% max drawdown alert
            'var_confidence': 0.95,  # 95% VaR confidence
            'enable_real_time': True,
            'enable_alerts': True
        }
    
    def start_monitoring(self, portfolio_id: str, ibkr_client=None) -> bool:
        """
        Start real-time monitoring for a portfolio
        
        Args:
            portfolio_id: Portfolio to monitor
            ibkr_client: IBKR client for real-time data
            
        Returns:
            True if monitoring started successfully
        """
        try:
            if portfolio_id in self.monitoring_threads:
                logger.warning(f"Monitoring already active for portfolio {portfolio_id}")
                return True
            
            # Create stop event
            stop_event = threading.Event()
            self.stop_events[portfolio_id] = stop_event
            
            # Create monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(portfolio_id, ibkr_client, stop_event),
                daemon=True
            )
            
            # Start monitoring
            monitor_thread.start()
            self.monitoring_threads[portfolio_id] = monitor_thread
            self.monitoring_status[portfolio_id] = MonitoringStatus.ACTIVE
            
            logger.info(f"Started real-time monitoring for portfolio {portfolio_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring for portfolio {portfolio_id}: {e}")
            self.monitoring_status[portfolio_id] = MonitoringStatus.ERROR
            return False
    
    def stop_monitoring(self, portfolio_id: str) -> bool:
        """Stop real-time monitoring for a portfolio"""
        try:
            if portfolio_id not in self.monitoring_threads:
                logger.warning(f"No active monitoring for portfolio {portfolio_id}")
                return True
            
            # Signal stop
            self.stop_events[portfolio_id].set()
            
            # Wait for thread to finish
            self.monitoring_threads[portfolio_id].join(timeout=5.0)
            
            # Cleanup
            del self.monitoring_threads[portfolio_id]
            del self.stop_events[portfolio_id]
            self.monitoring_status[portfolio_id] = MonitoringStatus.STOPPED
            
            logger.info(f"Stopped monitoring for portfolio {portfolio_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping monitoring for portfolio {portfolio_id}: {e}")
            return False
    
    def _monitoring_loop(self, portfolio_id: str, ibkr_client, stop_event: threading.Event):
        """Main monitoring loop for real-time updates"""
        try:
            logger.info(f"Starting monitoring loop for portfolio {portfolio_id}")
            
            while not stop_event.is_set():
                try:
                    # Get current portfolio data
                    portfolio_data = self._get_portfolio_data(portfolio_id, ibkr_client)
                    
                    if portfolio_data:
                        # Create portfolio snapshot
                        snapshot = self._create_portfolio_snapshot(portfolio_id, portfolio_data)
                        
                        # Store snapshot
                        self.portfolio_snapshots[portfolio_id].append(snapshot)
                        
                        # Update position history
                        self._update_position_history(portfolio_id, snapshot.positions)
                        
                        # Check for alerts
                        self._check_alerts(portfolio_id, snapshot)
                        
                        # Clear performance cache if needed
                        self._clear_expired_cache(portfolio_id)
                    
                    # Sleep until next update
                    stop_event.wait(self.update_frequency)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop for {portfolio_id}: {e}")
                    time.sleep(self.update_frequency)
            
            logger.info(f"Monitoring loop stopped for portfolio {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop for {portfolio_id}: {e}")
            self.monitoring_status[portfolio_id] = MonitoringStatus.ERROR
    
    def _get_portfolio_data(self, portfolio_id: str, ibkr_client) -> Optional[Dict[str, Any]]:
        """Get current portfolio data from IBKR or database"""
        try:
            # In production, this would fetch from IBKR client or database
            # Mock implementation for now
            mock_data = {
                'portfolio_id': portfolio_id,
                'total_value': 100000 + np.random.normal(0, 1000),
                'cash_balance': 20000 + np.random.normal(0, 500),
                'buying_power': 40000 + np.random.normal(0, 1000),
                'positions': [
                    {
                        'symbol': 'AAPL',
                        'quantity': 100,
                        'market_price': 150.0 + np.random.normal(0, 2),
                        'cost_basis': 145.0,
                        'strategy_id': 'momentum_breakout',
                        'sector': 'technology'
                    },
                    {
                        'symbol': 'MSFT',
                        'quantity': 50,
                        'market_price': 300.0 + np.random.normal(0, 5),
                        'cost_basis': 295.0,
                        'strategy_id': 'volatility_squeeze',
                        'sector': 'technology'
                    }
                ]
            }
            
            return mock_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return None
    
    def _create_portfolio_snapshot(self, portfolio_id: str, portfolio_data: Dict[str, Any]) -> PortfolioSnapshot:
        """Create portfolio snapshot from current data"""
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Create position snapshots
            positions = []
            total_cost_basis = 0.0
            total_unrealized_pnl = 0.0
            
            for pos_data in portfolio_data.get('positions', []):
                # Calculate position metrics
                market_value = pos_data['quantity'] * pos_data['market_price']
                position_cost_basis = pos_data['quantity'] * pos_data['cost_basis']
                unrealized_pnl = market_value - position_cost_basis
                unrealized_pnl_percent = unrealized_pnl / position_cost_basis if position_cost_basis > 0 else 0
                
                # Get intraday data (mock)
                day_change = np.random.normal(0, market_value * 0.02)
                day_change_percent = day_change / market_value if market_value > 0 else 0
                
                position = PositionSnapshot(
                    symbol=pos_data['symbol'],
                    quantity=pos_data['quantity'],
                    market_price=pos_data['market_price'],
                    market_value=market_value,
                    cost_basis=position_cost_basis,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    day_change=day_change,
                    day_change_percent=day_change_percent,
                    high_of_day=pos_data['market_price'] * 1.02,
                    low_of_day=pos_data['market_price'] * 0.98,
                    position_var=market_value * 0.02,  # 2% VaR estimate
                    beta=1.0 + np.random.normal(0, 0.2),
                    volatility=0.20 + np.random.normal(0, 0.05),
                    strategy_id=pos_data.get('strategy_id'),
                    sector=pos_data.get('sector'),
                    last_update=timestamp
                )
                
                positions.append(position)
                total_cost_basis += position_cost_basis
                total_unrealized_pnl += unrealized_pnl
            
            # Calculate portfolio metrics
            total_value = portfolio_data['total_value']
            total_pnl = total_unrealized_pnl  # + realized PnL (would come from trades)
            total_return_percent = total_pnl / total_cost_basis if total_cost_basis > 0 else 0
            
            # Get previous snapshot for daily calculations
            daily_pnl = 0.0
            daily_return_percent = 0.0
            if self.portfolio_snapshots[portfolio_id]:
                prev_snapshot = self.portfolio_snapshots[portfolio_id][-1]
                daily_pnl = total_value - prev_snapshot.total_value
                daily_return_percent = daily_pnl / prev_snapshot.total_value if prev_snapshot.total_value > 0 else 0
            
            # Calculate risk metrics
            portfolio_var = total_value * 0.02  # Simplified VaR
            concentration_risk = self._calculate_concentration_risk(positions)
            sector_exposure = self._calculate_sector_exposure(positions, total_value)
            strategy_exposure = self._calculate_strategy_exposure(positions, total_value)
            
            # Performance metrics (simplified)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_id)
            max_drawdown = self._calculate_max_drawdown(portfolio_id)
            volatility = self._calculate_portfolio_volatility(portfolio_id)
            beta = self._calculate_portfolio_beta(positions)
            
            snapshot = PortfolioSnapshot(
                portfolio_id=portfolio_id,
                timestamp=timestamp,
                total_value=total_value,
                cash_balance=portfolio_data['cash_balance'],
                buying_power=portfolio_data['buying_power'],
                total_cost_basis=total_cost_basis,
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=0.0,  # Would come from trade history
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                total_return_percent=total_return_percent,
                daily_return_percent=daily_return_percent,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                var_95=portfolio_var,
                portfolio_var=portfolio_var,
                concentration_risk=concentration_risk,
                sector_exposure=sector_exposure,
                strategy_exposure=strategy_exposure,
                positions=positions,
                position_count=len(positions),
                market_status="open",
                last_market_update=timestamp
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating portfolio snapshot: {e}")
            raise
    
    def _update_position_history(self, portfolio_id: str, positions: List[PositionSnapshot]):
        """Update position history for trend analysis"""
        try:
            for position in positions:
                symbol = position.symbol
                
                # Store position data point
                data_point = {
                    'timestamp': position.last_update,
                    'market_price': position.market_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'day_change': position.day_change
                }
                
                self.position_history[portfolio_id][symbol].append(data_point)
                
        except Exception as e:
            logger.error(f"Error updating position history: {e}")
    
    def _check_alerts(self, portfolio_id: str, snapshot: PortfolioSnapshot):
        """Check for alert conditions"""
        try:
            alerts = []
            
            # Max drawdown alert
            if snapshot.max_drawdown > self.config.get('max_drawdown_threshold', 0.10):
                alerts.append(f"Max drawdown exceeded: {snapshot.max_drawdown:.2%}")
            
            # Large daily loss alert
            if snapshot.daily_return_percent < -0.05:  # 5% daily loss
                alerts.append(f"Large daily loss: {snapshot.daily_return_percent:.2%}")
            
            # Position concentration alert
            if snapshot.concentration_risk > 0.30:  # 30% in single position
                alerts.append(f"High concentration risk: {snapshot.concentration_risk:.2%}")
            
            # VaR breach alert
            if abs(snapshot.daily_pnl) > snapshot.var_95:
                alerts.append(f"VaR breach: Daily P&L {snapshot.daily_pnl:.0f} exceeds VaR {snapshot.var_95:.0f}")
            
            if alerts and self.config.get('enable_alerts', True):
                logger.warning(f"Portfolio {portfolio_id} alerts: {'; '.join(alerts)}")
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def get_current_snapshot(self, portfolio_id: str) -> Optional[PortfolioSnapshot]:
        """Get the most recent portfolio snapshot"""
        try:
            if portfolio_id in self.portfolio_snapshots and self.portfolio_snapshots[portfolio_id]:
                return self.portfolio_snapshots[portfolio_id][-1]
            return None
            
        except Exception as e:
            logger.error(f"Error getting current snapshot: {e}")
            return None
    
    def get_performance_analytics(self, portfolio_id: str, period: str = "1M") -> Optional[PerformanceAnalytics]:
        """
        Get comprehensive performance analytics
        
        Args:
            portfolio_id: Portfolio to analyze
            period: Analysis period ("1D", "1W", "1M", "3M", "1Y", "YTD", "ITD")
            
        Returns:
            Performance analytics or None
        """
        try:
            # Check cache first
            cache_key = f"{portfolio_id}_{period}"
            if cache_key in self.performance_cache:
                cached_time = self.cache_expiry.get(cache_key, datetime.min)
                if datetime.now(timezone.utc) - cached_time < timedelta(seconds=self.config.get('cache_duration', 60)):
                    return self.performance_cache[cache_key]
            
            # Calculate performance analytics
            analytics = self._calculate_performance_analytics(portfolio_id, period)
            
            # Cache result
            self.performance_cache[cache_key] = analytics
            self.cache_expiry[cache_key] = datetime.now(timezone.utc)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return None
    
    def _calculate_performance_analytics(self, portfolio_id: str, period: str) -> PerformanceAnalytics:
        """Calculate comprehensive performance analytics"""
        try:
            # Get snapshots for the period
            snapshots = self._get_snapshots_for_period(portfolio_id, period)
            
            if not snapshots:
                raise ValueError(f"No snapshots available for period {period}")
            
            # Calculate date range
            start_date = snapshots[0].timestamp
            end_date = snapshots[-1].timestamp
            
            # Calculate returns
            start_value = snapshots[0].total_value
            end_value = snapshots[-1].total_value
            total_return = (end_value - start_value) / start_value if start_value > 0 else 0
            
            # Calculate period returns
            period_returns = []
            for i in range(1, len(snapshots)):
                prev_value = snapshots[i-1].total_value
                curr_value = snapshots[i].total_value
                period_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
                period_returns.append(period_return)
            
            # Risk metrics
            volatility = np.std(period_returns) * np.sqrt(252) if period_returns else 0  # Annualized
            downside_returns = [r for r in period_returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            
            # Drawdown analysis
            max_drawdown, drawdown_periods = self._calculate_detailed_drawdown(snapshots)
            
            # Risk-adjusted returns
            risk_free_rate = self.config.get('risk_free_rate', 0.02)
            sharpe_ratio = (total_return - risk_free_rate) / volatility if volatility > 0 else 0
            sortino_ratio = (total_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR calculations
            var_95 = np.percentile(period_returns, 5) if period_returns else 0
            var_99 = np.percentile(period_returns, 1) if period_returns else 0
            expected_shortfall = np.mean([r for r in period_returns if r <= var_95]) if period_returns else 0
            
            # Annualized return
            days = (end_date - start_date).days
            annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
            
            analytics = PerformanceAnalytics(
                portfolio_id=portfolio_id,
                analysis_period=period,
                start_date=start_date,
                end_date=end_date,
                total_return=total_return,
                annualized_return=annualized_return,
                compound_annual_growth_rate=annualized_return,
                volatility=volatility,
                downside_volatility=downside_volatility,
                max_drawdown=max_drawdown,
                max_drawdown_duration=0,  # Would need detailed calculation
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                information_ratio=0.0,  # Would need benchmark comparison
                benchmark_return=0.0,  # Would fetch benchmark data
                alpha=0.0,  # Would calculate vs benchmark
                beta=1.0,  # Would calculate vs benchmark
                tracking_error=0.0,  # Would calculate vs benchmark
                total_trades=0,  # Would come from trade history
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                drawdown_periods=drawdown_periods,
                recovery_periods=[],
                period_returns=period_returns,
                return_distribution=self._calculate_return_distribution(period_returns)
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error calculating performance analytics: {e}")
            raise
    
    def get_real_time_pnl(self, portfolio_id: str) -> Dict[str, Any]:
        """Get real-time P&L summary"""
        try:
            snapshot = self.get_current_snapshot(portfolio_id)
            if not snapshot:
                return {'error': 'No current snapshot available'}
            
            return {
                'portfolio_id': portfolio_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'total_value': snapshot.total_value,
                'total_pnl': snapshot.total_pnl,
                'total_return_percent': snapshot.total_return_percent,
                'daily_pnl': snapshot.daily_pnl,
                'daily_return_percent': snapshot.daily_return_percent,
                'unrealized_pnl': snapshot.total_unrealized_pnl,
                'realized_pnl': snapshot.total_realized_pnl,
                'cash_balance': snapshot.cash_balance,
                'buying_power': snapshot.buying_power,
                'position_count': snapshot.position_count,
                'market_status': snapshot.market_status
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time P&L: {e}")
            return {'error': str(e)}
    
    def get_position_details(self, portfolio_id: str, symbol: str = None) -> List[Dict[str, Any]]:
        """Get detailed position information"""
        try:
            snapshot = self.get_current_snapshot(portfolio_id)
            if not snapshot:
                return []
            
            positions = snapshot.positions
            if symbol:
                positions = [pos for pos in positions if pos.symbol == symbol]
            
            position_details = []
            for pos in positions:
                details = {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'market_price': pos.market_price,
                    'market_value': pos.market_value,
                    'cost_basis': pos.cost_basis,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                    'day_change': pos.day_change,
                    'day_change_percent': pos.day_change_percent,
                    'high_of_day': pos.high_of_day,
                    'low_of_day': pos.low_of_day,
                    'strategy_id': pos.strategy_id,
                    'sector': pos.sector,
                    'beta': pos.beta,
                    'volatility': pos.volatility,
                    'position_var': pos.position_var,
                    'last_update': pos.last_update.isoformat()
                }
                position_details.append(details)
            
            return position_details
            
        except Exception as e:
            logger.error(f"Error getting position details: {e}")
            return []
    
    # Helper methods
    def _get_snapshots_for_period(self, portfolio_id: str, period: str) -> List[PortfolioSnapshot]:
        """Get snapshots for specified period"""
        try:
            if portfolio_id not in self.portfolio_snapshots:
                return []
            
            snapshots = list(self.portfolio_snapshots[portfolio_id])
            if not snapshots:
                return []
            
            # Calculate cutoff date
            now = datetime.now(timezone.utc)
            if period == "1D":
                cutoff = now - timedelta(days=1)
            elif period == "1W":
                cutoff = now - timedelta(weeks=1)
            elif period == "1M":
                cutoff = now - timedelta(days=30)
            elif period == "3M":
                cutoff = now - timedelta(days=90)
            elif period == "1Y":
                cutoff = now - timedelta(days=365)
            elif period == "YTD":
                cutoff = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            else:  # ITD (Inception to Date)
                cutoff = datetime.min.replace(tzinfo=timezone.utc)
            
            # Filter snapshots
            filtered_snapshots = [s for s in snapshots if s.timestamp >= cutoff]
            return filtered_snapshots
            
        except Exception as e:
            logger.error(f"Error getting snapshots for period: {e}")
            return []
    
    def _calculate_concentration_risk(self, positions: List[PositionSnapshot]) -> float:
        """Calculate concentration risk (largest position as % of portfolio)"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.market_value for pos in positions)
            if total_value == 0:
                return 0.0
            
            max_position_value = max(pos.market_value for pos in positions)
            return max_position_value / total_value
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_sector_exposure(self, positions: List[PositionSnapshot], total_value: float) -> Dict[str, float]:
        """Calculate sector exposure percentages"""
        try:
            sector_exposure = defaultdict(float)
            
            for pos in positions:
                sector = pos.sector or 'unknown'
                sector_exposure[sector] += pos.market_value / total_value if total_value > 0 else 0
            
            return dict(sector_exposure)
            
        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return {}
    
    def _calculate_strategy_exposure(self, positions: List[PositionSnapshot], total_value: float) -> Dict[str, float]:
        """Calculate strategy exposure percentages"""
        try:
            strategy_exposure = defaultdict(float)
            
            for pos in positions:
                strategy = pos.strategy_id or 'unknown'
                strategy_exposure[strategy] += pos.market_value / total_value if total_value > 0 else 0
            
            return dict(strategy_exposure)
            
        except Exception as e:
            logger.error(f"Error calculating strategy exposure: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, portfolio_id: str) -> float:
        """Calculate Sharpe ratio from recent snapshots"""
        try:
            snapshots = self._get_snapshots_for_period(portfolio_id, "1M")
            if len(snapshots) < 2:
                return 0.0
            
            returns = []
            for i in range(1, len(snapshots)):
                prev_value = snapshots[i-1].total_value
                curr_value = snapshots[i].total_value
                ret = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
                returns.append(ret)
            
            if not returns:
                return 0.0
            
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            risk_free_rate = self.config.get('risk_free_rate', 0.02) / 252  # Daily risk-free rate
            
            return (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, portfolio_id: str) -> float:
        """Calculate maximum drawdown from recent snapshots"""
        try:
            snapshots = self._get_snapshots_for_period(portfolio_id, "3M")
            if not snapshots:
                return 0.0
            
            values = [s.total_value for s in snapshots]
            peak = values[0]
            max_dd = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_portfolio_volatility(self, portfolio_id: str) -> float:
        """Calculate portfolio volatility from recent snapshots"""
        try:
            snapshots = self._get_snapshots_for_period(portfolio_id, "1M")
            if len(snapshots) < 2:
                return 0.0
            
            returns = []
            for i in range(1, len(snapshots)):
                prev_value = snapshots[i-1].total_value
                curr_value = snapshots[i].total_value
                ret = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
                returns.append(ret)
            
            return np.std(returns) * np.sqrt(252) if returns else 0.0  # Annualized
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    def _calculate_portfolio_beta(self, positions: List[PositionSnapshot]) -> float:
        """Calculate portfolio beta (weighted average)"""
        try:
            if not positions:
                return 1.0
            
            total_value = sum(pos.market_value for pos in positions)
            if total_value == 0:
                return 1.0
            
            weighted_beta = 0.0
            for pos in positions:
                weight = pos.market_value / total_value
                weighted_beta += weight * pos.beta
            
            return weighted_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    def _calculate_detailed_drawdown(self, snapshots: List[PortfolioSnapshot]) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate detailed drawdown analysis"""
        try:
            if not snapshots:
                return 0.0, []
            
            values = [s.total_value for s in snapshots]
            timestamps = [s.timestamp for s in snapshots]
            
            peak = values[0]
            peak_date = timestamps[0]
            max_dd = 0.0
            drawdown_periods = []
            current_dd_start = None
            
            for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                if value > peak:
                    # New peak - end any current drawdown
                    if current_dd_start is not None:
                        drawdown_periods.append({
                            'start_date': current_dd_start,
                            'end_date': timestamp,
                            'duration_days': (timestamp - current_dd_start).days,
                            'max_drawdown': max_dd
                        })
                        current_dd_start = None
                    
                    peak = value
                    peak_date = timestamp
                else:
                    # In drawdown
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    
                    if current_dd_start is None:
                        current_dd_start = peak_date
                    
                    max_dd = max(max_dd, drawdown)
            
            # Handle ongoing drawdown
            if current_dd_start is not None:
                drawdown_periods.append({
                    'start_date': current_dd_start,
                    'end_date': timestamps[-1],
                    'duration_days': (timestamps[-1] - current_dd_start).days,
                    'max_drawdown': max_dd,
                    'ongoing': True
                })
            
            return max_dd, drawdown_periods
            
        except Exception as e:
            logger.error(f"Error calculating detailed drawdown: {e}")
            return 0.0, []
    
    def _calculate_return_distribution(self, returns: List[float]) -> Dict[str, float]:
        """Calculate return distribution statistics"""
        try:
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            return {
                'mean': float(np.mean(returns_array)),
                'median': float(np.median(returns_array)),
                'std': float(np.std(returns_array)),
                'skewness': float(self._calculate_skewness(returns_array)),
                'kurtosis': float(self._calculate_kurtosis(returns_array)),
                'min': float(np.min(returns_array)),
                'max': float(np.max(returns_array)),
                'percentile_5': float(np.percentile(returns_array, 5)),
                'percentile_25': float(np.percentile(returns_array, 25)),
                'percentile_75': float(np.percentile(returns_array, 75)),
                'percentile_95': float(np.percentile(returns_array, 95))
            }
            
        except Exception as e:
            logger.error(f"Error calculating return distribution: {e}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        except:
            return 0.0
    
    def _clear_expired_cache(self, portfolio_id: str):
        """Clear expired performance cache entries"""
        try:
            current_time = datetime.now(timezone.utc)
            cache_duration = timedelta(seconds=self.config.get('cache_duration', 60))
            
            expired_keys = []
            for key, expiry_time in self.cache_expiry.items():
                if key.startswith(portfolio_id) and current_time - expiry_time > cache_duration:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.performance_cache.pop(key, None)
                self.cache_expiry.pop(key, None)
                
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create portfolio monitor
    monitor = PortfolioMonitor()
    
    # Start monitoring a portfolio
    portfolio_id = "test_portfolio_001"
    monitor.start_monitoring(portfolio_id)
    
    # Wait a bit for some data
    time.sleep(5)
    
    # Get current snapshot
    snapshot = monitor.get_current_snapshot(portfolio_id)
    if snapshot:
        print(f"Portfolio Snapshot:")
        print(f"  Total Value: ${snapshot.total_value:,.2f}")
        print(f"  Total P&L: ${snapshot.total_pnl:,.2f} ({snapshot.total_return_percent:.2%})")
        print(f"  Daily P&L: ${snapshot.daily_pnl:,.2f} ({snapshot.daily_return_percent:.2%})")
        print(f"  Positions: {snapshot.position_count}")
        print(f"  Sharpe Ratio: {snapshot.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {snapshot.max_drawdown:.2%}")
        print(f"  Portfolio VaR: ${snapshot.var_95:,.0f}")
    
    # Get performance analytics
    analytics = monitor.get_performance_analytics(portfolio_id, "1M")
    if analytics:
        print(f"\nPerformance Analytics (1M):")
        print(f"  Total Return: {analytics.total_return:.2%}")
        print(f"  Annualized Return: {analytics.annualized_return:.2%}")
        print(f"  Volatility: {analytics.volatility:.2%}")
        print(f"  Sharpe Ratio: {analytics.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio: {analytics.sortino_ratio:.3f}")
        print(f"  Max Drawdown: {analytics.max_drawdown:.2%}")
        print(f"  VaR 95%: {analytics.var_95:.4f}")
    
    # Get real-time P&L
    pnl_summary = monitor.get_real_time_pnl(portfolio_id)
    print(f"\nReal-time P&L Summary:")
    for key, value in pnl_summary.items():
        if isinstance(value, float):
            if 'percent' in key:
                print(f"  {key}: {value:.2%}")
            elif 'pnl' in key or 'value' in key or 'balance' in key or 'power' in key:
                print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Stop monitoring
    monitor.stop_monitoring(portfolio_id)
    
    print("\nPortfolio monitoring example completed")

