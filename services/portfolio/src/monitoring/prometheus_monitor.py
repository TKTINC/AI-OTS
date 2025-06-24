"""
Prometheus Monitoring for Portfolio Management Service
Custom metrics for portfolio performance, risk, and IBKR integration
"""

import time
import threading
from typing import Dict, List, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class PortfolioPrometheusMonitor:
    """Prometheus monitoring for Portfolio Management Service"""
    
    def __init__(self, port: int = 8006):
        self.port = port
        self.monitoring_active = False
        
        # Portfolio Management Metrics
        self.portfolio_total_value = Gauge(
            'portfolio_total_value_dollars',
            'Total portfolio value in dollars',
            ['portfolio_id', 'account_id']
        )
        
        self.portfolio_pnl = Gauge(
            'portfolio_pnl_dollars',
            'Portfolio profit and loss in dollars',
            ['portfolio_id', 'account_id', 'pnl_type']  # unrealized, realized, daily
        )
        
        self.portfolio_return = Gauge(
            'portfolio_return_percent',
            'Portfolio return percentage',
            ['portfolio_id', 'account_id', 'period']  # daily, total
        )
        
        self.portfolio_positions_count = Gauge(
            'portfolio_positions_count',
            'Number of positions in portfolio',
            ['portfolio_id', 'account_id']
        )
        
        self.portfolio_cash_balance = Gauge(
            'portfolio_cash_balance_dollars',
            'Portfolio cash balance in dollars',
            ['portfolio_id', 'account_id']
        )
        
        self.portfolio_buying_power = Gauge(
            'portfolio_buying_power_dollars',
            'Portfolio buying power in dollars',
            ['portfolio_id', 'account_id']
        )
        
        # Risk Metrics
        self.portfolio_sharpe_ratio = Gauge(
            'portfolio_sharpe_ratio',
            'Portfolio Sharpe ratio',
            ['portfolio_id', 'account_id']
        )
        
        self.portfolio_max_drawdown = Gauge(
            'portfolio_max_drawdown_percent',
            'Portfolio maximum drawdown percentage',
            ['portfolio_id', 'account_id']
        )
        
        self.portfolio_volatility = Gauge(
            'portfolio_volatility_percent',
            'Portfolio volatility percentage',
            ['portfolio_id', 'account_id']
        )
        
        self.portfolio_var = Gauge(
            'portfolio_var_dollars',
            'Portfolio Value at Risk in dollars',
            ['portfolio_id', 'account_id', 'confidence']  # 95, 99
        )
        
        self.portfolio_beta = Gauge(
            'portfolio_beta',
            'Portfolio beta relative to market',
            ['portfolio_id', 'account_id']
        )
        
        # Position Sizing Metrics
        self.position_sizing_calculations = Counter(
            'position_sizing_calculations_total',
            'Total position sizing calculations',
            ['method', 'status']  # success, error
        )
        
        self.position_sizing_duration = Histogram(
            'position_sizing_duration_seconds',
            'Position sizing calculation duration',
            ['method'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.position_size_distribution = Histogram(
            'position_size_percent',
            'Distribution of position sizes as percentage of portfolio',
            ['strategy_id', 'symbol'],
            buckets=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        )
        
        # Portfolio Optimization Metrics
        self.optimization_calculations = Counter(
            'optimization_calculations_total',
            'Total portfolio optimization calculations',
            ['objective', 'status']  # max_sharpe, min_variance, etc.
        )
        
        self.optimization_duration = Histogram(
            'optimization_duration_seconds',
            'Portfolio optimization calculation duration',
            ['objective'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.optimization_improvement = Gauge(
            'optimization_improvement_percent',
            'Portfolio optimization improvement percentage',
            ['portfolio_id', 'objective', 'metric']  # return, sharpe, volatility
        )
        
        # Risk Budgeting Metrics
        self.risk_budget_utilization = Gauge(
            'risk_budget_utilization_percent',
            'Risk budget utilization percentage',
            ['budget_id', 'entity_id', 'risk_metric']
        )
        
        self.risk_budget_violations = Counter(
            'risk_budget_violations_total',
            'Total risk budget violations',
            ['budget_id', 'entity_id', 'violation_type']  # over_budget, critical
        )
        
        self.risk_allocation_updates = Counter(
            'risk_allocation_updates_total',
            'Total risk allocation updates',
            ['budget_id', 'status']  # success, error
        )
        
        # IBKR Integration Metrics
        self.ibkr_connection_status = Gauge(
            'ibkr_connection_status',
            'IBKR connection status (1=connected, 0=disconnected)',
            ['account_id', 'account_type']  # paper, live
        )
        
        self.ibkr_account_switches = Counter(
            'ibkr_account_switches_total',
            'Total IBKR account switches',
            ['from_account', 'to_account', 'status']  # success, error
        )
        
        self.ibkr_orders_placed = Counter(
            'ibkr_orders_placed_total',
            'Total orders placed through IBKR',
            ['account_id', 'order_type', 'symbol', 'status']  # filled, cancelled, error
        )
        
        self.ibkr_order_latency = Histogram(
            'ibkr_order_latency_seconds',
            'IBKR order placement latency',
            ['account_id', 'order_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Performance Attribution Metrics
        self.attribution_calculations = Counter(
            'attribution_calculations_total',
            'Total performance attribution calculations',
            ['portfolio_id', 'method', 'level']  # brinson, strategy, sector
        )
        
        self.attribution_effects = Gauge(
            'attribution_effects_percent',
            'Performance attribution effects',
            ['portfolio_id', 'segment_id', 'effect_type']  # allocation, selection, interaction
        )
        
        self.top_contributors = Gauge(
            'top_contributors_percent',
            'Top performance contributors',
            ['portfolio_id', 'contributor_id', 'contributor_type']  # strategy, sector, symbol
        )
        
        # Monitoring Metrics
        self.monitoring_active_portfolios = Gauge(
            'monitoring_active_portfolios',
            'Number of actively monitored portfolios'
        )
        
        self.monitoring_snapshots_generated = Counter(
            'monitoring_snapshots_generated_total',
            'Total monitoring snapshots generated',
            ['portfolio_id']
        )
        
        self.monitoring_update_duration = Histogram(
            'monitoring_update_duration_seconds',
            'Portfolio monitoring update duration',
            ['portfolio_id'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        )
        
        # System Metrics
        self.service_info = Info(
            'portfolio_service_info',
            'Portfolio Management Service information'
        )
        
        self.service_uptime = Gauge(
            'portfolio_service_uptime_seconds',
            'Portfolio Management Service uptime in seconds'
        )
        
        self.api_requests = Counter(
            'portfolio_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code']
        )
        
        self.api_request_duration = Histogram(
            'portfolio_api_request_duration_seconds',
            'API request duration',
            ['endpoint', 'method'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Initialize service info
        self.service_info.info({
            'version': '1.0.0',
            'service': 'portfolio-management',
            'description': 'AI Options Trading System - Portfolio Management Service'
        })
        
        self.start_time = time.time()
    
    def start_monitoring(self):
        """Start Prometheus monitoring server"""
        try:
            start_http_server(self.port)
            self.monitoring_active = True
            
            # Start uptime tracking
            self._start_uptime_tracking()
            
            logger.info(f"Prometheus monitoring started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Error starting Prometheus monitoring: {e}")
            raise
    
    def _start_uptime_tracking(self):
        """Start uptime tracking in background thread"""
        def update_uptime():
            while self.monitoring_active:
                uptime = time.time() - self.start_time
                self.service_uptime.set(uptime)
                time.sleep(10)  # Update every 10 seconds
        
        uptime_thread = threading.Thread(target=update_uptime, daemon=True)
        uptime_thread.start()
    
    def stop_monitoring(self):
        """Stop Prometheus monitoring"""
        self.monitoring_active = False
        logger.info("Prometheus monitoring stopped")
    
    # Portfolio Metrics Methods
    def update_portfolio_metrics(self, portfolio_id: str, account_id: str, snapshot: dict):
        """Update portfolio metrics from snapshot"""
        try:
            # Basic portfolio metrics
            self.portfolio_total_value.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('total_value', 0))
            
            self.portfolio_positions_count.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('position_count', 0))
            
            self.portfolio_cash_balance.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('cash_balance', 0))
            
            self.portfolio_buying_power.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('buying_power', 0))
            
            # P&L metrics
            self.portfolio_pnl.labels(
                portfolio_id=portfolio_id,
                account_id=account_id,
                pnl_type='total'
            ).set(snapshot.get('total_pnl', 0))
            
            self.portfolio_pnl.labels(
                portfolio_id=portfolio_id,
                account_id=account_id,
                pnl_type='daily'
            ).set(snapshot.get('daily_pnl', 0))
            
            # Return metrics
            self.portfolio_return.labels(
                portfolio_id=portfolio_id,
                account_id=account_id,
                period='total'
            ).set(snapshot.get('total_return_percent', 0))
            
            self.portfolio_return.labels(
                portfolio_id=portfolio_id,
                account_id=account_id,
                period='daily'
            ).set(snapshot.get('daily_return_percent', 0))
            
            # Risk metrics
            self.portfolio_sharpe_ratio.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('sharpe_ratio', 0))
            
            self.portfolio_max_drawdown.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('max_drawdown', 0))
            
            self.portfolio_volatility.labels(
                portfolio_id=portfolio_id,
                account_id=account_id
            ).set(snapshot.get('volatility', 0))
            
            self.portfolio_var.labels(
                portfolio_id=portfolio_id,
                account_id=account_id,
                confidence='95'
            ).set(snapshot.get('var_95', 0))
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def record_position_sizing(self, method: str, duration: float, success: bool, position_size: float, strategy_id: str, symbol: str):
        """Record position sizing metrics"""
        try:
            status = 'success' if success else 'error'
            
            self.position_sizing_calculations.labels(
                method=method,
                status=status
            ).inc()
            
            if success:
                self.position_sizing_duration.labels(method=method).observe(duration)
                self.position_size_distribution.labels(
                    strategy_id=strategy_id,
                    symbol=symbol
                ).observe(position_size)
            
        except Exception as e:
            logger.error(f"Error recording position sizing metrics: {e}")
    
    def record_optimization(self, objective: str, duration: float, success: bool, portfolio_id: str = None, improvement: dict = None):
        """Record portfolio optimization metrics"""
        try:
            status = 'success' if success else 'error'
            
            self.optimization_calculations.labels(
                objective=objective,
                status=status
            ).inc()
            
            if success:
                self.optimization_duration.labels(objective=objective).observe(duration)
                
                if portfolio_id and improvement:
                    for metric, value in improvement.items():
                        self.optimization_improvement.labels(
                            portfolio_id=portfolio_id,
                            objective=objective,
                            metric=metric
                        ).set(value)
            
        except Exception as e:
            logger.error(f"Error recording optimization metrics: {e}")
    
    def update_risk_budget_metrics(self, budget_id: str, allocations: dict, violations: list):
        """Update risk budgeting metrics"""
        try:
            # Update utilization for each entity
            for entity_id, allocation in allocations.items():
                utilization = allocation.get('utilization_percent', 0)
                risk_metric = allocation.get('risk_metric', 'var_95')
                
                self.risk_budget_utilization.labels(
                    budget_id=budget_id,
                    entity_id=entity_id,
                    risk_metric=risk_metric
                ).set(utilization)
            
            # Record violations
            for violation in violations:
                self.risk_budget_violations.labels(
                    budget_id=budget_id,
                    entity_id=violation.get('entity_id', 'unknown'),
                    violation_type=violation.get('type', 'over_budget')
                ).inc()
            
        except Exception as e:
            logger.error(f"Error updating risk budget metrics: {e}")
    
    def record_ibkr_metrics(self, account_id: str, account_type: str, connected: bool):
        """Record IBKR connection metrics"""
        try:
            connection_status = 1 if connected else 0
            self.ibkr_connection_status.labels(
                account_id=account_id,
                account_type=account_type
            ).set(connection_status)
            
        except Exception as e:
            logger.error(f"Error recording IBKR metrics: {e}")
    
    def record_account_switch(self, from_account: str, to_account: str, success: bool):
        """Record IBKR account switch"""
        try:
            status = 'success' if success else 'error'
            self.ibkr_account_switches.labels(
                from_account=from_account,
                to_account=to_account,
                status=status
            ).inc()
            
        except Exception as e:
            logger.error(f"Error recording account switch: {e}")
    
    def record_order(self, account_id: str, order_type: str, symbol: str, status: str, latency: float = None):
        """Record IBKR order metrics"""
        try:
            self.ibkr_orders_placed.labels(
                account_id=account_id,
                order_type=order_type,
                symbol=symbol,
                status=status
            ).inc()
            
            if latency is not None:
                self.ibkr_order_latency.labels(
                    account_id=account_id,
                    order_type=order_type
                ).observe(latency)
            
        except Exception as e:
            logger.error(f"Error recording order metrics: {e}")
    
    def record_attribution(self, portfolio_id: str, method: str, level: str, effects: dict, contributors: list):
        """Record performance attribution metrics"""
        try:
            self.attribution_calculations.labels(
                portfolio_id=portfolio_id,
                method=method,
                level=level
            ).inc()
            
            # Record attribution effects
            for segment_id, segment_effects in effects.items():
                for effect_type, value in segment_effects.items():
                    self.attribution_effects.labels(
                        portfolio_id=portfolio_id,
                        segment_id=segment_id,
                        effect_type=effect_type
                    ).set(value)
            
            # Record top contributors
            for contributor in contributors:
                self.top_contributors.labels(
                    portfolio_id=portfolio_id,
                    contributor_id=contributor['name'],
                    contributor_type=contributor['type']
                ).set(contributor['contribution'])
            
        except Exception as e:
            logger.error(f"Error recording attribution metrics: {e}")
    
    def update_monitoring_metrics(self, active_portfolios: int, portfolio_id: str = None, update_duration: float = None):
        """Update monitoring metrics"""
        try:
            self.monitoring_active_portfolios.set(active_portfolios)
            
            if portfolio_id:
                self.monitoring_snapshots_generated.labels(portfolio_id=portfolio_id).inc()
                
                if update_duration is not None:
                    self.monitoring_update_duration.labels(portfolio_id=portfolio_id).observe(update_duration)
            
        except Exception as e:
            logger.error(f"Error updating monitoring metrics: {e}")
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        try:
            self.api_requests.labels(
                endpoint=endpoint,
                method=method,
                status_code=str(status_code)
            ).inc()
            
            self.api_request_duration.labels(
                endpoint=endpoint,
                method=method
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Error recording API request metrics: {e}")

# Global monitor instance
portfolio_monitor = None

def get_portfolio_monitor() -> PortfolioPrometheusMonitor:
    """Get global portfolio monitor instance"""
    global portfolio_monitor
    if portfolio_monitor is None:
        portfolio_monitor = PortfolioPrometheusMonitor()
    return portfolio_monitor

def start_portfolio_monitoring(port: int = 8006):
    """Start portfolio monitoring"""
    monitor = get_portfolio_monitor()
    monitor.start_monitoring()
    return monitor

