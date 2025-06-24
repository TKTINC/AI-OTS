"""
Portfolio Management Service - Main Flask Application
Comprehensive portfolio management with IBKR integration, position sizing, optimization, and attribution
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import threading
import time

# Import portfolio management components
from core.portfolio_service import PortfolioService
from ibkr.ibkr_client import IBKRClient
from ibkr.account_manager import AccountManager
from position_sizing.position_sizer import PositionSizer
from optimization.portfolio_optimizer import PortfolioOptimizer
from risk_budgeting.risk_budget_manager import RiskBudgetManager
from monitoring.portfolio_monitor import PortfolioMonitor
from attribution.performance_attributor import PerformanceAttributor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config.update({
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key'),
    'IBKR_HOST': os.environ.get('IBKR_HOST', '127.0.0.1'),
    'IBKR_PORT': int(os.environ.get('IBKR_PORT', '7497')),
    'IBKR_CLIENT_ID': int(os.environ.get('IBKR_CLIENT_ID', '1')),
    'DATABASE_URL': os.environ.get('DATABASE_URL', 'sqlite:///portfolio.db'),
    'REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
    'ENVIRONMENT': os.environ.get('ENVIRONMENT', 'development')
})

# Global service instances
portfolio_service = None
account_manager = None
position_sizer = None
portfolio_optimizer = None
risk_budget_manager = None
portfolio_monitor = None
performance_attributor = None

def create_app():
    """Create and configure the Flask application"""
    global portfolio_service, account_manager, position_sizer
    global portfolio_optimizer, risk_budget_manager, portfolio_monitor, performance_attributor
    
    try:
        # Initialize services
        logger.info("Initializing Portfolio Management Service...")
        
        # Account Manager for IBKR integration
        account_manager = AccountManager()
        
        # Core portfolio service
        portfolio_service = PortfolioService(
            database_url=app.config['DATABASE_URL'],
            redis_url=app.config['REDIS_URL']
        )
        
        # Position sizing algorithms
        position_sizer = PositionSizer()
        
        # Portfolio optimization engine
        portfolio_optimizer = PortfolioOptimizer()
        
        # Risk budgeting system
        risk_budget_manager = RiskBudgetManager()
        
        # Real-time portfolio monitoring
        portfolio_monitor = PortfolioMonitor()
        
        # Performance attribution analysis
        performance_attributor = PerformanceAttributor()
        
        logger.info("Portfolio Management Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

# Initialize the application
create_app()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'service': 'portfolio-management',
            'version': '1.0.0',
            'components': {
                'portfolio_service': 'healthy' if portfolio_service else 'unhealthy',
                'account_manager': 'healthy' if account_manager else 'unhealthy',
                'position_sizer': 'healthy' if position_sizer else 'unhealthy',
                'portfolio_optimizer': 'healthy' if portfolio_optimizer else 'unhealthy',
                'risk_budget_manager': 'healthy' if risk_budget_manager else 'unhealthy',
                'portfolio_monitor': 'healthy' if portfolio_monitor else 'unhealthy',
                'performance_attributor': 'healthy' if performance_attributor else 'unhealthy'
            }
        }
        
        # Check if any component is unhealthy
        unhealthy_components = [k for k, v in health_status['components'].items() if v == 'unhealthy']
        if unhealthy_components:
            health_status['status'] = 'degraded'
            health_status['unhealthy_components'] = unhealthy_components
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 503

# IBKR Account Management Endpoints
@app.route('/api/v1/accounts', methods=['GET'])
def get_accounts():
    """Get available IBKR accounts"""
    try:
        accounts = account_manager.get_available_accounts()
        return jsonify({
            'accounts': accounts,
            'current_account': account_manager.get_current_account_id(),
            'connection_status': account_manager.get_connection_status()
        })
    except Exception as e:
        logger.error(f"Error getting accounts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/accounts/<account_id>/switch', methods=['POST'])
def switch_account(account_id):
    """Switch to different IBKR account (paper/live)"""
    try:
        success = account_manager.switch_account(account_id)
        if success:
            # Connect to the new account
            connection_success = account_manager.connect_current_account()
            return jsonify({
                'success': True,
                'account_id': account_id,
                'connected': connection_success,
                'message': f'Switched to account {account_id}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to switch to account {account_id}'
            }), 400
    except Exception as e:
        logger.error(f"Error switching account: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/accounts/status', methods=['GET'])
def get_account_status():
    """Get current account status and connection info"""
    try:
        status = account_manager.get_account_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting account status: {e}")
        return jsonify({'error': str(e)}), 500

# Portfolio Management Endpoints
@app.route('/api/v1/portfolios', methods=['GET'])
def get_portfolios():
    """Get all portfolios"""
    try:
        portfolios = portfolio_service.get_all_portfolios()
        return jsonify({'portfolios': portfolios})
    except Exception as e:
        logger.error(f"Error getting portfolios: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/portfolios', methods=['POST'])
def create_portfolio():
    """Create new portfolio"""
    try:
        data = request.get_json()
        required_fields = ['account_id', 'portfolio_name']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        portfolio = portfolio_service.create_portfolio(
            account_id=data['account_id'],
            portfolio_name=data['portfolio_name'],
            initial_cash=data.get('initial_cash', 100000.0),
            risk_tolerance=data.get('risk_tolerance', 'moderate'),
            investment_objective=data.get('investment_objective', 'growth')
        )
        
        return jsonify({'portfolio': portfolio}), 201
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/portfolios/<portfolio_id>', methods=['GET'])
def get_portfolio(portfolio_id):
    """Get specific portfolio"""
    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
        if portfolio:
            return jsonify({'portfolio': portfolio})
        else:
            return jsonify({'error': 'Portfolio not found'}), 404
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/portfolios/<portfolio_id>/positions', methods=['GET'])
def get_portfolio_positions(portfolio_id):
    """Get portfolio positions"""
    try:
        positions = portfolio_service.get_portfolio_positions(portfolio_id)
        return jsonify({'positions': positions})
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/portfolios/<portfolio_id>/positions', methods=['POST'])
def add_position(portfolio_id):
    """Add position to portfolio"""
    try:
        data = request.get_json()
        required_fields = ['symbol', 'quantity', 'price']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        position = portfolio_service.add_position(
            portfolio_id=portfolio_id,
            symbol=data['symbol'],
            quantity=data['quantity'],
            price=data['price'],
            strategy_id=data.get('strategy_id'),
            signal_id=data.get('signal_id')
        )
        
        return jsonify({'position': position}), 201
    except Exception as e:
        logger.error(f"Error adding position: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/portfolios/<portfolio_id>/positions/<position_id>', methods=['PUT'])
def update_position(portfolio_id, position_id):
    """Update portfolio position"""
    try:
        data = request.get_json()
        
        position = portfolio_service.update_position(
            portfolio_id=portfolio_id,
            position_id=position_id,
            quantity=data.get('quantity'),
            price=data.get('price'),
            action=data.get('action', 'update')
        )
        
        return jsonify({'position': position})
    except Exception as e:
        logger.error(f"Error updating position: {e}")
        return jsonify({'error': str(e)}), 500

# Position Sizing Endpoints
@app.route('/api/v1/position-sizing/calculate', methods=['POST'])
def calculate_position_size():
    """Calculate optimal position size"""
    try:
        data = request.get_json()
        required_fields = ['signal_data', 'portfolio_data']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        sizing_result = position_sizer.calculate_position_size(
            signal_data=data['signal_data'],
            portfolio_data=data['portfolio_data'],
            method=data.get('method', 'confidence_weighted'),
            config=data.get('config', {})
        )
        
        return jsonify({'sizing_result': sizing_result})
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/position-sizing/methods', methods=['GET'])
def get_sizing_methods():
    """Get available position sizing methods"""
    try:
        methods = position_sizer.get_available_methods()
        return jsonify({'methods': methods})
    except Exception as e:
        logger.error(f"Error getting sizing methods: {e}")
        return jsonify({'error': str(e)}), 500

# Portfolio Optimization Endpoints
@app.route('/api/v1/optimization/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio allocation"""
    try:
        data = request.get_json()
        required_fields = ['assets']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        optimization_result = portfolio_optimizer.optimize_portfolio(
            assets=data['assets'],
            objective=data.get('objective', 'max_sharpe'),
            constraints=data.get('constraints', {}),
            config=data.get('config', {})
        )
        
        return jsonify({'optimization_result': optimization_result})
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/optimization/rebalance', methods=['POST'])
def rebalance_portfolio():
    """Generate rebalancing plan"""
    try:
        data = request.get_json()
        required_fields = ['current_assets', 'target_weights']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        rebalancing_plan = portfolio_optimizer.rebalance_portfolio(
            current_assets=data['current_assets'],
            target_weights=data['target_weights'],
            constraints=data.get('constraints', {})
        )
        
        return jsonify({'rebalancing_plan': rebalancing_plan})
    except Exception as e:
        logger.error(f"Error generating rebalancing plan: {e}")
        return jsonify({'error': str(e)}), 500

# Risk Budgeting Endpoints
@app.route('/api/v1/risk-budgets', methods=['GET'])
def get_risk_budgets():
    """Get all risk budgets"""
    try:
        budgets = risk_budget_manager.get_all_risk_budgets()
        return jsonify({'risk_budgets': budgets})
    except Exception as e:
        logger.error(f"Error getting risk budgets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/risk-budgets', methods=['POST'])
def create_risk_budget():
    """Create new risk budget"""
    try:
        data = request.get_json()
        required_fields = ['budget_id', 'total_budget', 'entities']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        budget = risk_budget_manager.create_risk_budget(
            budget_id=data['budget_id'],
            total_budget=data['total_budget'],
            risk_metric=data.get('risk_metric', 'var_95'),
            allocation_type=data.get('allocation_type', 'equal_risk'),
            entities=data['entities']
        )
        
        return jsonify({'risk_budget': budget}), 201
    except Exception as e:
        logger.error(f"Error creating risk budget: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/risk-budgets/<budget_id>/allocations', methods=['GET'])
def get_risk_allocations(budget_id):
    """Get current risk allocations"""
    try:
        allocations = risk_budget_manager.get_current_allocations(budget_id)
        return jsonify({'allocations': allocations})
    except Exception as e:
        logger.error(f"Error getting risk allocations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/risk-budgets/<budget_id>/update', methods=['POST'])
def update_risk_allocations(budget_id):
    """Update risk allocations"""
    try:
        data = request.get_json()
        
        report = risk_budget_manager.update_risk_allocations(
            budget_id=budget_id,
            portfolio_data=data.get('portfolio_data', {}),
            market_regime=data.get('market_regime', 'normal')
        )
        
        return jsonify({'allocation_report': report})
    except Exception as e:
        logger.error(f"Error updating risk allocations: {e}")
        return jsonify({'error': str(e)}), 500

# Portfolio Monitoring Endpoints
@app.route('/api/v1/monitoring/<portfolio_id>/start', methods=['POST'])
def start_monitoring(portfolio_id):
    """Start real-time portfolio monitoring"""
    try:
        ibkr_client = account_manager.get_current_client()
        success = portfolio_monitor.start_monitoring(portfolio_id, ibkr_client)
        
        return jsonify({
            'success': success,
            'message': f'Monitoring started for portfolio {portfolio_id}' if success else 'Failed to start monitoring'
        })
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/monitoring/<portfolio_id>/stop', methods=['POST'])
def stop_monitoring(portfolio_id):
    """Stop real-time portfolio monitoring"""
    try:
        success = portfolio_monitor.stop_monitoring(portfolio_id)
        
        return jsonify({
            'success': success,
            'message': f'Monitoring stopped for portfolio {portfolio_id}' if success else 'Failed to stop monitoring'
        })
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/monitoring/<portfolio_id>/snapshot', methods=['GET'])
def get_portfolio_snapshot(portfolio_id):
    """Get current portfolio snapshot"""
    try:
        snapshot = portfolio_monitor.get_current_snapshot(portfolio_id)
        if snapshot:
            # Convert snapshot to dictionary
            snapshot_dict = {
                'portfolio_id': snapshot.portfolio_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'total_value': snapshot.total_value,
                'total_pnl': snapshot.total_pnl,
                'total_return_percent': snapshot.total_return_percent,
                'daily_pnl': snapshot.daily_pnl,
                'daily_return_percent': snapshot.daily_return_percent,
                'cash_balance': snapshot.cash_balance,
                'buying_power': snapshot.buying_power,
                'position_count': snapshot.position_count,
                'sharpe_ratio': snapshot.sharpe_ratio,
                'max_drawdown': snapshot.max_drawdown,
                'volatility': snapshot.volatility,
                'var_95': snapshot.var_95,
                'sector_exposure': snapshot.sector_exposure,
                'strategy_exposure': snapshot.strategy_exposure
            }
            return jsonify({'snapshot': snapshot_dict})
        else:
            return jsonify({'error': 'No snapshot available'}), 404
    except Exception as e:
        logger.error(f"Error getting portfolio snapshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/monitoring/<portfolio_id>/pnl', methods=['GET'])
def get_real_time_pnl(portfolio_id):
    """Get real-time P&L summary"""
    try:
        pnl_summary = portfolio_monitor.get_real_time_pnl(portfolio_id)
        return jsonify(pnl_summary)
    except Exception as e:
        logger.error(f"Error getting real-time P&L: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/monitoring/<portfolio_id>/performance', methods=['GET'])
def get_performance_analytics(portfolio_id):
    """Get performance analytics"""
    try:
        period = request.args.get('period', '1M')
        analytics = portfolio_monitor.get_performance_analytics(portfolio_id, period)
        
        if analytics:
            # Convert analytics to dictionary
            analytics_dict = {
                'portfolio_id': analytics.portfolio_id,
                'analysis_period': analytics.analysis_period,
                'start_date': analytics.start_date.isoformat(),
                'end_date': analytics.end_date.isoformat(),
                'total_return': analytics.total_return,
                'annualized_return': analytics.annualized_return,
                'volatility': analytics.volatility,
                'sharpe_ratio': analytics.sharpe_ratio,
                'sortino_ratio': analytics.sortino_ratio,
                'max_drawdown': analytics.max_drawdown,
                'var_95': analytics.var_95,
                'calmar_ratio': analytics.calmar_ratio,
                'return_distribution': analytics.return_distribution
            }
            return jsonify({'analytics': analytics_dict})
        else:
            return jsonify({'error': 'No analytics available'}), 404
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        return jsonify({'error': str(e)}), 500

# Performance Attribution Endpoints
@app.route('/api/v1/attribution/<portfolio_id>/calculate', methods=['POST'])
def calculate_attribution(portfolio_id):
    """Calculate performance attribution"""
    try:
        data = request.get_json()
        
        portfolio_data = data.get('portfolio_data', {})
        benchmark_data = data.get('benchmark_data')
        period = data.get('period', '1M')
        method = data.get('method', 'brinson')
        
        # Convert method string to enum
        from attribution.performance_attributor import AttributionMethod
        method_enum = getattr(AttributionMethod, method.upper(), AttributionMethod.BRINSON)
        
        report = performance_attributor.calculate_attribution(
            portfolio_id=portfolio_id,
            portfolio_data=portfolio_data,
            benchmark_data=benchmark_data,
            period=period,
            method=method_enum
        )
        
        # Convert report to dictionary
        report_dict = {
            'portfolio_id': report.portfolio_id,
            'report_date': report.report_date.isoformat(),
            'analysis_period': report.analysis_period,
            'portfolio_return': report.portfolio_return,
            'benchmark_return': report.benchmark_return,
            'active_return': report.active_return,
            'tracking_error': report.tracking_error,
            'information_ratio': report.information_ratio,
            'total_allocation_effect': report.total_allocation_effect,
            'total_selection_effect': report.total_selection_effect,
            'total_interaction_effect': report.total_interaction_effect,
            'top_contributors': report.top_contributors,
            'top_detractors': report.top_detractors,
            'strategy_attribution': report.strategy_attribution,
            'sector_attribution': report.sector_attribution
        }
        
        return jsonify({'attribution_report': report_dict})
    except Exception as e:
        logger.error(f"Error calculating attribution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/attribution/<portfolio_id>/summary', methods=['GET'])
def get_attribution_summary(portfolio_id):
    """Get attribution summary"""
    try:
        period = request.args.get('period', '1M')
        summary = performance_attributor.get_attribution_summary(portfolio_id, period)
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting attribution summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/attribution/<portfolio_id>/detailed', methods=['GET'])
def get_detailed_attribution(portfolio_id):
    """Get detailed attribution by level"""
    try:
        level = request.args.get('level', 'strategy')
        
        # Convert level string to enum
        from attribution.performance_attributor import AttributionLevel
        level_enum = getattr(AttributionLevel, level.upper(), AttributionLevel.STRATEGY)
        
        detailed_attribution = performance_attributor.get_detailed_attribution(portfolio_id, level_enum)
        return jsonify({'detailed_attribution': detailed_attribution})
    except Exception as e:
        logger.error(f"Error getting detailed attribution: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# Application startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8005))
    debug = os.environ.get('ENVIRONMENT', 'development') == 'development'
    
    logger.info(f"Starting Portfolio Management Service on port {port}")
    logger.info(f"Environment: {app.config['ENVIRONMENT']}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )

