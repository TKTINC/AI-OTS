"""
Risk Management Service Flask Application
Main application entry point for the AI Options Trading System Risk Management Service

This module provides the main Flask application for the Risk Management Service including:
- Risk monitoring and assessment APIs
- Position limits and controls endpoints
- Drawdown protection management
- Stress testing and scenario analysis
- Risk alerting and notifications
- Compliance and audit framework
- Risk dashboard and visualization

Author: Manus AI
Version: 4.0.0
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import uuid

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Import risk management modules
from core.risk_service import RiskService, RiskLevel, RiskMetrics
from limits.position_limits import PositionLimitManager, LimitType, EnforcementAction
from drawdown.drawdown_protection import DrawdownProtectionManager, DrawdownLevel, ProtectionAction
from stress_testing.stress_tester import StressTester, ScenarioType, StressLevel
from alerting.risk_alerting import RiskAlertManager, AlertType, AlertSeverity
from compliance.compliance_manager import ComplianceManager, RegulatoryFramework, ViolationType
from dashboard.risk_dashboard import DashboardManager, DashboardType, ChartType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config.update({
    'SECRET_KEY': os.getenv('SECRET_KEY', 'risk-management-secret-key'),
    'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/ai_options_trading'),
    'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'DEBUG': os.getenv('DEBUG', 'False').lower() == 'true'
})

# Initialize database connection
def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(app.config['DATABASE_URL'])

# Initialize Redis connection
def get_redis_connection():
    """Get Redis connection"""
    return redis.from_url(app.config['REDIS_URL'])

# Initialize services
db_connection = get_db_connection()
redis_client = get_redis_connection()

risk_service = RiskService(db_connection, redis_client)
limit_manager = PositionLimitManager(db_connection, redis_client)
drawdown_manager = DrawdownProtectionManager(db_connection, redis_client)
stress_tester = StressTester(db_connection, redis_client)
alert_manager = RiskAlertManager(db_connection, redis_client)
compliance_manager = ComplianceManager(db_connection, redis_client)
dashboard_manager = DashboardManager(db_connection, redis_client)

# Start background services
risk_service.start_monitoring()
alert_manager.start_monitoring()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        with db_connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        # Check Redis connection
        redis_client.ping()
        
        return jsonify({
            'status': 'healthy',
            'service': 'risk-management',
            'version': '4.0.0',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Risk Monitoring Endpoints
@app.route('/api/v1/risk/metrics/<portfolio_id>', methods=['GET'])
def get_risk_metrics(portfolio_id: str):
    """Get current risk metrics for portfolio"""
    try:
        metrics = risk_service.get_risk_metrics(portfolio_id)
        
        if not metrics:
            return jsonify({'error': 'Portfolio not found'}), 404
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'risk_metrics': metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/risk/metrics/<portfolio_id>/history', methods=['GET'])
def get_risk_metrics_history(portfolio_id: str):
    """Get risk metrics history"""
    try:
        days = request.args.get('days', 30, type=int)
        
        history = risk_service.get_risk_metrics_history(portfolio_id, days)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'history': history,
            'days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting risk metrics history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/risk/alerts/<portfolio_id>', methods=['GET'])
def get_risk_alerts(portfolio_id: str):
    """Get risk alerts for portfolio"""
    try:
        severity = request.args.get('severity')
        limit = request.args.get('limit', 50, type=int)
        
        alerts = risk_service.get_risk_alerts(portfolio_id, severity, limit)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'alerts': [alert.to_dict() for alert in alerts],
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        return jsonify({'error': str(e)}), 500

# Position Limits Endpoints
@app.route('/api/v1/limits/<portfolio_id>', methods=['GET'])
def get_position_limits(portfolio_id: str):
    """Get position limits for portfolio"""
    try:
        limits = limit_manager.get_position_limits(portfolio_id)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'limits': [limit.to_dict() for limit in limits],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting position limits: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/limits/<portfolio_id>', methods=['POST'])
def create_position_limit(portfolio_id: str):
    """Create new position limit"""
    try:
        data = request.get_json()
        
        limit = limit_manager.create_position_limit(
            portfolio_id=portfolio_id,
            limit_type=LimitType(data['limit_type']),
            scope=data['scope'],
            limit_value=data['limit_value'],
            enforcement_action=EnforcementAction(data['enforcement_action']),
            description=data.get('description', '')
        )
        
        return jsonify({
            'limit': limit.to_dict(),
            'message': 'Position limit created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating position limit: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/limits/<portfolio_id>/violations', methods=['GET'])
def get_limit_violations(portfolio_id: str):
    """Get limit violations for portfolio"""
    try:
        violations = limit_manager.check_position_limits(portfolio_id)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'violations': [violation.to_dict() for violation in violations],
            'count': len(violations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting limit violations: {e}")
        return jsonify({'error': str(e)}), 500

# Drawdown Protection Endpoints
@app.route('/api/v1/drawdown/<portfolio_id>', methods=['GET'])
def get_drawdown_status(portfolio_id: str):
    """Get drawdown protection status"""
    try:
        status = drawdown_manager.get_drawdown_status(portfolio_id)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'drawdown_status': status.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting drawdown status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/drawdown/<portfolio_id>/thresholds', methods=['POST'])
def update_drawdown_thresholds(portfolio_id: str):
    """Update drawdown protection thresholds"""
    try:
        data = request.get_json()
        
        result = drawdown_manager.update_drawdown_thresholds(
            portfolio_id=portfolio_id,
            thresholds=data['thresholds']
        )
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'thresholds_updated': result,
            'message': 'Drawdown thresholds updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating drawdown thresholds: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/drawdown/<portfolio_id>/events', methods=['GET'])
def get_drawdown_events(portfolio_id: str):
    """Get drawdown events history"""
    try:
        days = request.args.get('days', 30, type=int)
        
        events = drawdown_manager.get_drawdown_events(portfolio_id, days)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'events': [event.to_dict() for event in events],
            'days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting drawdown events: {e}")
        return jsonify({'error': str(e)}), 500

# Stress Testing Endpoints
@app.route('/api/v1/stress-test/<portfolio_id>', methods=['POST'])
def run_stress_test(portfolio_id: str):
    """Run stress test on portfolio"""
    try:
        data = request.get_json()
        
        scenario_type = ScenarioType(data.get('scenario_type', 'market_crash'))
        stress_level = StressLevel(data.get('stress_level', 'moderate'))
        
        result = stress_tester.run_stress_test(
            portfolio_id=portfolio_id,
            scenario_type=scenario_type,
            stress_level=stress_level
        )
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'stress_test_result': result.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/stress-test/<portfolio_id>/scenarios', methods=['GET'])
def get_stress_test_scenarios(portfolio_id: str):
    """Get available stress test scenarios"""
    try:
        scenarios = stress_tester.get_available_scenarios()
        
        return jsonify({
            'scenarios': scenarios,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting stress test scenarios: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/stress-test/<portfolio_id>/history', methods=['GET'])
def get_stress_test_history(portfolio_id: str):
    """Get stress test history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        history = stress_tester.get_stress_test_history(portfolio_id, limit)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'history': [result.to_dict() for result in history],
            'count': len(history),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting stress test history: {e}")
        return jsonify({'error': str(e)}), 500

# Compliance Endpoints
@app.route('/api/v1/compliance/<portfolio_id>', methods=['GET'])
def get_compliance_status(portfolio_id: str):
    """Get compliance status for portfolio"""
    try:
        status = compliance_manager.check_portfolio_compliance(portfolio_id)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'compliance_status': status.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting compliance status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/compliance/<portfolio_id>/violations', methods=['GET'])
def get_compliance_violations(portfolio_id: str):
    """Get compliance violations"""
    try:
        days = request.args.get('days', 30, type=int)
        
        violations = compliance_manager.get_compliance_violations(portfolio_id, days)
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'violations': [violation.to_dict() for violation in violations],
            'days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting compliance violations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/compliance/<portfolio_id>/report', methods=['POST'])
def generate_compliance_report(portfolio_id: str):
    """Generate compliance report"""
    try:
        data = request.get_json()
        
        framework = RegulatoryFramework(data.get('framework', 'sec'))
        report_type = data.get('report_type', 'risk_management')
        start_date = datetime.fromisoformat(data['start_date'])
        end_date = datetime.fromisoformat(data['end_date'])
        
        report = compliance_manager.generate_regulatory_report(
            portfolio_id=portfolio_id,
            framework=framework,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'report': report.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        return jsonify({'error': str(e)}), 500

# Dashboard Endpoints
@app.route('/api/v1/dashboard', methods=['POST'])
def create_dashboard():
    """Create new risk dashboard"""
    try:
        data = request.get_json()
        
        dashboard = dashboard_manager.create_dashboard(
            dashboard_type=DashboardType(data['dashboard_type']),
            title=data['title'],
            description=data['description'],
            user_id=data['user_id'],
            portfolio_id=data.get('portfolio_id')
        )
        
        return jsonify({
            'dashboard': dashboard.to_dict(),
            'message': 'Dashboard created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/dashboard/<dashboard_id>', methods=['GET'])
def get_dashboard(dashboard_id: str):
    """Get dashboard data"""
    try:
        dashboard_data = dashboard_manager.get_dashboard_data(dashboard_id)
        
        if 'error' in dashboard_data:
            return jsonify(dashboard_data), 404
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/dashboard/<dashboard_id>/update', methods=['POST'])
def update_dashboard_data(dashboard_id: str):
    """Update dashboard data"""
    try:
        result = dashboard_manager.update_dashboard_data(dashboard_id)
        
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error updating dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/dashboard/<dashboard_id>/realtime/start', methods=['POST'])
def start_realtime_updates(dashboard_id: str):
    """Start real-time dashboard updates"""
    try:
        dashboard_manager.start_real_time_updates(dashboard_id)
        
        return jsonify({
            'dashboard_id': dashboard_id,
            'message': 'Real-time updates started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting real-time updates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/dashboard/<dashboard_id>/realtime/stop', methods=['POST'])
def stop_realtime_updates(dashboard_id: str):
    """Stop real-time dashboard updates"""
    try:
        dashboard_manager.stop_real_time_updates(dashboard_id)
        
        return jsonify({
            'dashboard_id': dashboard_id,
            'message': 'Real-time updates stopped',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error stopping real-time updates: {e}")
        return jsonify({'error': str(e)}), 500

# Alert Management Endpoints
@app.route('/api/v1/alerts/<portfolio_id>/subscribe', methods=['POST'])
def subscribe_to_alerts(portfolio_id: str):
    """Subscribe to risk alerts"""
    try:
        data = request.get_json()
        
        result = alert_manager.subscribe_to_alerts(
            portfolio_id=portfolio_id,
            user_id=data['user_id'],
            channels=data['channels'],
            preferences=data.get('preferences', {})
        )
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'subscription': result,
            'message': 'Alert subscription created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error subscribing to alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id: str):
    """Acknowledge risk alert"""
    try:
        data = request.get_json()
        
        result = alert_manager.acknowledge_alert(
            alert_id=alert_id,
            user_id=data['user_id'],
            notes=data.get('notes', '')
        )
        
        return jsonify({
            'alert_id': alert_id,
            'acknowledged': result,
            'message': 'Alert acknowledged successfully'
        })
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        return jsonify({'error': str(e)}), 500

# System Management Endpoints
@app.route('/api/v1/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        status = {
            'risk_service': risk_service.get_service_status(),
            'limit_manager': limit_manager.get_service_status(),
            'drawdown_manager': drawdown_manager.get_service_status(),
            'stress_tester': stress_tester.get_service_status(),
            'alert_manager': alert_manager.get_service_status(),
            'compliance_manager': compliance_manager.get_service_status(),
            'dashboard_manager': dashboard_manager.get_service_status()
        }
        
        return jsonify({
            'system_status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/system/metrics', methods=['GET'])
def get_system_metrics():
    """Get system performance metrics"""
    try:
        metrics = {
            'active_portfolios': risk_service.get_active_portfolio_count(),
            'total_alerts': alert_manager.get_alert_count(),
            'active_violations': limit_manager.get_violation_count(),
            'stress_tests_today': stress_tester.get_daily_test_count(),
            'compliance_score': compliance_manager.get_overall_compliance_score()
        }
        
        return jsonify({
            'system_metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

# Cleanup on shutdown
@app.teardown_appcontext
def close_connections(error):
    """Close database and Redis connections"""
    try:
        if hasattr(g, 'db_connection'):
            g.db_connection.close()
        if hasattr(g, 'redis_client'):
            g.redis_client.close()
    except Exception as e:
        logger.error(f"Error closing connections: {e}")

if __name__ == '__main__':
    # Initialize database tables
    try:
        risk_service.initialize_database()
        limit_manager.initialize_database()
        drawdown_manager.initialize_database()
        stress_tester.initialize_database()
        alert_manager.initialize_database()
        compliance_manager.initialize_database()
        dashboard_manager.initialize_database()
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    
    # Start the application
    port = int(os.getenv('PORT', 8006))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Risk Management Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

