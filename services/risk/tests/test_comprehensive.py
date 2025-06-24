"""
Comprehensive Testing Framework for Risk Management Service
Unit, integration, and performance tests for all risk management components

This module provides comprehensive testing for:
- Risk monitoring and assessment
- Position limits and controls
- Drawdown protection
- Stress testing
- Risk alerting
- Compliance management
- Dashboard functionality

Author: Manus AI
Version: 4.0.0
"""

import unittest
import time
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading
import concurrent.futures

import numpy as np
import pandas as pd
import redis
import psycopg2
from psycopg2.extras import RealDictCursor

# Import risk management modules for testing
from core.risk_service import RiskService, RiskLevel, RiskMetrics
from limits.position_limits import PositionLimitManager, LimitType, EnforcementAction
from drawdown.drawdown_protection import DrawdownProtectionManager, DrawdownLevel, ProtectionAction
from stress_testing.stress_tester import StressTester, ScenarioType, StressLevel
from alerting.risk_alerting import RiskAlertManager, AlertType, AlertSeverity
from compliance.compliance_manager import ComplianceManager, RegulatoryFramework, ViolationType
from dashboard.risk_dashboard import DashboardManager, DashboardType, ChartType

class TestRiskService(unittest.TestCase):
    """Test cases for Risk Service"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.risk_service = RiskService(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_calculate_var_95(self):
        """Test VaR 95% calculation"""
        # Mock portfolio data
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25},
                {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'volatility': 0.20}
            ],
            'total_value': 30000.0,
            'correlation_matrix': [[1.0, 0.3], [0.3, 1.0]]
        }
        
        var_95 = self.risk_service._calculate_var_95(portfolio_data)
        
        # VaR should be positive and reasonable (1-5% of portfolio)
        self.assertGreater(var_95, 0)
        self.assertLess(var_95, portfolio_data['total_value'] * 0.1)
        self.assertIsInstance(var_95, float)
    
    def test_calculate_var_99(self):
        """Test VaR 99% calculation"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25}
            ],
            'total_value': 15000.0,
            'correlation_matrix': [[1.0]]
        }
        
        var_99 = self.risk_service._calculate_var_99(portfolio_data)
        
        # VaR 99% should be higher than VaR 95%
        var_95 = self.risk_service._calculate_var_95(portfolio_data)
        self.assertGreater(var_99, var_95)
        self.assertIsInstance(var_99, float)
    
    def test_calculate_expected_shortfall(self):
        """Test Expected Shortfall calculation"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25}
            ],
            'total_value': 15000.0,
            'correlation_matrix': [[1.0]]
        }
        
        es = self.risk_service._calculate_expected_shortfall(portfolio_data)
        
        # Expected Shortfall should be higher than VaR 99%
        var_99 = self.risk_service._calculate_var_99(portfolio_data)
        self.assertGreater(es, var_99)
        self.assertIsInstance(es, float)
    
    def test_calculate_portfolio_volatility(self):
        """Test portfolio volatility calculation"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25, 'weight': 0.5},
                {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'volatility': 0.20, 'weight': 0.5}
            ],
            'correlation_matrix': [[1.0, 0.3], [0.3, 1.0]]
        }
        
        volatility = self.risk_service._calculate_portfolio_volatility(portfolio_data)
        
        # Portfolio volatility should be between individual asset volatilities due to diversification
        self.assertGreater(volatility, 0.15)  # Lower than highest individual vol due to diversification
        self.assertLess(volatility, 0.25)     # Higher than lowest individual vol
        self.assertIsInstance(volatility, float)
    
    def test_determine_risk_level(self):
        """Test risk level determination"""
        # Test LOW risk
        low_risk_metrics = RiskMetrics(
            portfolio_id=self.test_portfolio_id,
            var_95=500.0,
            var_99=750.0,
            expected_shortfall=900.0,
            max_drawdown=0.05,
            current_drawdown=0.02,
            volatility=0.12,
            beta=0.8,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            concentration_risk=0.15,
            liquidity_risk=0.1,
            correlation_risk=0.2
        )
        
        risk_level = self.risk_service._determine_risk_level(low_risk_metrics)
        self.assertEqual(risk_level, RiskLevel.LOW)
        
        # Test HIGH risk
        high_risk_metrics = RiskMetrics(
            portfolio_id=self.test_portfolio_id,
            var_95=5000.0,
            var_99=7500.0,
            expected_shortfall=9000.0,
            max_drawdown=0.25,
            current_drawdown=0.18,
            volatility=0.35,
            beta=1.8,
            sharpe_ratio=0.3,
            sortino_ratio=0.4,
            concentration_risk=0.45,
            liquidity_risk=0.4,
            correlation_risk=0.8
        )
        
        risk_level = self.risk_service._determine_risk_level(high_risk_metrics)
        self.assertEqual(risk_level, RiskLevel.HIGH)
    
    def test_risk_monitoring_performance(self):
        """Test risk monitoring performance"""
        start_time = time.time()
        
        # Simulate monitoring 10 portfolios
        for i in range(10):
            portfolio_id = f"portfolio_{i}"
            self.risk_service.calculate_risk_metrics(portfolio_id)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 5 seconds
        self.assertLess(execution_time, 5.0)
        print(f"Risk monitoring performance: {execution_time:.2f}s for 10 portfolios")

class TestPositionLimitManager(unittest.TestCase):
    """Test cases for Position Limit Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.limit_manager = PositionLimitManager(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_create_position_limit(self):
        """Test position limit creation"""
        limit = self.limit_manager.create_position_limit(
            portfolio_id=self.test_portfolio_id,
            limit_type=LimitType.POSITION_SIZE,
            scope="AAPL",
            limit_value=25000.0,
            enforcement_action=EnforcementAction.BLOCK_NEW,
            description="AAPL position size limit"
        )
        
        self.assertEqual(limit.portfolio_id, self.test_portfolio_id)
        self.assertEqual(limit.limit_type, LimitType.POSITION_SIZE)
        self.assertEqual(limit.scope, "AAPL")
        self.assertEqual(limit.limit_value, 25000.0)
        self.assertEqual(limit.enforcement_action, EnforcementAction.BLOCK_NEW)
    
    def test_check_position_size_limit(self):
        """Test position size limit checking"""
        # Mock position data
        position_data = {
            'AAPL': {'value': 30000.0, 'quantity': 200},
            'MSFT': {'value': 20000.0, 'quantity': 67}
        }
        
        # Create limit
        limit = self.limit_manager.create_position_limit(
            portfolio_id=self.test_portfolio_id,
            limit_type=LimitType.POSITION_SIZE,
            scope="AAPL",
            limit_value=25000.0,
            enforcement_action=EnforcementAction.ALERT_ONLY
        )
        
        violations = self.limit_manager._check_position_size_limits(
            self.test_portfolio_id, position_data
        )
        
        # Should detect AAPL violation
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].scope, "AAPL")
        self.assertGreater(violations[0].current_value, violations[0].limit_value)
    
    def test_check_sector_exposure_limit(self):
        """Test sector exposure limit checking"""
        # Mock portfolio data
        portfolio_data = {
            'total_value': 100000.0,
            'sector_exposures': {
                'Technology': 45000.0,  # 45% - should violate 40% limit
                'Healthcare': 25000.0,  # 25%
                'Finance': 30000.0      # 30%
            }
        }
        
        violations = self.limit_manager._check_sector_exposure_limits(
            self.test_portfolio_id, portfolio_data
        )
        
        # Should detect Technology sector violation
        tech_violations = [v for v in violations if v.scope == "Technology"]
        self.assertEqual(len(tech_violations), 1)
        self.assertGreater(tech_violations[0].current_value, 0.40)  # 40% limit
    
    def test_enforcement_action_execution(self):
        """Test enforcement action execution"""
        violation = Mock()
        violation.enforcement_action = EnforcementAction.REDUCE_POSITION
        violation.scope = "AAPL"
        violation.current_value = 30000.0
        violation.limit_value = 25000.0
        
        result = self.limit_manager._execute_enforcement_action(violation)
        
        self.assertTrue(result['action_taken'])
        self.assertEqual(result['action_type'], 'reduce_position')
        self.assertIn('target_reduction', result)

class TestDrawdownProtectionManager(unittest.TestCase):
    """Test cases for Drawdown Protection Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.drawdown_manager = DrawdownProtectionManager(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation"""
        # Mock portfolio history
        portfolio_history = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-02', 'value': 105000.0},  # Peak
            {'date': '2024-01-03', 'value': 98000.0},   # Current
            {'date': '2024-01-04', 'value': 95000.0}    # Lowest
        ]
        
        current_drawdown = self.drawdown_manager._calculate_current_drawdown(portfolio_history)
        max_drawdown = self.drawdown_manager._calculate_max_drawdown(portfolio_history)
        
        # Current drawdown: (105000 - 98000) / 105000 = 6.67%
        self.assertAlmostEqual(current_drawdown, 0.0667, places=3)
        
        # Max drawdown: (105000 - 95000) / 105000 = 9.52%
        self.assertAlmostEqual(max_drawdown, 0.0952, places=3)
    
    def test_determine_drawdown_level(self):
        """Test drawdown level determination"""
        # Test NORMAL level
        normal_drawdown = 0.03  # 3%
        level = self.drawdown_manager._determine_drawdown_level(normal_drawdown)
        self.assertEqual(level, DrawdownLevel.NORMAL)
        
        # Test SEVERE level
        severe_drawdown = 0.18  # 18%
        level = self.drawdown_manager._determine_drawdown_level(severe_drawdown)
        self.assertEqual(level, DrawdownLevel.SEVERE)
        
        # Test CRITICAL level
        critical_drawdown = 0.25  # 25%
        level = self.drawdown_manager._determine_drawdown_level(critical_drawdown)
        self.assertEqual(level, DrawdownLevel.CRITICAL)
    
    def test_protection_action_selection(self):
        """Test protection action selection"""
        # Test SEVERE drawdown actions
        actions = self.drawdown_manager._get_protection_actions(DrawdownLevel.SEVERE)
        
        expected_actions = [
            ProtectionAction.REDUCE_RISK,
            ProtectionAction.HALT_NEW_TRADES,
            ProtectionAction.REDUCE_POSITIONS
        ]
        
        for action in expected_actions:
            self.assertIn(action, actions)
    
    def test_drawdown_duration_tracking(self):
        """Test drawdown duration tracking"""
        # Mock drawdown events
        drawdown_events = [
            {'date': '2024-01-01', 'drawdown': 0.08, 'level': 'WARNING'},
            {'date': '2024-01-02', 'drawdown': 0.12, 'level': 'MODERATE'},
            {'date': '2024-01-03', 'drawdown': 0.15, 'level': 'MODERATE'},
            {'date': '2024-01-04', 'drawdown': 0.10, 'level': 'MODERATE'},
            {'date': '2024-01-05', 'drawdown': 0.04, 'level': 'NORMAL'}
        ]
        
        duration = self.drawdown_manager._calculate_drawdown_duration(drawdown_events)
        
        # Should be 4 days (from WARNING to NORMAL)
        self.assertEqual(duration, 4)

class TestStressTester(unittest.TestCase):
    """Test cases for Stress Tester"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.stress_tester = StressTester(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_market_crash_scenario(self):
        """Test market crash scenario"""
        # Mock portfolio data
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'beta': 1.2},
                {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'beta': 1.1}
            ],
            'total_value': 30000.0
        }
        
        result = self.stress_tester._run_market_crash_scenario(
            portfolio_data, StressLevel.MODERATE
        )
        
        # Should show portfolio loss
        self.assertLess(result['portfolio_value_after'], portfolio_data['total_value'])
        self.assertGreater(result['portfolio_loss_pct'], 0)
        self.assertLess(result['portfolio_loss_pct'], 1.0)  # Loss should be < 100%
    
    def test_volatility_spike_scenario(self):
        """Test volatility spike scenario"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25}
            ],
            'total_value': 15000.0,
            'options_positions': [
                {'symbol': 'AAPL', 'type': 'call', 'strike': 155.0, 'expiry': '2024-02-01', 'quantity': 10}
            ]
        }
        
        result = self.stress_tester._run_volatility_spike_scenario(
            portfolio_data, StressLevel.SEVERE
        )
        
        # Volatility spike should affect options more than stocks
        self.assertIn('options_impact', result)
        self.assertIn('stocks_impact', result)
        self.assertGreater(abs(result['options_impact']), abs(result['stocks_impact']))
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25, 'expected_return': 0.08}
            ],
            'total_value': 15000.0,
            'correlation_matrix': [[1.0]]
        }
        
        result = self.stress_tester._run_monte_carlo_simulation(portfolio_data, 1000)
        
        # Should have statistical results
        self.assertIn('simulations', result)
        self.assertIn('percentiles', result)
        self.assertIn('var_95', result)
        self.assertIn('var_99', result)
        self.assertEqual(len(result['simulations']), 1000)
    
    def test_stress_test_performance(self):
        """Test stress testing performance"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'volatility': 0.25},
                {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'volatility': 0.20}
            ],
            'total_value': 30000.0
        }
        
        start_time = time.time()
        
        # Run multiple stress tests
        for scenario in [ScenarioType.MARKET_CRASH, ScenarioType.VOLATILITY_SPIKE, ScenarioType.INTEREST_RATE]:
            result = self.stress_tester.run_stress_test(
                self.test_portfolio_id, scenario, StressLevel.MODERATE
            )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 10 seconds
        self.assertLess(execution_time, 10.0)
        print(f"Stress testing performance: {execution_time:.2f}s for 3 scenarios")

class TestRiskAlertManager(unittest.TestCase):
    """Test cases for Risk Alert Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.alert_manager = RiskAlertManager(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_create_risk_alert(self):
        """Test risk alert creation"""
        alert = self.alert_manager.create_risk_alert(
            portfolio_id=self.test_portfolio_id,
            alert_type=AlertType.VAR_BREACH,
            severity=AlertSeverity.ERROR,
            title="VaR Limit Exceeded",
            description="Portfolio VaR has exceeded the 95% limit",
            data={'current_var': 5500.0, 'limit': 5000.0}
        )
        
        self.assertEqual(alert.portfolio_id, self.test_portfolio_id)
        self.assertEqual(alert.alert_type, AlertType.VAR_BREACH)
        self.assertEqual(alert.severity, AlertSeverity.ERROR)
        self.assertIn('current_var', alert.data)
    
    def test_alert_filtering(self):
        """Test alert filtering and deduplication"""
        # Create multiple similar alerts
        for i in range(5):
            self.alert_manager.create_risk_alert(
                portfolio_id=self.test_portfolio_id,
                alert_type=AlertType.VAR_BREACH,
                severity=AlertSeverity.WARNING,
                title="VaR Warning",
                description=f"VaR warning {i}",
                data={'var': 4500.0 + i * 100}
            )
        
        # Should filter duplicate alerts
        recent_alerts = self.alert_manager.get_recent_alerts(
            self.test_portfolio_id, hours=1
        )
        
        # Should have fewer than 5 alerts due to deduplication
        self.assertLessEqual(len(recent_alerts), 3)
    
    def test_notification_delivery(self):
        """Test notification delivery"""
        alert = self.alert_manager.create_risk_alert(
            portfolio_id=self.test_portfolio_id,
            alert_type=AlertType.DRAWDOWN,
            severity=AlertSeverity.CRITICAL,
            title="Critical Drawdown",
            description="Portfolio drawdown exceeded 20%",
            data={'drawdown': 0.22}
        )
        
        # Mock notification preferences
        preferences = {
            'user_id': 'user_123',
            'channels': ['email', 'sms'],
            'min_severity': 'ERROR'
        }
        
        result = self.alert_manager.send_notifications(alert, preferences)
        
        self.assertTrue(result['notifications_sent'])
        self.assertGreater(result['delivery_count'], 0)
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment"""
        alert = self.alert_manager.create_risk_alert(
            portfolio_id=self.test_portfolio_id,
            alert_type=AlertType.POSITION_LIMIT,
            severity=AlertSeverity.ERROR,
            title="Position Limit Breach",
            description="AAPL position exceeded limit"
        )
        
        # Acknowledge alert
        result = self.alert_manager.acknowledge_alert(
            alert.alert_id, 'user_123', 'Reviewed and approved'
        )
        
        self.assertTrue(result)
        
        # Check acknowledgment status
        updated_alert = self.alert_manager.get_alert(alert.alert_id)
        self.assertTrue(updated_alert.acknowledged)
        self.assertEqual(updated_alert.acknowledged_by, 'user_123')

class TestComplianceManager(unittest.TestCase):
    """Test cases for Compliance Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.compliance_manager = ComplianceManager(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_sec_compliance_check(self):
        """Test SEC compliance checking"""
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'value': 25000.0},  # 25% - within 20% limit
                {'symbol': 'MSFT', 'value': 15000.0}   # 15%
            ],
            'total_value': 100000.0,
            'leverage': 2.5  # Within 3:1 limit
        }
        
        violations = self.compliance_manager._check_sec_compliance(
            self.test_portfolio_id, portfolio_data
        )
        
        # Should detect position concentration violation
        concentration_violations = [v for v in violations if v.violation_type == ViolationType.LIMIT_BREACH]
        self.assertEqual(len(concentration_violations), 1)
    
    def test_basel_iii_compliance_check(self):
        """Test Basel III compliance checking"""
        portfolio_data = {
            'total_value': 100000.0,
            'var_95': 3000.0,  # 3% - exceeds 2.5% limit
            'leverage': 2.8,   # Within 3:1 limit
            'stress_test_date': datetime.now() - timedelta(days=45)  # Outdated
        }
        
        violations = self.compliance_manager._check_basel_iii_compliance(
            self.test_portfolio_id, portfolio_data
        )
        
        # Should detect VaR and stress test violations
        self.assertGreaterEqual(len(violations), 2)
        
        var_violations = [v for v in violations if 'VaR' in v.description]
        stress_violations = [v for v in violations if 'stress test' in v.description]
        
        self.assertEqual(len(var_violations), 1)
        self.assertEqual(len(stress_violations), 1)
    
    def test_audit_trail_integrity(self):
        """Test audit trail integrity"""
        # Create audit event
        event_id = self.compliance_manager.audit_manager.log_event(
            event_type='TRADE_EXECUTION',
            action='buy_option',
            description='Purchased AAPL call option',
            details={'symbol': 'AAPL', 'quantity': 10, 'price': 5.50}
        )
        
        # Verify integrity
        integrity_check = self.compliance_manager.audit_manager.verify_audit_integrity(event_id)
        
        self.assertTrue(integrity_check['valid'])
        self.assertIn('hash_verified', integrity_check)
        self.assertTrue(integrity_check['hash_verified'])
    
    def test_regulatory_report_generation(self):
        """Test regulatory report generation"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        report = self.compliance_manager.generate_regulatory_report(
            portfolio_id=self.test_portfolio_id,
            framework=RegulatoryFramework.SEC,
            report_type='risk_management',
            start_date=start_date,
            end_date=end_date
        )
        
        self.assertEqual(report.portfolio_id, self.test_portfolio_id)
        self.assertEqual(report.framework, RegulatoryFramework.SEC)
        self.assertIsNotNone(report.report_data)
        self.assertIn('compliance_summary', report.report_data)

class TestDashboardManager(unittest.TestCase):
    """Test cases for Dashboard Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.dashboard_manager = DashboardManager(self.mock_db, self.mock_redis)
        self.test_portfolio_id = "test_portfolio_123"
    
    def test_create_overview_dashboard(self):
        """Test overview dashboard creation"""
        dashboard = self.dashboard_manager.create_dashboard(
            dashboard_type=DashboardType.OVERVIEW,
            title="Portfolio Risk Overview",
            description="Real-time portfolio risk monitoring",
            user_id="user_123",
            portfolio_id=self.test_portfolio_id
        )
        
        self.assertEqual(dashboard.dashboard_type, DashboardType.OVERVIEW)
        self.assertEqual(dashboard.title, "Portfolio Risk Overview")
        self.assertGreater(len(dashboard.widgets), 0)
        
        # Check for expected widgets
        widget_types = [w.widget_type for w in dashboard.widgets]
        self.assertIn("risk_overview", widget_types)
        self.assertIn("var_gauge", widget_types)
        self.assertIn("position_heatmap", widget_types)
    
    def test_dashboard_data_update(self):
        """Test dashboard data updates"""
        dashboard = self.dashboard_manager.create_dashboard(
            dashboard_type=DashboardType.OVERVIEW,
            title="Test Dashboard",
            description="Test dashboard",
            user_id="user_123",
            portfolio_id=self.test_portfolio_id
        )
        
        # Update dashboard data
        result = self.dashboard_manager.update_dashboard_data(dashboard.dashboard_id)
        
        self.assertIn('widgets_updated', result)
        self.assertGreater(result['widgets_updated'], 0)
        self.assertIn('last_updated', result)
    
    def test_real_time_updates(self):
        """Test real-time dashboard updates"""
        dashboard = self.dashboard_manager.create_dashboard(
            dashboard_type=DashboardType.REAL_TIME,
            title="Real-time Dashboard",
            description="Real-time monitoring",
            user_id="user_123",
            portfolio_id=self.test_portfolio_id
        )
        
        # Start real-time updates
        self.dashboard_manager.start_real_time_updates(dashboard.dashboard_id)
        
        # Check if dashboard is in active list
        self.assertIn(dashboard.dashboard_id, self.dashboard_manager.active_dashboards)
        
        # Stop real-time updates
        self.dashboard_manager.stop_real_time_updates(dashboard.dashboard_id)
        
        # Check if dashboard is removed from active list
        self.assertNotIn(dashboard.dashboard_id, self.dashboard_manager.active_dashboards)

class TestIntegration(unittest.TestCase):
    """Integration tests for risk management system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        
        # Initialize all services
        self.risk_service = RiskService(self.mock_db, self.mock_redis)
        self.limit_manager = PositionLimitManager(self.mock_db, self.mock_redis)
        self.drawdown_manager = DrawdownProtectionManager(self.mock_db, self.mock_redis)
        self.stress_tester = StressTester(self.mock_db, self.mock_redis)
        self.alert_manager = RiskAlertManager(self.mock_db, self.mock_redis)
        self.compliance_manager = ComplianceManager(self.mock_db, self.mock_redis)
        self.dashboard_manager = DashboardManager(self.mock_db, self.mock_redis)
        
        self.test_portfolio_id = "integration_test_portfolio"
    
    def test_end_to_end_risk_workflow(self):
        """Test complete risk management workflow"""
        # 1. Calculate risk metrics
        risk_metrics = self.risk_service.calculate_risk_metrics(self.test_portfolio_id)
        self.assertIsNotNone(risk_metrics)
        
        # 2. Check position limits
        violations = self.limit_manager.check_position_limits(self.test_portfolio_id)
        
        # 3. Check drawdown protection
        drawdown_status = self.drawdown_manager.check_drawdown_protection(self.test_portfolio_id)
        
        # 4. Run stress test
        stress_result = self.stress_tester.run_stress_test(
            self.test_portfolio_id, ScenarioType.MARKET_CRASH, StressLevel.MODERATE
        )
        
        # 5. Generate alerts if needed
        if violations or drawdown_status.level != DrawdownLevel.NORMAL:
            alert = self.alert_manager.create_risk_alert(
                portfolio_id=self.test_portfolio_id,
                alert_type=AlertType.PORTFOLIO_RISK,
                severity=AlertSeverity.WARNING,
                title="Risk Workflow Alert",
                description="Automated risk workflow detected issues"
            )
            self.assertIsNotNone(alert)
        
        # 6. Check compliance
        compliance_status = self.compliance_manager.check_portfolio_compliance(self.test_portfolio_id)
        
        # 7. Update dashboard
        dashboard = self.dashboard_manager.create_dashboard(
            dashboard_type=DashboardType.OVERVIEW,
            title="Integration Test Dashboard",
            description="End-to-end test dashboard",
            user_id="integration_test_user",
            portfolio_id=self.test_portfolio_id
        )
        
        update_result = self.dashboard_manager.update_dashboard_data(dashboard.dashboard_id)
        self.assertIn('widgets_updated', update_result)
    
    def test_concurrent_risk_monitoring(self):
        """Test concurrent risk monitoring for multiple portfolios"""
        portfolio_ids = [f"concurrent_test_portfolio_{i}" for i in range(5)]
        
        def monitor_portfolio(portfolio_id):
            """Monitor single portfolio"""
            try:
                # Calculate risk metrics
                risk_metrics = self.risk_service.calculate_risk_metrics(portfolio_id)
                
                # Check limits
                violations = self.limit_manager.check_position_limits(portfolio_id)
                
                # Check drawdown
                drawdown_status = self.drawdown_manager.check_drawdown_protection(portfolio_id)
                
                return {
                    'portfolio_id': portfolio_id,
                    'risk_metrics': risk_metrics,
                    'violations': len(violations),
                    'drawdown_level': drawdown_status.level if drawdown_status else None
                }
            except Exception as e:
                return {'portfolio_id': portfolio_id, 'error': str(e)}
        
        # Run concurrent monitoring
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(monitor_portfolio, pid) for pid in portfolio_ids]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify results
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('portfolio_id', result)
            if 'error' not in result:
                self.assertIsNotNone(result['risk_metrics'])
        
        # Should complete within 10 seconds
        self.assertLess(execution_time, 10.0)
        print(f"Concurrent monitoring performance: {execution_time:.2f}s for 5 portfolios")

class TestPerformance(unittest.TestCase):
    """Performance tests for risk management system"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.mock_db = Mock()
        self.mock_redis = Mock()
        self.risk_service = RiskService(self.mock_db, self.mock_redis)
    
    def test_risk_calculation_performance(self):
        """Test risk calculation performance"""
        # Large portfolio with 100 positions
        large_portfolio_data = {
            'positions': [
                {
                    'symbol': f'STOCK_{i}',
                    'quantity': 100 + i,
                    'price': 50.0 + i * 2,
                    'volatility': 0.15 + (i % 10) * 0.01,
                    'beta': 0.8 + (i % 5) * 0.1
                }
                for i in range(100)
            ],
            'total_value': 500000.0
        }
        
        # Generate correlation matrix
        correlation_matrix = np.random.rand(100, 100)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        large_portfolio_data['correlation_matrix'] = correlation_matrix.tolist()
        
        start_time = time.time()
        
        # Calculate all risk metrics
        var_95 = self.risk_service._calculate_var_95(large_portfolio_data)
        var_99 = self.risk_service._calculate_var_99(large_portfolio_data)
        expected_shortfall = self.risk_service._calculate_expected_shortfall(large_portfolio_data)
        volatility = self.risk_service._calculate_portfolio_volatility(large_portfolio_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 2 seconds for 100 positions
        self.assertLess(execution_time, 2.0)
        
        # Verify calculations completed
        self.assertGreater(var_95, 0)
        self.assertGreater(var_99, var_95)
        self.assertGreater(expected_shortfall, var_99)
        self.assertGreater(volatility, 0)
        
        print(f"Risk calculation performance: {execution_time:.3f}s for 100 positions")
    
    def test_stress_testing_performance(self):
        """Test stress testing performance"""
        portfolio_data = {
            'positions': [
                {'symbol': f'STOCK_{i}', 'quantity': 100, 'price': 100.0, 'volatility': 0.25}
                for i in range(20)
            ],
            'total_value': 200000.0
        }
        
        start_time = time.time()
        
        # Run multiple stress tests
        scenarios = [ScenarioType.MARKET_CRASH, ScenarioType.VOLATILITY_SPIKE, ScenarioType.INTEREST_RATE]
        stress_levels = [StressLevel.MILD, StressLevel.MODERATE, StressLevel.SEVERE]
        
        results = []
        for scenario in scenarios:
            for level in stress_levels:
                result = self.stress_tester._run_market_crash_scenario(portfolio_data, level)
                results.append(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 5 seconds for 9 stress tests
        self.assertLess(execution_time, 5.0)
        self.assertEqual(len(results), 9)
        
        print(f"Stress testing performance: {execution_time:.3f}s for 9 scenarios")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestRiskService,
        TestPositionLimitManager,
        TestDrawdownProtectionManager,
        TestStressTester,
        TestRiskAlertManager,
        TestComplianceManager,
        TestDashboardManager,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

