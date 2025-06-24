"""
Comprehensive Testing Framework for Portfolio Management Service
Unit tests, integration tests, and performance tests
"""

import unittest
import json
import time
import threading
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import components to test
from core.portfolio_service import PortfolioService
from ibkr.ibkr_client import IBKRClient
from ibkr.account_manager import AccountManager
from position_sizing.position_sizer import PositionSizer, PositionSizingMethod
from optimization.portfolio_optimizer import PortfolioOptimizer, OptimizationObjective
from risk_budgeting.risk_budget_manager import RiskBudgetManager, RiskBudgetType, RiskMetric
from monitoring.portfolio_monitor import PortfolioMonitor
from attribution.performance_attributor import PerformanceAttributor, AttributionMethod, AttributionLevel

class TestPortfolioService(unittest.TestCase):
    """Test cases for Portfolio Service"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.portfolio_service = PortfolioService(
            database_url='sqlite:///:memory:',
            redis_url='redis://localhost:6379'
        )
        
        # Mock portfolio data
        self.mock_portfolio_data = {
            'account_id': 'DU123456',
            'portfolio_name': 'Test Portfolio',
            'initial_cash': 100000.0,
            'risk_tolerance': 'moderate',
            'investment_objective': 'growth'
        }
        
        # Mock position data
        self.mock_position_data = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'strategy_id': 'momentum_breakout',
            'signal_id': 'signal_001'
        }
    
    def test_create_portfolio(self):
        """Test portfolio creation"""
        portfolio = self.portfolio_service.create_portfolio(**self.mock_portfolio_data)
        
        self.assertIsNotNone(portfolio)
        self.assertEqual(portfolio['portfolio_name'], 'Test Portfolio')
        self.assertEqual(portfolio['account_id'], 'DU123456')
        self.assertEqual(portfolio['initial_cash'], 100000.0)
    
    def test_get_portfolio(self):
        """Test portfolio retrieval"""
        # Create portfolio first
        created_portfolio = self.portfolio_service.create_portfolio(**self.mock_portfolio_data)
        portfolio_id = created_portfolio['portfolio_id']
        
        # Retrieve portfolio
        retrieved_portfolio = self.portfolio_service.get_portfolio(portfolio_id)
        
        self.assertIsNotNone(retrieved_portfolio)
        self.assertEqual(retrieved_portfolio['portfolio_id'], portfolio_id)
    
    def test_add_position(self):
        """Test adding position to portfolio"""
        # Create portfolio first
        portfolio = self.portfolio_service.create_portfolio(**self.mock_portfolio_data)
        portfolio_id = portfolio['portfolio_id']
        
        # Add position
        position = self.portfolio_service.add_position(
            portfolio_id=portfolio_id,
            **self.mock_position_data
        )
        
        self.assertIsNotNone(position)
        self.assertEqual(position['symbol'], 'AAPL')
        self.assertEqual(position['quantity'], 100)
        self.assertEqual(position['price'], 150.0)
    
    def test_get_portfolio_positions(self):
        """Test retrieving portfolio positions"""
        # Create portfolio and add position
        portfolio = self.portfolio_service.create_portfolio(**self.mock_portfolio_data)
        portfolio_id = portfolio['portfolio_id']
        
        self.portfolio_service.add_position(portfolio_id=portfolio_id, **self.mock_position_data)
        
        # Get positions
        positions = self.portfolio_service.get_portfolio_positions(portfolio_id)
        
        self.assertIsInstance(positions, list)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'AAPL')

class TestIBKRIntegration(unittest.TestCase):
    """Test cases for IBKR integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.account_manager = AccountManager()
        
        # Mock account configuration
        self.mock_accounts = {
            'DU123456': {
                'account_type': 'paper',
                'account_name': 'Paper Trading Account',
                'status': 'active',
                'max_order_value': 50000.0,
                'max_daily_trades': 100,
                'allowed_symbols': ['AAPL', 'MSFT', 'GOOGL']
            },
            'U123456': {
                'account_type': 'live',
                'account_name': 'Live Trading Account',
                'status': 'active',
                'max_order_value': 25000.0,
                'max_daily_trades': 50,
                'allowed_symbols': ['AAPL', 'MSFT']
            }
        }
    
    @patch('ibkr.account_manager.AccountManager._load_account_config')
    def test_get_available_accounts(self, mock_load_config):
        """Test getting available accounts"""
        mock_load_config.return_value = self.mock_accounts
        
        accounts = self.account_manager.get_available_accounts()
        
        self.assertIsInstance(accounts, dict)
        self.assertIn('DU123456', accounts)
        self.assertIn('U123456', accounts)
    
    @patch('ibkr.account_manager.AccountManager._load_account_config')
    def test_switch_account(self, mock_load_config):
        """Test account switching"""
        mock_load_config.return_value = self.mock_accounts
        
        # Switch to paper account
        success = self.account_manager.switch_account('DU123456')
        self.assertTrue(success)
        self.assertEqual(self.account_manager.get_current_account_id(), 'DU123456')
        
        # Switch to live account
        success = self.account_manager.switch_account('U123456')
        self.assertTrue(success)
        self.assertEqual(self.account_manager.get_current_account_id(), 'U123456')
    
    @patch('ibkr.account_manager.AccountManager._load_account_config')
    def test_account_validation(self, mock_load_config):
        """Test account validation"""
        mock_load_config.return_value = self.mock_accounts
        
        # Valid account
        is_valid = self.account_manager.validate_account('DU123456')
        self.assertTrue(is_valid)
        
        # Invalid account
        is_valid = self.account_manager.validate_account('INVALID123')
        self.assertFalse(is_valid)

class TestPositionSizing(unittest.TestCase):
    """Test cases for Position Sizing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.position_sizer = PositionSizer()
        
        # Mock signal data
        self.mock_signal_data = {
            'symbol': 'AAPL',
            'signal_type': 'BUY_CALL',
            'confidence': 0.8,
            'expected_return': 0.06,
            'max_loss': 0.03,
            'strategy_id': 'momentum_breakout'
        }
        
        # Mock portfolio data
        self.mock_portfolio_data = {
            'total_value': 100000.0,
            'cash_balance': 20000.0,
            'current_positions': [
                {'symbol': 'MSFT', 'market_value': 15000.0, 'sector': 'technology'},
                {'symbol': 'GOOGL', 'market_value': 10000.0, 'sector': 'technology'}
            ],
            'risk_metrics': {
                'portfolio_volatility': 0.15,
                'portfolio_beta': 1.1,
                'max_position_risk': 0.005
            }
        }
    
    def test_confidence_weighted_sizing(self):
        """Test confidence-weighted position sizing"""
        result = self.position_sizer.calculate_position_size(
            signal_data=self.mock_signal_data,
            portfolio_data=self.mock_portfolio_data,
            method=PositionSizingMethod.CONFIDENCE_WEIGHTED
        )
        
        self.assertIsNotNone(result)
        self.assertIn('position_size', result)
        self.assertIn('position_value', result)
        self.assertIn('confidence_adjustment', result)
        self.assertGreater(result['position_size'], 0)
        self.assertLess(result['position_size'], 1.0)  # Should be less than 100%
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion position sizing"""
        result = self.position_sizer.calculate_position_size(
            signal_data=self.mock_signal_data,
            portfolio_data=self.mock_portfolio_data,
            method=PositionSizingMethod.KELLY_CRITERION
        )
        
        self.assertIsNotNone(result)
        self.assertIn('kelly_fraction', result)
        self.assertIn('applied_fraction', result)
        self.assertGreater(result['position_size'], 0)
    
    def test_risk_parity_sizing(self):
        """Test risk parity position sizing"""
        result = self.position_sizer.calculate_position_size(
            signal_data=self.mock_signal_data,
            portfolio_data=self.mock_portfolio_data,
            method=PositionSizingMethod.RISK_PARITY
        )
        
        self.assertIsNotNone(result)
        self.assertIn('risk_contribution', result)
        self.assertGreater(result['position_size'], 0)
    
    def test_position_size_validation(self):
        """Test position size validation"""
        # Test with high confidence signal
        high_confidence_signal = self.mock_signal_data.copy()
        high_confidence_signal['confidence'] = 0.95
        
        result = self.position_sizer.calculate_position_size(
            signal_data=high_confidence_signal,
            portfolio_data=self.mock_portfolio_data,
            method=PositionSizingMethod.CONFIDENCE_WEIGHTED
        )
        
        # Should have larger position size for higher confidence
        self.assertGreater(result['confidence_adjustment'], 1.0)
        
        # Test with low confidence signal
        low_confidence_signal = self.mock_signal_data.copy()
        low_confidence_signal['confidence'] = 0.6
        
        result_low = self.position_sizer.calculate_position_size(
            signal_data=low_confidence_signal,
            portfolio_data=self.mock_portfolio_data,
            method=PositionSizingMethod.CONFIDENCE_WEIGHTED
        )
        
        # Should have smaller position size for lower confidence
        self.assertLess(result_low['confidence_adjustment'], 1.0)

class TestPortfolioOptimization(unittest.TestCase):
    """Test cases for Portfolio Optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Mock asset data
        self.mock_assets = [
            {
                'symbol': 'AAPL',
                'expected_return': 0.08,
                'volatility': 0.25,
                'market_value': 15000.0,
                'sector': 'technology'
            },
            {
                'symbol': 'MSFT',
                'expected_return': 0.07,
                'volatility': 0.22,
                'market_value': 12000.0,
                'sector': 'technology'
            },
            {
                'symbol': 'JNJ',
                'expected_return': 0.05,
                'volatility': 0.18,
                'market_value': 10000.0,
                'sector': 'healthcare'
            }
        ]
        
        # Mock constraints
        self.mock_constraints = {
            'max_weight': 0.15,
            'min_weight': 0.01,
            'max_sector_weight': 0.30,
            'max_turnover': 0.20
        }
    
    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization"""
        result = self.portfolio_optimizer.optimize_portfolio(
            assets=self.mock_assets,
            objective=OptimizationObjective.MAX_SHARPE,
            constraints=self.mock_constraints
        )
        
        self.assertIsNotNone(result)
        self.assertIn('optimal_weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('volatility', result)
        self.assertIn('sharpe_ratio', result)
        
        # Check that weights sum to 1
        weights = result['optimal_weights']
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
    
    def test_min_variance_optimization(self):
        """Test minimum variance optimization"""
        result = self.portfolio_optimizer.optimize_portfolio(
            assets=self.mock_assets,
            objective=OptimizationObjective.MIN_VARIANCE,
            constraints=self.mock_constraints
        )
        
        self.assertIsNotNone(result)
        self.assertIn('optimal_weights', result)
        self.assertIn('volatility', result)
        
        # Minimum variance should have lower volatility than equal weight
        equal_weight_vol = self.portfolio_optimizer._calculate_equal_weight_volatility(self.mock_assets)
        self.assertLessEqual(result['volatility'], equal_weight_vol)
    
    def test_rebalancing_plan(self):
        """Test portfolio rebalancing plan generation"""
        current_assets = self.mock_assets
        target_weights = {'AAPL': 0.4, 'MSFT': 0.35, 'JNJ': 0.25}
        
        rebalancing_plan = self.portfolio_optimizer.rebalance_portfolio(
            current_assets=current_assets,
            target_weights=target_weights,
            constraints=self.mock_constraints
        )
        
        self.assertIsNotNone(rebalancing_plan)
        self.assertIn('trades', rebalancing_plan)
        self.assertIn('total_turnover', rebalancing_plan)
        self.assertIn('transaction_costs', rebalancing_plan)
    
    def test_constraint_validation(self):
        """Test optimization constraint validation"""
        # Test with invalid constraints (max weight too low)
        invalid_constraints = self.mock_constraints.copy()
        invalid_constraints['max_weight'] = 0.005  # Too low for 3 assets
        
        result = self.portfolio_optimizer.optimize_portfolio(
            assets=self.mock_assets,
            objective=OptimizationObjective.MAX_SHARPE,
            constraints=invalid_constraints
        )
        
        # Should handle constraint violations gracefully
        self.assertIsNotNone(result)
        self.assertIn('warnings', result)

class TestRiskBudgeting(unittest.TestCase):
    """Test cases for Risk Budgeting"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_budget_manager = RiskBudgetManager()
        
        # Mock entities for risk budgeting
        self.mock_entities = ['momentum_breakout', 'volatility_squeeze', 'gamma_scalping']
        
        # Mock portfolio data
        self.mock_portfolio_data = {
            'total_value': 100000.0,
            'positions': [
                {
                    'symbol': 'AAPL', 'market_value': 15000.0, 'volatility': 0.25,
                    'strategy_id': 'momentum_breakout', 'sector': 'technology'
                },
                {
                    'symbol': 'MSFT', 'market_value': 12000.0, 'volatility': 0.22,
                    'strategy_id': 'momentum_breakout', 'sector': 'technology'
                },
                {
                    'symbol': 'GOOGL', 'market_value': 8000.0, 'volatility': 0.28,
                    'strategy_id': 'volatility_squeeze', 'sector': 'technology'
                }
            ]
        }
    
    def test_create_risk_budget(self):
        """Test risk budget creation"""
        budget = self.risk_budget_manager.create_risk_budget(
            budget_id='test_budget',
            total_budget=0.02,  # 2% VaR budget
            risk_metric=RiskMetric.VAR_95,
            allocation_type=RiskBudgetType.EQUAL_RISK,
            entities=self.mock_entities
        )
        
        self.assertIsNotNone(budget)
        self.assertEqual(budget.budget_id, 'test_budget')
        self.assertEqual(budget.total_budget, 0.02)
        self.assertEqual(len(budget.allocations), len(self.mock_entities))
    
    def test_equal_risk_allocation(self):
        """Test equal risk allocation"""
        budget = self.risk_budget_manager.create_risk_budget(
            budget_id='equal_risk_budget',
            total_budget=0.02,
            risk_metric=RiskMetric.VAR_95,
            allocation_type=RiskBudgetType.EQUAL_RISK,
            entities=self.mock_entities
        )
        
        # Each entity should get equal allocation
        expected_allocation = 0.02 / len(self.mock_entities)
        for entity_id, allocation in budget.allocations.items():
            self.assertAlmostEqual(allocation.allocated_budget, expected_allocation, places=4)
    
    def test_confidence_weighted_allocation(self):
        """Test confidence-weighted allocation"""
        budget = self.risk_budget_manager.create_risk_budget(
            budget_id='confidence_budget',
            total_budget=0.02,
            risk_metric=RiskMetric.VAR_95,
            allocation_type=RiskBudgetType.CONFIDENCE_WEIGHTED,
            entities=self.mock_entities
        )
        
        # Allocations should vary based on confidence
        allocations = list(budget.allocations.values())
        self.assertNotEqual(allocations[0].allocated_budget, allocations[1].allocated_budget)
    
    def test_risk_allocation_update(self):
        """Test risk allocation updates"""
        # Create budget
        budget = self.risk_budget_manager.create_risk_budget(
            budget_id='update_test_budget',
            total_budget=0.02,
            risk_metric=RiskMetric.VAR_95,
            allocation_type=RiskBudgetType.EQUAL_RISK,
            entities=self.mock_entities
        )
        
        # Update allocations
        report = self.risk_budget_manager.update_risk_allocations(
            budget_id='update_test_budget',
            portfolio_data=self.mock_portfolio_data,
            market_regime='normal'
        )
        
        self.assertIsNotNone(report)
        self.assertIn('total_utilization', report)
        self.assertIn('entity_utilizations', report)
        self.assertIn('violations', report)
    
    def test_budget_violation_detection(self):
        """Test budget violation detection"""
        # Create budget with small allocation
        budget = self.risk_budget_manager.create_risk_budget(
            budget_id='violation_test_budget',
            total_budget=0.001,  # Very small budget to trigger violations
            risk_metric=RiskMetric.VAR_95,
            allocation_type=RiskBudgetType.EQUAL_RISK,
            entities=self.mock_entities
        )
        
        # Update with portfolio data that should exceed budget
        report = self.risk_budget_manager.update_risk_allocations(
            budget_id='violation_test_budget',
            portfolio_data=self.mock_portfolio_data,
            market_regime='normal'
        )
        
        # Should detect violations
        self.assertGreater(len(report['violations']), 0)

class TestPortfolioMonitoring(unittest.TestCase):
    """Test cases for Portfolio Monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.portfolio_monitor = PortfolioMonitor()
        self.portfolio_id = 'test_portfolio_monitoring'
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping portfolio monitoring"""
        # Start monitoring
        success = self.portfolio_monitor.start_monitoring(self.portfolio_id)
        self.assertTrue(success)
        
        # Check monitoring status
        self.assertIn(self.portfolio_id, self.portfolio_monitor.monitoring_threads)
        
        # Stop monitoring
        success = self.portfolio_monitor.stop_monitoring(self.portfolio_id)
        self.assertTrue(success)
        
        # Check monitoring stopped
        self.assertNotIn(self.portfolio_id, self.portfolio_monitor.monitoring_threads)
    
    def test_portfolio_snapshot_creation(self):
        """Test portfolio snapshot creation"""
        # Start monitoring
        self.portfolio_monitor.start_monitoring(self.portfolio_id)
        
        # Wait for some snapshots
        time.sleep(2)
        
        # Get current snapshot
        snapshot = self.portfolio_monitor.get_current_snapshot(self.portfolio_id)
        
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.portfolio_id, self.portfolio_id)
        self.assertIsInstance(snapshot.total_value, float)
        self.assertIsInstance(snapshot.positions, list)
        
        # Stop monitoring
        self.portfolio_monitor.stop_monitoring(self.portfolio_id)
    
    def test_performance_analytics_calculation(self):
        """Test performance analytics calculation"""
        # Start monitoring to generate some data
        self.portfolio_monitor.start_monitoring(self.portfolio_id)
        time.sleep(3)  # Wait for data
        
        # Get performance analytics
        analytics = self.portfolio_monitor.get_performance_analytics(self.portfolio_id, '1D')
        
        if analytics:  # May be None if insufficient data
            self.assertEqual(analytics.portfolio_id, self.portfolio_id)
            self.assertIsInstance(analytics.total_return, float)
            self.assertIsInstance(analytics.volatility, float)
            self.assertIsInstance(analytics.sharpe_ratio, float)
        
        # Stop monitoring
        self.portfolio_monitor.stop_monitoring(self.portfolio_id)
    
    def test_real_time_pnl_tracking(self):
        """Test real-time P&L tracking"""
        # Start monitoring
        self.portfolio_monitor.start_monitoring(self.portfolio_id)
        time.sleep(2)  # Wait for data
        
        # Get real-time P&L
        pnl_summary = self.portfolio_monitor.get_real_time_pnl(self.portfolio_id)
        
        self.assertIsInstance(pnl_summary, dict)
        if 'error' not in pnl_summary:
            self.assertIn('total_value', pnl_summary)
            self.assertIn('total_pnl', pnl_summary)
            self.assertIn('daily_pnl', pnl_summary)
        
        # Stop monitoring
        self.portfolio_monitor.stop_monitoring(self.portfolio_id)

class TestPerformanceAttribution(unittest.TestCase):
    """Test cases for Performance Attribution"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.performance_attributor = PerformanceAttributor()
        
        # Mock portfolio data
        self.mock_portfolio_data = {
            'positions': [
                {
                    'symbol': 'AAPL', 'market_value': 15000, 'cost_basis': 14000,
                    'strategy_id': 'momentum_breakout', 'sector': 'technology', 'volatility': 0.25
                },
                {
                    'symbol': 'MSFT', 'market_value': 12000, 'cost_basis': 11500,
                    'strategy_id': 'momentum_breakout', 'sector': 'technology', 'volatility': 0.22
                },
                {
                    'symbol': 'GOOGL', 'market_value': 8000, 'cost_basis': 8500,
                    'strategy_id': 'volatility_squeeze', 'sector': 'technology', 'volatility': 0.28
                },
                {
                    'symbol': 'JNJ', 'market_value': 10000, 'cost_basis': 9800,
                    'strategy_id': 'gamma_scalping', 'sector': 'healthcare', 'volatility': 0.18
                }
            ]
        }
    
    def test_attribution_calculation(self):
        """Test attribution calculation"""
        report = self.performance_attributor.calculate_attribution(
            portfolio_id='test_attribution_portfolio',
            portfolio_data=self.mock_portfolio_data,
            period='1M',
            method=AttributionMethod.BRINSON
        )
        
        self.assertIsNotNone(report)
        self.assertEqual(report.portfolio_id, 'test_attribution_portfolio')
        self.assertIsInstance(report.portfolio_return, float)
        self.assertIsInstance(report.benchmark_return, float)
        self.assertIsInstance(report.segments, list)
        self.assertGreater(len(report.segments), 0)
    
    def test_strategy_attribution(self):
        """Test strategy-level attribution"""
        report = self.performance_attributor.calculate_attribution(
            portfolio_id='test_strategy_attribution',
            portfolio_data=self.mock_portfolio_data,
            period='1M'
        )
        
        # Check strategy attribution
        strategy_segments = [seg for seg in report.segments if seg.segment_type == AttributionLevel.STRATEGY]
        self.assertGreater(len(strategy_segments), 0)
        
        # Should have momentum_breakout, volatility_squeeze, gamma_scalping
        strategy_ids = [seg.segment_id for seg in strategy_segments]
        self.assertIn('momentum_breakout', strategy_ids)
        self.assertIn('volatility_squeeze', strategy_ids)
        self.assertIn('gamma_scalping', strategy_ids)
    
    def test_sector_attribution(self):
        """Test sector-level attribution"""
        report = self.performance_attributor.calculate_attribution(
            portfolio_id='test_sector_attribution',
            portfolio_data=self.mock_portfolio_data,
            period='1M'
        )
        
        # Check sector attribution
        sector_segments = [seg for seg in report.segments if seg.segment_type == AttributionLevel.SECTOR]
        self.assertGreater(len(sector_segments), 0)
        
        # Should have technology and healthcare sectors
        sector_ids = [seg.segment_id for seg in sector_segments]
        self.assertIn('technology', sector_ids)
        self.assertIn('healthcare', sector_ids)
    
    def test_attribution_effects_calculation(self):
        """Test attribution effects calculation"""
        report = self.performance_attributor.calculate_attribution(
            portfolio_id='test_effects',
            portfolio_data=self.mock_portfolio_data,
            period='1M'
        )
        
        # Check that attribution effects are calculated
        self.assertIsInstance(report.total_allocation_effect, float)
        self.assertIsInstance(report.total_selection_effect, float)
        self.assertIsInstance(report.total_interaction_effect, float)
        
        # Check individual segment effects
        for segment in report.segments:
            self.assertIsInstance(segment.allocation_effect, float)
            self.assertIsInstance(segment.selection_effect, float)
            self.assertIsInstance(segment.interaction_effect, float)
    
    def test_top_contributors_detractors(self):
        """Test top contributors and detractors identification"""
        report = self.performance_attributor.calculate_attribution(
            portfolio_id='test_contributors',
            portfolio_data=self.mock_portfolio_data,
            period='1M'
        )
        
        # Check top contributors
        self.assertIsInstance(report.top_contributors, list)
        
        # Check top detractors
        self.assertIsInstance(report.top_detractors, list)
        
        # Contributors should have positive effects, detractors negative
        for contributor in report.top_contributors:
            self.assertGreaterEqual(contributor['contribution'], 0)
        
        for detractor in report.top_detractors:
            self.assertLessEqual(detractor['contribution'], 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete portfolio management system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.portfolio_service = PortfolioService(
            database_url='sqlite:///:memory:',
            redis_url='redis://localhost:6379'
        )
        self.position_sizer = PositionSizer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_budget_manager = RiskBudgetManager()
        self.portfolio_monitor = PortfolioMonitor()
        self.performance_attributor = PerformanceAttributor()
    
    def test_complete_portfolio_workflow(self):
        """Test complete portfolio management workflow"""
        # 1. Create portfolio
        portfolio = self.portfolio_service.create_portfolio(
            account_id='DU123456',
            portfolio_name='Integration Test Portfolio',
            initial_cash=100000.0
        )
        portfolio_id = portfolio['portfolio_id']
        
        # 2. Add positions using position sizing
        signal_data = {
            'symbol': 'AAPL',
            'signal_type': 'BUY_CALL',
            'confidence': 0.8,
            'expected_return': 0.06,
            'max_loss': 0.03,
            'strategy_id': 'momentum_breakout'
        }
        
        portfolio_data = {
            'total_value': 100000.0,
            'cash_balance': 100000.0,
            'current_positions': [],
            'risk_metrics': {'portfolio_volatility': 0.15}
        }
        
        sizing_result = self.position_sizer.calculate_position_size(
            signal_data=signal_data,
            portfolio_data=portfolio_data,
            method=PositionSizingMethod.CONFIDENCE_WEIGHTED
        )
        
        # Add position based on sizing result
        position = self.portfolio_service.add_position(
            portfolio_id=portfolio_id,
            symbol='AAPL',
            quantity=int(sizing_result['position_value'] / 150.0),  # Assume $150 per share
            price=150.0,
            strategy_id='momentum_breakout'
        )
        
        self.assertIsNotNone(position)
        
        # 3. Create risk budget
        risk_budget = self.risk_budget_manager.create_risk_budget(
            budget_id=f'budget_{portfolio_id}',
            total_budget=0.02,
            risk_metric=RiskMetric.VAR_95,
            allocation_type=RiskBudgetType.CONFIDENCE_WEIGHTED,
            entities=['momentum_breakout', 'volatility_squeeze']
        )
        
        self.assertIsNotNone(risk_budget)
        
        # 4. Start monitoring
        success = self.portfolio_monitor.start_monitoring(portfolio_id)
        self.assertTrue(success)
        
        # Wait for some monitoring data
        time.sleep(2)
        
        # 5. Get portfolio snapshot
        snapshot = self.portfolio_monitor.get_current_snapshot(portfolio_id)
        self.assertIsNotNone(snapshot)
        
        # 6. Calculate attribution (with mock data)
        attribution_portfolio_data = {
            'positions': [
                {
                    'symbol': 'AAPL', 'market_value': 15000, 'cost_basis': 14000,
                    'strategy_id': 'momentum_breakout', 'sector': 'technology', 'volatility': 0.25
                }
            ]
        }
        
        attribution_report = self.performance_attributor.calculate_attribution(
            portfolio_id=portfolio_id,
            portfolio_data=attribution_portfolio_data,
            period='1M'
        )
        
        self.assertIsNotNone(attribution_report)
        
        # 7. Stop monitoring
        success = self.portfolio_monitor.stop_monitoring(portfolio_id)
        self.assertTrue(success)
    
    def test_portfolio_optimization_integration(self):
        """Test portfolio optimization integration"""
        # Create portfolio with multiple positions
        portfolio = self.portfolio_service.create_portfolio(
            account_id='DU123456',
            portfolio_name='Optimization Test Portfolio',
            initial_cash=100000.0
        )
        portfolio_id = portfolio['portfolio_id']
        
        # Add multiple positions
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'strategy_id': 'momentum_breakout'},
            {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'strategy_id': 'momentum_breakout'},
            {'symbol': 'GOOGL', 'quantity': 30, 'price': 2500.0, 'strategy_id': 'volatility_squeeze'}
        ]
        
        for pos_data in positions:
            self.portfolio_service.add_position(portfolio_id=portfolio_id, **pos_data)
        
        # Get current positions
        current_positions = self.portfolio_service.get_portfolio_positions(portfolio_id)
        
        # Prepare assets for optimization
        assets = []
        for position in current_positions:
            assets.append({
                'symbol': position['symbol'],
                'expected_return': 0.08,  # Mock expected return
                'volatility': 0.25,      # Mock volatility
                'market_value': position['quantity'] * position['price'],
                'sector': 'technology'   # Mock sector
            })
        
        # Optimize portfolio
        optimization_result = self.portfolio_optimizer.optimize_portfolio(
            assets=assets,
            objective=OptimizationObjective.MAX_SHARPE,
            constraints={'max_weight': 0.4, 'min_weight': 0.1}
        )
        
        self.assertIsNotNone(optimization_result)
        self.assertIn('optimal_weights', optimization_result)
        
        # Generate rebalancing plan
        target_weights = optimization_result['optimal_weights']
        rebalancing_plan = self.portfolio_optimizer.rebalance_portfolio(
            current_assets=assets,
            target_weights=target_weights
        )
        
        self.assertIsNotNone(rebalancing_plan)
        self.assertIn('trades', rebalancing_plan)

class TestPerformance(unittest.TestCase):
    """Performance tests for the portfolio management system"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.portfolio_service = PortfolioService(
            database_url='sqlite:///:memory:',
            redis_url='redis://localhost:6379'
        )
        self.position_sizer = PositionSizer()
        self.portfolio_monitor = PortfolioMonitor()
    
    def test_position_sizing_performance(self):
        """Test position sizing calculation performance"""
        signal_data = {
            'symbol': 'AAPL',
            'signal_type': 'BUY_CALL',
            'confidence': 0.8,
            'expected_return': 0.06,
            'max_loss': 0.03,
            'strategy_id': 'momentum_breakout'
        }
        
        portfolio_data = {
            'total_value': 100000.0,
            'cash_balance': 20000.0,
            'current_positions': [],
            'risk_metrics': {'portfolio_volatility': 0.15}
        }
        
        # Time multiple position sizing calculations
        start_time = time.time()
        
        for _ in range(100):
            result = self.position_sizer.calculate_position_size(
                signal_data=signal_data,
                portfolio_data=portfolio_data,
                method=PositionSizingMethod.CONFIDENCE_WEIGHTED
            )
            self.assertIsNotNone(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Should complete 100 calculations in reasonable time
        self.assertLess(avg_time, 0.01)  # Less than 10ms per calculation
        print(f"Position sizing average time: {avg_time:.4f} seconds")
    
    def test_portfolio_monitoring_performance(self):
        """Test portfolio monitoring performance"""
        portfolio_id = 'performance_test_portfolio'
        
        # Start monitoring
        start_time = time.time()
        success = self.portfolio_monitor.start_monitoring(portfolio_id)
        self.assertTrue(success)
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Check snapshot generation rate
        snapshots = self.portfolio_monitor.portfolio_snapshots[portfolio_id]
        snapshot_count = len(snapshots)
        
        # Should generate multiple snapshots per second
        self.assertGreater(snapshot_count, 2)  # At least 2 snapshots in 3 seconds
        
        # Stop monitoring
        self.portfolio_monitor.stop_monitoring(portfolio_id)
        
        print(f"Generated {snapshot_count} snapshots in 3 seconds")
    
    def test_concurrent_portfolio_monitoring(self):
        """Test concurrent portfolio monitoring performance"""
        portfolio_ids = [f'concurrent_test_portfolio_{i}' for i in range(5)]
        
        # Start monitoring for multiple portfolios
        start_time = time.time()
        
        for portfolio_id in portfolio_ids:
            success = self.portfolio_monitor.start_monitoring(portfolio_id)
            self.assertTrue(success)
        
        # Let them run concurrently
        time.sleep(3)
        
        # Check all portfolios are generating snapshots
        total_snapshots = 0
        for portfolio_id in portfolio_ids:
            snapshots = self.portfolio_monitor.portfolio_snapshots[portfolio_id]
            snapshot_count = len(snapshots)
            total_snapshots += snapshot_count
            self.assertGreater(snapshot_count, 1)  # Each should have at least 1 snapshot
        
        # Stop all monitoring
        for portfolio_id in portfolio_ids:
            self.portfolio_monitor.stop_monitoring(portfolio_id)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Concurrent monitoring: {total_snapshots} total snapshots for {len(portfolio_ids)} portfolios in {total_time:.2f} seconds")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPortfolioService,
        TestIBKRIntegration,
        TestPositionSizing,
        TestPortfolioOptimization,
        TestRiskBudgeting,
        TestPortfolioMonitoring,
        TestPerformanceAttribution,
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
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    print("Running Portfolio Management Service Test Suite")
    print("=" * 60)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)

