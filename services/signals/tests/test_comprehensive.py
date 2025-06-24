"""
Comprehensive Testing Framework for Signal Generation Service
Unit tests, integration tests, performance tests, and end-to-end testing
"""

import unittest
import pytest
import asyncio
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
import requests
import redis
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.signal_service import SignalService, Signal, SignalType, SignalPriority
from strategies.options_strategies import OptionsStrategies
from patterns.pattern_recognition import PatternRecognition
from scoring.signal_scoring import SignalScoring
from broadcasting.signal_broadcaster import SignalBroadcaster, NotificationPreferences, NotificationChannel
from history.signal_tracker import SignalTracker, SignalExecution, SignalStatus, SignalOutcome
from integration.service_integrator import ServiceIntegrator

class TestSignalService(unittest.TestCase):
    """Test cases for core signal service"""
    
    def setUp(self):
        """Set up test environment"""
        self.signal_service = SignalService()
        self.test_symbol = "AAPL"
        
    def test_signal_creation(self):
        """Test signal creation"""
        signal_data = {
            "symbol": self.test_symbol,
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08,
            "entry_price": 150.0,
            "target_price": 162.0,
            "stop_loss": 145.0
        }
        
        signal = self.signal_service.create_signal(signal_data)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.symbol, self.test_symbol)
        self.assertEqual(signal.confidence, 0.75)
        self.assertIsNotNone(signal.signal_id)
        
    def test_signal_validation(self):
        """Test signal validation"""
        # Valid signal
        valid_signal_data = {
            "symbol": "AAPL",
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08,
            "entry_price": 150.0,
            "target_price": 162.0,
            "stop_loss": 145.0
        }
        
        self.assertTrue(self.signal_service.validate_signal_data(valid_signal_data))
        
        # Invalid signal - missing required fields
        invalid_signal_data = {
            "symbol": "AAPL",
            "confidence": 0.75
        }
        
        self.assertFalse(self.signal_service.validate_signal_data(invalid_signal_data))
        
        # Invalid signal - bad confidence range
        invalid_confidence_data = {
            "symbol": "AAPL",
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "confidence": 1.5,  # Invalid confidence > 1
            "expected_return": 0.08,
            "entry_price": 150.0,
            "target_price": 162.0,
            "stop_loss": 145.0
        }
        
        self.assertFalse(self.signal_service.validate_signal_data(invalid_confidence_data))
    
    def test_signal_expiry(self):
        """Test signal expiry functionality"""
        signal_data = {
            "symbol": self.test_symbol,
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08,
            "entry_price": 150.0,
            "target_price": 162.0,
            "stop_loss": 145.0,
            "time_horizon": 1  # 1 hour
        }
        
        signal = self.signal_service.create_signal(signal_data)
        
        # Signal should not be expired immediately
        self.assertFalse(self.signal_service.is_signal_expired(signal))
        
        # Manually set expiry time to past
        signal.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        self.assertTrue(self.signal_service.is_signal_expired(signal))

class TestOptionsStrategies(unittest.TestCase):
    """Test cases for options strategies"""
    
    def setUp(self):
        """Set up test environment"""
        self.strategies = OptionsStrategies()
        self.mock_market_data = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "volume": 1000000,
            "high": 152.0,
            "low": 148.0,
            "previous_close": 149.0
        }
        
        self.mock_technical_data = {
            "rsi": 65.0,
            "macd": {"macd": 1.2, "signal": 0.8, "histogram": 0.4},
            "bollinger_bands": {"upper": 155.0, "middle": 150.0, "lower": 145.0},
            "volume_sma": 800000
        }
        
        self.mock_options_data = {
            "iv_percentile": 45.0,
            "iv_rank": 0.4,
            "gamma": 0.05,
            "delta": 0.6,
            "theta": -0.02,
            "vega": 0.15
        }
    
    def test_momentum_breakout_strategy(self):
        """Test momentum breakout strategy"""
        signal = self.strategies.momentum_breakout_strategy(
            self.mock_market_data,
            self.mock_technical_data,
            self.mock_options_data
        )
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal["symbol"], "AAPL")
        self.assertIn("signal_type", signal)
        self.assertIn("confidence", signal)
        self.assertGreaterEqual(signal["confidence"], 0.0)
        self.assertLessEqual(signal["confidence"], 1.0)
    
    def test_volatility_squeeze_strategy(self):
        """Test volatility squeeze strategy"""
        # Mock data for volatility squeeze
        squeeze_technical_data = self.mock_technical_data.copy()
        squeeze_technical_data["bollinger_bands"] = {
            "upper": 151.0, "middle": 150.0, "lower": 149.0  # Tight bands
        }
        
        signal = self.strategies.volatility_squeeze_strategy(
            self.mock_market_data,
            squeeze_technical_data,
            self.mock_options_data
        )
        
        self.assertIsNotNone(signal)
        self.assertIn("reasoning", signal)
        self.assertIn("volatility squeeze", signal["reasoning"].lower())
    
    def test_strategy_risk_management(self):
        """Test strategy risk management"""
        signal = self.strategies.momentum_breakout_strategy(
            self.mock_market_data,
            self.mock_technical_data,
            self.mock_options_data
        )
        
        # Check risk management fields
        self.assertIn("max_risk", signal)
        self.assertIn("risk_reward_ratio", signal)
        self.assertGreaterEqual(signal["risk_reward_ratio"], 2.0)  # Minimum 2:1 ratio

class TestPatternRecognition(unittest.TestCase):
    """Test cases for pattern recognition"""
    
    def setUp(self):
        """Set up test environment"""
        self.pattern_recognition = PatternRecognition()
        
        # Mock price data for pattern testing
        self.mock_price_data = [
            {"timestamp": "2024-01-01T10:00:00Z", "open": 148.0, "high": 150.0, "low": 147.0, "close": 149.0, "volume": 100000},
            {"timestamp": "2024-01-01T10:05:00Z", "open": 149.0, "high": 151.0, "low": 148.5, "close": 150.5, "volume": 120000},
            {"timestamp": "2024-01-01T10:10:00Z", "open": 150.5, "high": 152.0, "low": 150.0, "close": 151.5, "volume": 110000},
            {"timestamp": "2024-01-01T10:15:00Z", "open": 151.5, "high": 153.0, "low": 151.0, "close": 152.0, "volume": 130000},
            {"timestamp": "2024-01-01T10:20:00Z", "open": 152.0, "high": 152.5, "low": 150.5, "close": 151.0, "volume": 90000}
        ]
    
    def test_trend_analysis(self):
        """Test trend analysis"""
        trend = self.pattern_recognition.analyze_trend(self.mock_price_data)
        
        self.assertIsNotNone(trend)
        self.assertIn("direction", trend)
        self.assertIn("strength", trend)
        self.assertIn(trend["direction"], ["bullish", "bearish", "neutral"])
        self.assertGreaterEqual(trend["strength"], 0.0)
        self.assertLessEqual(trend["strength"], 1.0)
    
    def test_support_resistance_detection(self):
        """Test support and resistance detection"""
        levels = self.pattern_recognition.detect_support_resistance(self.mock_price_data)
        
        self.assertIsNotNone(levels)
        self.assertIn("support_levels", levels)
        self.assertIn("resistance_levels", levels)
        self.assertIsInstance(levels["support_levels"], list)
        self.assertIsInstance(levels["resistance_levels"], list)
    
    def test_breakout_detection(self):
        """Test breakout pattern detection"""
        breakout = self.pattern_recognition.detect_breakout_patterns(self.mock_price_data)
        
        self.assertIsNotNone(breakout)
        self.assertIn("breakout_detected", breakout)
        self.assertIsInstance(breakout["breakout_detected"], bool)
        
        if breakout["breakout_detected"]:
            self.assertIn("direction", breakout)
            self.assertIn("strength", breakout)

class TestSignalScoring(unittest.TestCase):
    """Test cases for signal scoring"""
    
    def setUp(self):
        """Set up test environment"""
        self.signal_scoring = SignalScoring()
        
        self.mock_signal_data = {
            "symbol": "AAPL",
            "strategy_name": "Momentum Breakout",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08,
            "entry_price": 150.0,
            "target_price": 162.0,
            "stop_loss": 145.0
        }
        
        self.mock_market_context = {
            "current_price": 150.0,
            "volume": 1000000,
            "volatility": 0.25,
            "market_cap": 3000000000000,  # $3T
            "sector": "Technology"
        }
    
    def test_signal_scoring(self):
        """Test signal scoring functionality"""
        score = self.signal_scoring.score_signal(self.mock_signal_data, self.mock_market_context)
        
        self.assertIsNotNone(score)
        self.assertIn("overall_score", score)
        self.assertIn("quality_grade", score)
        self.assertIn("priority", score)
        self.assertIn("confidence_score", score)
        
        # Check score ranges
        self.assertGreaterEqual(score["overall_score"], 0)
        self.assertLessEqual(score["overall_score"], 100)
        self.assertIn(score["quality_grade"], ["A", "B", "C", "D"])
    
    def test_portfolio_ranking(self):
        """Test portfolio-level signal ranking"""
        signals = [
            {**self.mock_signal_data, "signal_id": "1", "confidence": 0.8},
            {**self.mock_signal_data, "signal_id": "2", "confidence": 0.6, "symbol": "MSFT"},
            {**self.mock_signal_data, "signal_id": "3", "confidence": 0.9, "expected_return": 0.12}
        ]
        
        ranked_signals = self.signal_scoring.rank_signals_for_portfolio(signals, {})
        
        self.assertEqual(len(ranked_signals), 3)
        
        # Check that signals are ranked (highest score first)
        for i in range(len(ranked_signals) - 1):
            self.assertGreaterEqual(
                ranked_signals[i]["portfolio_score"],
                ranked_signals[i + 1]["portfolio_score"]
            )

class TestSignalBroadcaster(unittest.TestCase):
    """Test cases for signal broadcasting"""
    
    def setUp(self):
        """Set up test environment"""
        # Use in-memory Redis for testing
        self.redis_mock = Mock()
        with patch('redis.Redis', return_value=self.redis_mock):
            self.broadcaster = SignalBroadcaster()
    
    def test_signal_broadcast(self):
        """Test signal broadcasting"""
        signal_data = {
            "signal_id": "test_001",
            "symbol": "AAPL",
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "priority": 3,
            "confidence": 0.75,
            "expected_return": 0.08
        }
        
        result = self.broadcaster.broadcast_signal(signal_data)
        self.assertTrue(result)
    
    def test_user_subscription(self):
        """Test user subscription management"""
        preferences = NotificationPreferences(
            user_id="test_user",
            channels=[NotificationChannel.WEBSOCKET],
            min_priority=2,
            symbols=["AAPL", "MSFT"]
        )
        
        result = self.broadcaster.subscribe_user("test_user", preferences)
        self.assertTrue(result)
        
        # Test unsubscription
        result = self.broadcaster.unsubscribe_user("test_user")
        self.assertTrue(result)

class TestSignalTracker(unittest.TestCase):
    """Test cases for signal tracking"""
    
    def setUp(self):
        """Set up test environment"""
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.tracker = SignalTracker(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.temp_db.name)
    
    def test_signal_tracking(self):
        """Test signal generation tracking"""
        signal_data = {
            "signal_id": "test_001",
            "symbol": "AAPL",
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08
        }
        
        result = self.tracker.track_signal_generation(signal_data)
        self.assertTrue(result)
        
        # Verify signal was stored
        history = self.tracker.get_signal_history(limit=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["signal_id"], "test_001")
    
    def test_execution_tracking(self):
        """Test execution tracking"""
        execution = SignalExecution(
            signal_id="test_001",
            execution_id="exec_001",
            user_id="user_001",
            symbol="AAPL",
            strategy_name="Test Strategy",
            signal_type="BUY_CALL",
            executed_at=datetime.now(timezone.utc),
            execution_price=150.0,
            position_size=1000
        )
        
        result = self.tracker.track_signal_execution(execution)
        self.assertTrue(result)
        
        # Verify execution was stored
        executions = self.tracker.get_execution_history(limit=1)
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0]["execution_id"], "exec_001")
    
    def test_performance_report(self):
        """Test performance report generation"""
        # Add some test data first
        signal_data = {
            "signal_id": "test_001",
            "symbol": "AAPL",
            "strategy_name": "Test Strategy",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08
        }
        self.tracker.track_signal_generation(signal_data)
        
        execution = SignalExecution(
            signal_id="test_001",
            execution_id="exec_001",
            user_id="user_001",
            symbol="AAPL",
            strategy_name="Test Strategy",
            signal_type="BUY_CALL",
            executed_at=datetime.now(timezone.utc),
            execution_price=150.0,
            position_size=1000,
            realized_pnl=80.0,
            actual_return=0.08,
            outcome=SignalOutcome.WIN
        )
        self.tracker.track_signal_execution(execution)
        
        report = self.tracker.generate_performance_report()
        
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.total_signals, 0)
        self.assertGreaterEqual(report.win_rate, 0.0)
        self.assertLessEqual(report.win_rate, 1.0)

class TestServiceIntegrator(unittest.TestCase):
    """Test cases for service integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.redis_mock = Mock()
        with patch('redis.Redis', return_value=self.redis_mock):
            self.integrator = ServiceIntegrator()
    
    @patch('requests.get')
    def test_market_data_retrieval(self, mock_get):
        """Test market data retrieval"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "volume": 1000000
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        data = self.integrator.get_market_data("AAPL")
        
        self.assertIsNotNone(data)
        self.assertEqual(data["symbol"], "AAPL")
        self.assertEqual(data["current_price"], 150.0)
    
    @patch('requests.get')
    def test_service_health_check(self, mock_get):
        """Test service health checking"""
        # Mock healthy response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_get.return_value = mock_response
        
        health = self.integrator.check_service_health()
        
        self.assertIsNotNone(health)
        self.assertIn("overall_health", health)
        self.assertIn("services", health)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        service_name = "test_service"
        
        # Initialize circuit breaker
        self.integrator.circuit_breakers[service_name] = {
            "state": "CLOSED",
            "failure_count": 0,
            "last_failure_time": None,
            "failure_threshold": 2,
            "recovery_timeout": 1
        }
        
        @self.integrator.circuit_breaker(service_name)
        def failing_function():
            raise Exception("Service unavailable")
        
        # First failure
        with self.assertRaises(Exception):
            failing_function()
        
        # Second failure should open circuit
        with self.assertRaises(Exception):
            failing_function()
        
        # Circuit should now be open
        self.assertEqual(self.integrator.circuit_breakers[service_name]["state"], "OPEN")
        
        # Third call should fail immediately due to open circuit
        with self.assertRaises(Exception):
            failing_function()

class TestPerformance(unittest.TestCase):
    """Performance and load testing"""
    
    def setUp(self):
        """Set up test environment"""
        self.signal_service = SignalService()
    
    def test_signal_generation_performance(self):
        """Test signal generation performance"""
        start_time = time.time()
        
        # Generate 100 signals
        for i in range(100):
            signal_data = {
                "symbol": f"TEST{i:03d}",
                "strategy_name": "Performance Test",
                "signal_type": "BUY_CALL",
                "confidence": 0.75,
                "expected_return": 0.08,
                "entry_price": 100.0 + i,
                "target_price": 108.0 + i,
                "stop_loss": 95.0 + i
            }
            
            signal = self.signal_service.create_signal(signal_data)
            self.assertIsNotNone(signal)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should generate 100 signals in less than 1 second
        self.assertLess(duration, 1.0)
        
        # Calculate signals per second
        signals_per_second = 100 / duration
        self.assertGreater(signals_per_second, 100)
        
        print(f"Signal generation performance: {signals_per_second:.1f} signals/second")
    
    def test_concurrent_signal_processing(self):
        """Test concurrent signal processing"""
        def generate_signals(thread_id, num_signals):
            results = []
            for i in range(num_signals):
                signal_data = {
                    "symbol": f"T{thread_id}_{i:03d}",
                    "strategy_name": "Concurrent Test",
                    "signal_type": "BUY_CALL",
                    "confidence": 0.75,
                    "expected_return": 0.08,
                    "entry_price": 100.0 + i,
                    "target_price": 108.0 + i,
                    "stop_loss": 95.0 + i
                }
                
                signal = self.signal_service.create_signal(signal_data)
                results.append(signal is not None)
            
            return results
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            # Submit 5 threads, each generating 20 signals
            for thread_id in range(5):
                future = executor.submit(generate_signals, thread_id, 20)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in futures:
                results = future.result()
                all_results.extend(results)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All signals should be generated successfully
        self.assertEqual(len(all_results), 100)
        self.assertTrue(all(all_results))
        
        # Should complete in reasonable time
        self.assertLess(duration, 2.0)
        
        print(f"Concurrent processing: 100 signals in {duration:.2f} seconds")

class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        
        # Initialize components
        self.signal_service = SignalService()
        self.tracker = SignalTracker(self.temp_db.name)
        
        # Mock Redis for broadcaster
        self.redis_mock = Mock()
        with patch('redis.Redis', return_value=self.redis_mock):
            self.broadcaster = SignalBroadcaster()
    
    def tearDown(self):
        """Clean up test environment"""
        os.unlink(self.temp_db.name)
    
    def test_complete_signal_lifecycle(self):
        """Test complete signal lifecycle from generation to tracking"""
        # 1. Generate signal
        signal_data = {
            "symbol": "AAPL",
            "strategy_name": "End-to-End Test",
            "signal_type": "BUY_CALL",
            "confidence": 0.75,
            "expected_return": 0.08,
            "entry_price": 150.0,
            "target_price": 162.0,
            "stop_loss": 145.0
        }
        
        signal = self.signal_service.create_signal(signal_data)
        self.assertIsNotNone(signal)
        
        # 2. Track signal generation
        result = self.tracker.track_signal_generation(signal_data)
        self.assertTrue(result)
        
        # 3. Broadcast signal
        result = self.broadcaster.broadcast_signal(signal_data)
        self.assertTrue(result)
        
        # 4. Track signal execution
        execution = SignalExecution(
            signal_id=signal.signal_id,
            execution_id="e2e_exec_001",
            user_id="e2e_user",
            symbol="AAPL",
            strategy_name="End-to-End Test",
            signal_type="BUY_CALL",
            executed_at=datetime.now(timezone.utc),
            execution_price=150.5,
            position_size=1000
        )
        
        result = self.tracker.track_signal_execution(execution)
        self.assertTrue(result)
        
        # 5. Close position
        result = self.tracker.close_signal_position(
            execution.execution_id,
            exit_price=162.0,
            realized_pnl=1150.0,  # (162-150.5) * 100 shares
            outcome=SignalOutcome.WIN
        )
        self.assertTrue(result)
        
        # 6. Generate performance report
        report = self.tracker.generate_performance_report()
        self.assertIsNotNone(report)
        self.assertEqual(report.executed_signals, 1)
        self.assertEqual(report.winning_trades, 1)
        self.assertEqual(report.win_rate, 1.0)
        
        print("End-to-end test completed successfully")

# Test runner configuration
if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSignalService))
    test_suite.addTest(unittest.makeSuite(TestOptionsStrategies))
    test_suite.addTest(unittest.makeSuite(TestPatternRecognition))
    test_suite.addTest(unittest.makeSuite(TestSignalScoring))
    test_suite.addTest(unittest.makeSuite(TestSignalBroadcaster))
    test_suite.addTest(unittest.makeSuite(TestSignalTracker))
    test_suite.addTest(unittest.makeSuite(TestServiceIntegrator))
    test_suite.addTest(unittest.makeSuite(TestPerformance))
    test_suite.addTest(unittest.makeSuite(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)

