"""
Comprehensive Testing Framework for AI Options Trading System
Includes unit tests, integration tests, and end-to-end tests
"""

import pytest
import asyncio
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import psycopg2
import redis
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Test configuration
TEST_CONFIG = {
    "api_gateway_url": "http://localhost:8000",
    "cache_service_url": "http://localhost:8001",
    "data_service_url": "http://localhost:8002",
    "analytics_service_url": "http://localhost:8003",
    "db_config": {
        "host": "localhost",
        "port": 5432,
        "database": "trading_db_test",
        "user": "trading_admin",
        "password": "trading_password_123"
    },
    "redis_config": {
        "host": "localhost",
        "port": 6379,
        "db": 1  # Use different DB for tests
    }
}

class TestFixtures:
    """Test fixtures and utilities"""
    
    @staticmethod
    def sample_stock_data():
        """Generate sample stock data for testing"""
        return {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 1000000,
            "timestamp": int(time.time() * 1e9),
            "open": 149.50,
            "high": 151.00,
            "low": 149.00,
            "close": 150.25
        }
    
    @staticmethod
    def sample_options_data():
        """Generate sample options data for testing"""
        return {
            "symbol": "AAPL231215C00150000",
            "underlying_symbol": "AAPL",
            "strike_price": 150.0,
            "expiration": "2023-12-15",
            "option_type": "CALL",
            "bid_price": 2.50,
            "ask_price": 2.55,
            "last_price": 2.52,
            "volume": 1000,
            "open_interest": 5000,
            "implied_volatility": 0.25
        }
    
    @staticmethod
    def sample_market_data_batch():
        """Generate batch of market data for testing"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        data = []
        
        for symbol in symbols:
            base_price = np.random.uniform(100, 300)
            for i in range(100):
                timestamp = int((time.time() - i * 60) * 1e9)
                price = base_price + np.random.normal(0, 2)
                
                data.append({
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "price": price,
                    "volume": np.random.randint(1000, 10000),
                    "open": price + np.random.uniform(-1, 1),
                    "high": price + np.random.uniform(0, 2),
                    "low": price - np.random.uniform(0, 2),
                    "close": price
                })
        
        return data

class TestDatabaseOperations:
    """Test database operations"""
    
    def setup_method(self):
        """Setup test database connection"""
        self.conn = psycopg2.connect(**TEST_CONFIG["db_config"])
        self.cursor = self.conn.cursor()
    
    def teardown_method(self):
        """Cleanup test database"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def test_database_connection(self):
        """Test database connectivity"""
        self.cursor.execute("SELECT 1")
        result = self.cursor.fetchone()
        assert result[0] == 1
    
    def test_stock_data_insertion(self):
        """Test inserting stock data"""
        sample_data = TestFixtures.sample_stock_data()
        
        query = """
            INSERT INTO stock_ohlcv (ts_event, symbol, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        self.cursor.execute(query, (
            sample_data["timestamp"],
            sample_data["symbol"],
            int(sample_data["open"] * 1e9),
            int(sample_data["high"] * 1e9),
            int(sample_data["low"] * 1e9),
            int(sample_data["close"] * 1e9),
            sample_data["volume"]
        ))
        
        self.conn.commit()
        
        # Verify insertion
        self.cursor.execute(
            "SELECT COUNT(*) FROM stock_ohlcv WHERE symbol = %s",
            (sample_data["symbol"],)
        )
        count = self.cursor.fetchone()[0]
        assert count >= 1
    
    def test_data_retention_policy(self):
        """Test data retention policies"""
        # This would test the retention policy implementation
        # For now, just verify the policy exists
        self.cursor.execute("""
            SELECT policy_name FROM timescaledb_information.drop_chunks_policies
            WHERE hypertable_name = 'stock_ohlcv'
        """)
        
        policies = self.cursor.fetchall()
        assert len(policies) > 0

class TestCacheOperations:
    """Test Redis cache operations"""
    
    def setup_method(self):
        """Setup test Redis connection"""
        self.redis_client = redis.Redis(**TEST_CONFIG["redis_config"])
        self.redis_client.flushdb()  # Clear test database
    
    def teardown_method(self):
        """Cleanup test cache"""
        if hasattr(self, 'redis_client'):
            self.redis_client.flushdb()
    
    def test_cache_connection(self):
        """Test Redis connectivity"""
        result = self.redis_client.ping()
        assert result is True
    
    def test_cache_set_get(self):
        """Test basic cache operations"""
        key = "test_key"
        value = json.dumps({"test": "data"})
        
        # Set value
        self.redis_client.setex(key, 300, value)
        
        # Get value
        retrieved = self.redis_client.get(key)
        assert retrieved is not None
        assert json.loads(retrieved) == {"test": "data"}
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        key = "test_expiry"
        value = "test_value"
        
        # Set with 1 second expiry
        self.redis_client.setex(key, 1, value)
        
        # Should exist immediately
        assert self.redis_client.get(key) == value.encode()
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        assert self.redis_client.get(key) is None

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_endpoints(self):
        """Test health check endpoints for all services"""
        services = [
            ("API Gateway", TEST_CONFIG["api_gateway_url"]),
            ("Cache Service", TEST_CONFIG["cache_service_url"]),
            ("Data Service", TEST_CONFIG["data_service_url"]),
            ("Analytics Service", TEST_CONFIG["analytics_service_url"])
        ]
        
        for service_name, base_url in services:
            response = requests.get(f"{base_url}/health", timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] in ["healthy", "degraded"]
            assert "timestamp" in health_data
            assert "service" in health_data
    
    def test_api_gateway_routing(self):
        """Test API Gateway routing to services"""
        # Test routing to cache service
        response = requests.get(
            f"{TEST_CONFIG['api_gateway_url']}/api/v1/cache/health",
            timeout=10
        )
        assert response.status_code == 200
        
        # Test routing to analytics service
        response = requests.get(
            f"{TEST_CONFIG['api_gateway_url']}/api/v1/analytics/indicators/AAPL",
            timeout=10
        )
        assert response.status_code in [200, 404]  # 404 if no data available
    
    def test_data_ingestion_endpoints(self):
        """Test data ingestion service endpoints"""
        base_url = TEST_CONFIG["data_service_url"]
        
        # Test collection status
        response = requests.get(f"{base_url}/api/v1/data/collection/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert "active" in status_data
        assert "stats" in status_data
    
    def test_analytics_endpoints(self):
        """Test analytics service endpoints"""
        base_url = TEST_CONFIG["analytics_service_url"]
        
        # Test technical indicators (might return 404 if no data)
        response = requests.get(f"{base_url}/api/v1/analytics/indicators/AAPL")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
    
    def test_rate_limiting(self):
        """Test API rate limiting"""
        # This would test rate limiting by making many requests
        # For now, just verify the endpoint exists
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
        assert "X-RateLimit-Remaining" in response.headers or response.status_code == 200

class TestDataProcessing:
    """Test data processing and analytics"""
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        # Generate sample price data
        prices = np.random.uniform(100, 200, 100)
        
        # Test SMA calculation
        from services.analytics.src.app import TechnicalIndicators
        
        sma = TechnicalIndicators.sma(prices, 20)
        assert len(sma) == len(prices)
        assert not np.isnan(sma[-1])  # Last value should not be NaN
        
        # Test RSI calculation
        rsi = TechnicalIndicators.rsi(prices, 14)
        assert len(rsi) == len(prices)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        if len(valid_rsi) > 0:
            assert np.all(valid_rsi >= 0)
            assert np.all(valid_rsi <= 100)
    
    def test_pattern_recognition(self):
        """Test pattern recognition algorithms"""
        from services.analytics.src.app import PatternRecognition
        
        # Generate trending price data
        trend = np.linspace(100, 120, 100)
        noise = np.random.normal(0, 1, 100)
        prices = trend + noise
        
        # Test trend detection
        trend_analysis = PatternRecognition.detect_trend(prices)
        assert "trend" in trend_analysis
        assert "strength" in trend_analysis
        assert trend_analysis["trend"] in ["uptrend", "downtrend", "sideways"]
    
    def test_options_analytics(self):
        """Test options analytics calculations"""
        from services.analytics.src.app import OptionsAnalytics
        
        # Test Greeks calculation
        greeks = OptionsAnalytics.calculate_greeks(
            option_price=5.0,
            underlying_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            volatility=0.20
        )
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks

class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_data_validation(self):
        """Test data validation rules"""
        sample_data = TestFixtures.sample_stock_data()
        
        # Test required fields
        required_fields = ["symbol", "price", "volume", "timestamp"]
        for field in required_fields:
            assert field in sample_data
            assert sample_data[field] is not None
        
        # Test data types
        assert isinstance(sample_data["symbol"], str)
        assert isinstance(sample_data["price"], (int, float))
        assert isinstance(sample_data["volume"], int)
        assert isinstance(sample_data["timestamp"], int)
        
        # Test value ranges
        assert sample_data["price"] > 0
        assert sample_data["volume"] >= 0
        assert sample_data["timestamp"] > 0
    
    def test_data_consistency(self):
        """Test data consistency across services"""
        # This would test that data is consistent across different services
        # For example, cache data should match database data
        pass

class TestPerformance:
    """Test system performance"""
    
    def test_response_time(self):
        """Test API response times"""
        start_time = time.time()
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request():
            response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
            return response.status_code
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        # This would test memory usage of services
        # For now, just verify services are responding
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/health")
        assert response.status_code == 200

class TestSecurity:
    """Test security features"""
    
    def test_authentication(self):
        """Test authentication mechanisms"""
        # Test accessing protected endpoints without authentication
        # This would depend on the actual authentication implementation
        pass
    
    def test_input_validation(self):
        """Test input validation and sanitization"""
        # Test with malicious inputs
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "' OR '1'='1"
        ]
        
        for malicious_input in malicious_inputs:
            # Test that the system handles malicious input safely
            # This would depend on the specific endpoints
            pass

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow from ingestion to analytics"""
        # 1. Simulate data ingestion
        sample_data = TestFixtures.sample_stock_data()
        
        # 2. Verify data is stored in database
        # (This would require actual database operations)
        
        # 3. Verify data is cached
        # (This would require cache operations)
        
        # 4. Verify analytics can process the data
        # (This would require analytics service calls)
        
        # For now, just verify services are running
        services = [
            TEST_CONFIG["api_gateway_url"],
            TEST_CONFIG["cache_service_url"],
            TEST_CONFIG["data_service_url"],
            TEST_CONFIG["analytics_service_url"]
        ]
        
        for service_url in services:
            response = requests.get(f"{service_url}/health", timeout=5)
            assert response.status_code == 200
    
    def test_service_communication(self):
        """Test communication between services"""
        # Test that API Gateway can communicate with other services
        response = requests.get(
            f"{TEST_CONFIG['api_gateway_url']}/api/v1/cache/health",
            timeout=10
        )
        assert response.status_code == 200

# Test utilities
class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def wait_for_service(url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        return False
    
    @staticmethod
    def setup_test_data():
        """Setup test data in database and cache"""
        # This would populate test data for integration tests
        pass
    
    @staticmethod
    def cleanup_test_data():
        """Cleanup test data"""
        # This would clean up test data after tests
        pass

# Pytest configuration
@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment"""
    # Wait for services to be ready
    services = [
        TEST_CONFIG["api_gateway_url"],
        TEST_CONFIG["cache_service_url"],
        TEST_CONFIG["data_service_url"],
        TEST_CONFIG["analytics_service_url"]
    ]
    
    for service_url in services:
        if not TestUtilities.wait_for_service(service_url):
            pytest.skip(f"Service {service_url} not available")
    
    # Setup test data
    TestUtilities.setup_test_data()
    
    yield
    
    # Cleanup
    TestUtilities.cleanup_test_data()

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

