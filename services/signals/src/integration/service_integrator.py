"""
Service Integration Layer for AI Options Trading System
Connects signal generation with analytics, data ingestion, and cache services
"""

import asyncio
import aiohttp
import requests
import redis
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    host: str
    port: int
    health_endpoint: str = "/health"
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_endpoint}"

class ServiceIntegrator:
    """Integrates signal generation with other microservices"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Service endpoints
        self.services = {
            "data_ingestion": ServiceEndpoint("data_ingestion", "localhost", 8001),
            "analytics": ServiceEndpoint("analytics", "localhost", 8002),
            "cache": ServiceEndpoint("cache", "localhost", 8003),
            "api_gateway": ServiceEndpoint("api_gateway", "localhost", 8000)
        }
        
        # Circuit breaker state
        self.circuit_breakers = {}
        self.service_health = {}
        
        # Request cache
        self.request_cache_ttl = 300  # 5 minutes
        
        # Thread pool for concurrent requests
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize circuit breakers
        for service_name in self.services:
            self.circuit_breakers[service_name] = {
                "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "failure_count": 0,
                "last_failure_time": None,
                "failure_threshold": 5,
                "recovery_timeout": 60  # seconds
            }
    
    def circuit_breaker(self, service_name: str):
        """Circuit breaker decorator for service calls"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                breaker = self.circuit_breakers.get(service_name, {})
                
                # Check circuit breaker state
                if breaker.get("state") == "OPEN":
                    # Check if recovery timeout has passed
                    if (time.time() - breaker.get("last_failure_time", 0)) > breaker.get("recovery_timeout", 60):
                        breaker["state"] = "HALF_OPEN"
                        logger.info(f"Circuit breaker for {service_name} moved to HALF_OPEN")
                    else:
                        raise Exception(f"Circuit breaker OPEN for {service_name}")
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Reset circuit breaker on success
                    if breaker.get("state") in ["HALF_OPEN", "CLOSED"]:
                        breaker["state"] = "CLOSED"
                        breaker["failure_count"] = 0
                        breaker["last_failure_time"] = None
                    
                    return result
                    
                except Exception as e:
                    # Increment failure count
                    breaker["failure_count"] = breaker.get("failure_count", 0) + 1
                    breaker["last_failure_time"] = time.time()
                    
                    # Open circuit breaker if threshold reached
                    if breaker["failure_count"] >= breaker.get("failure_threshold", 5):
                        breaker["state"] = "OPEN"
                        logger.error(f"Circuit breaker OPENED for {service_name}")
                    
                    raise e
            
            return wrapper
        return decorator
    
    def cache_request(self, cache_key: str, ttl: int = None):
        """Cache decorator for expensive requests"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key_data = f"{cache_key}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key_hash = hashlib.md5(key_data.encode()).hexdigest()
                
                # Try to get from cache
                try:
                    cached_result = self.redis_client.get(f"integration_cache:{cache_key_hash}")
                    if cached_result:
                        return json.loads(cached_result)
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                try:
                    cache_ttl = ttl or self.request_cache_ttl
                    self.redis_client.setex(
                        f"integration_cache:{cache_key_hash}",
                        cache_ttl,
                        json.dumps(result, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")
                
                return result
            
            return wrapper
        return decorator
    
    @circuit_breaker("data_ingestion")
    @cache_request("market_data", ttl=60)
    def get_market_data(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Dict[str, Any]:
        """Get market data from data ingestion service"""
        try:
            service = self.services["data_ingestion"]
            url = f"{service.base_url}/api/v1/market-data/{symbol}"
            
            params = {
                "timeframe": timeframe,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=service.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Retrieved market data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise
    
    @circuit_breaker("data_ingestion")
    @cache_request("options_data", ttl=300)
    def get_options_data(self, symbol: str, expiration: str = None) -> Dict[str, Any]:
        """Get options data from data ingestion service"""
        try:
            service = self.services["data_ingestion"]
            url = f"{service.base_url}/api/v1/options/{symbol}"
            
            params = {}
            if expiration:
                params["expiration"] = expiration
            
            response = requests.get(url, params=params, timeout=service.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Retrieved options data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting options data for {symbol}: {e}")
            raise
    
    @circuit_breaker("analytics")
    @cache_request("technical_analysis", ttl=300)
    def get_technical_analysis(self, symbol: str, indicators: List[str] = None) -> Dict[str, Any]:
        """Get technical analysis from analytics service"""
        try:
            service = self.services["analytics"]
            url = f"{service.base_url}/api/v1/technical-analysis/{symbol}"
            
            params = {}
            if indicators:
                params["indicators"] = ",".join(indicators)
            
            response = requests.get(url, params=params, timeout=service.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Retrieved technical analysis for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting technical analysis for {symbol}: {e}")
            raise
    
    @circuit_breaker("analytics")
    @cache_request("pattern_analysis", ttl=600)
    def get_pattern_analysis(self, symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
        """Get pattern analysis from analytics service"""
        try:
            service = self.services["analytics"]
            url = f"{service.base_url}/api/v1/patterns/{symbol}"
            
            params = {"timeframe": timeframe}
            
            response = requests.get(url, params=params, timeout=service.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Retrieved pattern analysis for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting pattern analysis for {symbol}: {e}")
            raise
    
    @circuit_breaker("analytics")
    @cache_request("options_analytics", ttl=300)
    def get_options_analytics(self, symbol: str, strike: float = None, expiration: str = None) -> Dict[str, Any]:
        """Get options analytics from analytics service"""
        try:
            service = self.services["analytics"]
            url = f"{service.base_url}/api/v1/options-analytics/{symbol}"
            
            params = {}
            if strike:
                params["strike"] = strike
            if expiration:
                params["expiration"] = expiration
            
            response = requests.get(url, params=params, timeout=service.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Retrieved options analytics for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting options analytics for {symbol}: {e}")
            raise
    
    @circuit_breaker("cache")
    def cache_signal_data(self, signal_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache signal data in cache service"""
        try:
            service = self.services["cache"]
            url = f"{service.base_url}/api/v1/cache/set"
            
            payload = {
                "key": f"signal:{signal_id}",
                "value": data,
                "ttl": ttl
            }
            
            response = requests.post(url, json=payload, timeout=service.timeout)
            response.raise_for_status()
            
            logger.debug(f"Cached signal data for {signal_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching signal data: {e}")
            return False
    
    @circuit_breaker("cache")
    def get_cached_signal_data(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get cached signal data from cache service"""
        try:
            service = self.services["cache"]
            url = f"{service.base_url}/api/v1/cache/get"
            
            params = {"key": f"signal:{signal_id}"}
            
            response = requests.get(url, params=params, timeout=service.timeout)
            response.raise_for_status()
            
            data = response.json()
            if data.get("success") and data.get("value"):
                logger.debug(f"Retrieved cached signal data for {signal_id}")
                return data["value"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached signal data: {e}")
            return None
    
    def get_comprehensive_market_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market analysis by combining multiple services"""
        try:
            # Use thread pool for concurrent requests
            futures = {}
            
            # Submit all requests concurrently
            futures["market_data"] = self.executor.submit(self.get_market_data, symbol)
            futures["options_data"] = self.executor.submit(self.get_options_data, symbol)
            futures["technical_analysis"] = self.executor.submit(self.get_technical_analysis, symbol)
            futures["pattern_analysis"] = self.executor.submit(self.get_pattern_analysis, symbol)
            futures["options_analytics"] = self.executor.submit(self.get_options_analytics, symbol)
            
            # Collect results
            results = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Error getting {key} for {symbol}: {e}")
                    results[key] = {"error": str(e)}
            
            # Combine results
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_data": results.get("market_data", {}),
                "options_data": results.get("options_data", {}),
                "technical_analysis": results.get("technical_analysis", {}),
                "pattern_analysis": results.get("pattern_analysis", {}),
                "options_analytics": results.get("options_analytics", {}),
                "integration_status": {
                    "services_called": len(futures),
                    "services_successful": len([r for r in results.values() if "error" not in r]),
                    "services_failed": len([r for r in results.values() if "error" in r])
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting comprehensive market analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def enrich_signal_with_analytics(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich signal data with additional analytics"""
        try:
            symbol = signal_data.get("symbol")
            if not symbol:
                return signal_data
            
            # Get comprehensive analysis
            analysis = self.get_comprehensive_market_analysis(symbol)
            
            # Enrich signal data
            enriched_signal = signal_data.copy()
            enriched_signal["enrichment"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_analysis": analysis,
                "enrichment_quality": self._calculate_enrichment_quality(analysis)
            }
            
            # Add derived insights
            enriched_signal["derived_insights"] = self._generate_derived_insights(signal_data, analysis)
            
            return enriched_signal
            
        except Exception as e:
            logger.error(f"Error enriching signal with analytics: {e}")
            return signal_data
    
    def validate_signal_with_services(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal data against multiple services"""
        try:
            symbol = signal_data.get("symbol")
            validation_results = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validations": {},
                "overall_score": 0.0,
                "recommendations": []
            }
            
            # Market data validation
            try:
                market_data = self.get_market_data(symbol, limit=20)
                if market_data and not market_data.get("error"):
                    current_price = market_data.get("current_price", 0)
                    signal_price = signal_data.get("entry_price", 0)
                    
                    price_diff = abs(current_price - signal_price) / current_price if current_price > 0 else 1
                    
                    validation_results["validations"]["market_data"] = {
                        "valid": price_diff < 0.05,  # Within 5%
                        "current_price": current_price,
                        "signal_price": signal_price,
                        "price_difference": price_diff,
                        "score": max(0, 1 - price_diff * 10)
                    }
                else:
                    validation_results["validations"]["market_data"] = {
                        "valid": False,
                        "error": "Market data unavailable",
                        "score": 0.0
                    }
            except Exception as e:
                validation_results["validations"]["market_data"] = {
                    "valid": False,
                    "error": str(e),
                    "score": 0.0
                }
            
            # Technical analysis validation
            try:
                technical = self.get_technical_analysis(symbol)
                if technical and not technical.get("error"):
                    signal_type = signal_data.get("signal_type", "")
                    
                    # Check if technical indicators support the signal
                    rsi = technical.get("indicators", {}).get("rsi", 50)
                    trend = technical.get("trend", {}).get("direction", "neutral")
                    
                    technical_score = 0.5  # Base score
                    
                    if "BUY" in signal_type.upper():
                        if rsi < 70 and trend in ["bullish", "neutral"]:
                            technical_score = 0.8
                        elif rsi > 80 or trend == "bearish":
                            technical_score = 0.2
                    elif "SELL" in signal_type.upper():
                        if rsi > 30 and trend in ["bearish", "neutral"]:
                            technical_score = 0.8
                        elif rsi < 20 or trend == "bullish":
                            technical_score = 0.2
                    
                    validation_results["validations"]["technical_analysis"] = {
                        "valid": technical_score > 0.5,
                        "rsi": rsi,
                        "trend": trend,
                        "score": technical_score
                    }
                else:
                    validation_results["validations"]["technical_analysis"] = {
                        "valid": False,
                        "error": "Technical analysis unavailable",
                        "score": 0.0
                    }
            except Exception as e:
                validation_results["validations"]["technical_analysis"] = {
                    "valid": False,
                    "error": str(e),
                    "score": 0.0
                }
            
            # Options validation (if applicable)
            if "CALL" in signal_data.get("signal_type", "").upper() or "PUT" in signal_data.get("signal_type", "").upper():
                try:
                    options_analytics = self.get_options_analytics(symbol)
                    if options_analytics and not options_analytics.get("error"):
                        iv_percentile = options_analytics.get("iv_analysis", {}).get("iv_percentile", 50)
                        
                        options_score = 0.5  # Base score
                        
                        # Prefer buying options when IV is low, selling when high
                        if "BUY" in signal_data.get("signal_type", "").upper():
                            options_score = max(0.1, 1 - iv_percentile / 100)
                        elif "SELL" in signal_data.get("signal_type", "").upper():
                            options_score = max(0.1, iv_percentile / 100)
                        
                        validation_results["validations"]["options_analysis"] = {
                            "valid": options_score > 0.4,
                            "iv_percentile": iv_percentile,
                            "score": options_score
                        }
                    else:
                        validation_results["validations"]["options_analysis"] = {
                            "valid": False,
                            "error": "Options analysis unavailable",
                            "score": 0.0
                        }
                except Exception as e:
                    validation_results["validations"]["options_analysis"] = {
                        "valid": False,
                        "error": str(e),
                        "score": 0.0
                    }
            
            # Calculate overall score
            scores = [v.get("score", 0) for v in validation_results["validations"].values()]
            validation_results["overall_score"] = sum(scores) / len(scores) if scores else 0
            
            # Generate recommendations
            if validation_results["overall_score"] < 0.3:
                validation_results["recommendations"].append("Signal validation score is low - consider additional analysis")
            
            if validation_results["validations"].get("market_data", {}).get("price_difference", 0) > 0.02:
                validation_results["recommendations"].append("Significant price difference detected - verify entry price")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating signal with services: {e}")
            return {"error": str(e)}
    
    def check_service_health(self) -> Dict[str, Any]:
        """Check health of all integrated services"""
        health_status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {},
            "overall_health": "healthy"
        }
        
        unhealthy_count = 0
        
        for service_name, service in self.services.items():
            try:
                response = requests.get(service.health_url, timeout=5)
                
                if response.status_code == 200:
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "circuit_breaker": self.circuit_breakers[service_name]["state"]
                    }
                else:
                    health_status["services"][service_name] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "circuit_breaker": self.circuit_breakers[service_name]["state"]
                    }
                    unhealthy_count += 1
                    
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "circuit_breaker": self.circuit_breakers[service_name]["state"]
                }
                unhealthy_count += 1
        
        # Determine overall health
        if unhealthy_count == 0:
            health_status["overall_health"] = "healthy"
        elif unhealthy_count < len(self.services) / 2:
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
        
        return health_status
    
    def _calculate_enrichment_quality(self, analysis: Dict[str, Any]) -> float:
        """Calculate quality score for enrichment data"""
        try:
            quality_factors = []
            
            # Check data completeness
            for key in ["market_data", "technical_analysis", "pattern_analysis", "options_analytics"]:
                data = analysis.get(key, {})
                if data and not data.get("error"):
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.0)
            
            # Check data freshness (if timestamp available)
            # This would be implemented based on actual data structure
            
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating enrichment quality: {e}")
            return 0.0
    
    def _generate_derived_insights(self, signal_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate derived insights from signal and analysis data"""
        try:
            insights = {
                "market_regime": "unknown",
                "volatility_environment": "unknown",
                "trend_strength": 0.0,
                "support_resistance_levels": [],
                "risk_factors": [],
                "opportunity_factors": []
            }
            
            # Analyze market regime
            technical = analysis.get("technical_analysis", {})
            if technical and not technical.get("error"):
                trend = technical.get("trend", {})
                if trend:
                    insights["market_regime"] = trend.get("direction", "unknown")
                    insights["trend_strength"] = trend.get("strength", 0.0)
            
            # Analyze volatility environment
            options_analytics = analysis.get("options_analytics", {})
            if options_analytics and not options_analytics.get("error"):
                iv_analysis = options_analytics.get("iv_analysis", {})
                if iv_analysis:
                    iv_percentile = iv_analysis.get("iv_percentile", 50)
                    if iv_percentile > 75:
                        insights["volatility_environment"] = "high"
                    elif iv_percentile < 25:
                        insights["volatility_environment"] = "low"
                    else:
                        insights["volatility_environment"] = "normal"
            
            # Extract support/resistance levels
            pattern_analysis = analysis.get("pattern_analysis", {})
            if pattern_analysis and not pattern_analysis.get("error"):
                levels = pattern_analysis.get("support_resistance", {})
                if levels:
                    insights["support_resistance_levels"] = [
                        {"level": level, "type": "support"} for level in levels.get("support_levels", [])
                    ] + [
                        {"level": level, "type": "resistance"} for level in levels.get("resistance_levels", [])
                    ]
            
            # Identify risk factors
            if insights["volatility_environment"] == "high":
                insights["risk_factors"].append("High volatility environment increases risk")
            
            if insights["trend_strength"] < 0.3:
                insights["risk_factors"].append("Weak trend may lead to choppy price action")
            
            # Identify opportunity factors
            if insights["volatility_environment"] == "low" and "BUY" in signal_data.get("signal_type", ""):
                insights["opportunity_factors"].append("Low volatility environment favorable for buying options")
            
            if insights["trend_strength"] > 0.7:
                insights["opportunity_factors"].append("Strong trend provides good directional opportunity")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating derived insights: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown the integrator"""
        self.executor.shutdown(wait=True)
        logger.info("Service integrator shutdown complete")

# Factory function
def create_service_integrator(redis_host: str = "localhost", redis_port: int = 6379) -> ServiceIntegrator:
    """Create service integrator"""
    return ServiceIntegrator(redis_host, redis_port)

if __name__ == "__main__":
    # Example usage
    integrator = create_service_integrator()
    
    # Check service health
    health = integrator.check_service_health()
    print(f"Overall health: {health['overall_health']}")
    
    # Get comprehensive analysis
    analysis = integrator.get_comprehensive_market_analysis("AAPL")
    print(f"Analysis services called: {analysis.get('integration_status', {}).get('services_called', 0)}")
    
    # Example signal validation
    signal_data = {
        "signal_id": "test_001",
        "symbol": "AAPL",
        "signal_type": "BUY_CALL",
        "entry_price": 150.0,
        "confidence": 0.75
    }
    
    validation = integrator.validate_signal_with_services(signal_data)
    print(f"Signal validation score: {validation.get('overall_score', 0):.2f}")
    
    # Shutdown
    integrator.shutdown()

