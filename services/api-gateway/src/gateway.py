"""
API Gateway for AI Options Trading System
Central entry point for all microservices with routing, authentication, and rate limiting
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import requests
import logging
import time
import json
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    url: str
    health_endpoint: str = "/health"
    timeout: int = 5
    retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

@dataclass
class RouteConfig:
    """Route configuration"""
    path: str
    service: str
    methods: List[str]
    auth_required: bool = True
    rate_limit: Optional[int] = None
    timeout: int = 30

class CircuitBreaker:
    """Circuit breaker for service calls"""
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = "open"
                
                raise e

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int, burst: int = None):
        self.rate = rate  # tokens per second
        self.burst = burst or rate * 2
        self.tokens = self.burst
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False

class APIGateway:
    """API Gateway for microservices"""
    
    def __init__(self):
        """Initialize API Gateway"""
        self.app = Flask(__name__)
        CORS(self.app, origins="*")  # Enable CORS for all origins
        
        # Service registry
        self.services = {}
        self.circuit_breakers = {}
        self.rate_limiters = {}
        self.routes = {}
        
        # Health monitoring
        self.service_health = {}
        self.health_check_interval = 30
        self.health_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health-check")
        
        # Request tracking
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        
        # JWT settings
        self.jwt_secret = "your-jwt-secret-key"  # Should be from environment
        self.jwt_algorithm = "HS256"
        
        self._setup_routes()
        self._register_services()
        self._start_health_monitoring()
    
    def _setup_routes(self):
        """Setup API Gateway routes"""
        
        @self.app.before_request
        def before_request():
            """Before request middleware"""
            g.start_time = time.time()
            g.request_id = request.headers.get('X-Request-ID', f"req_{int(time.time() * 1000)}")
            
            # Log request
            logger.info(f"Request {g.request_id}: {request.method} {request.path}")
        
        @self.app.after_request
        def after_request(response):
            """After request middleware"""
            duration = time.time() - g.start_time
            
            # Update stats
            self.request_stats["total_requests"] += 1
            if response.status_code < 400:
                self.request_stats["successful_requests"] += 1
            else:
                self.request_stats["failed_requests"] += 1
            
            # Update average response time
            total = self.request_stats["total_requests"]
            current_avg = self.request_stats["average_response_time"]
            self.request_stats["average_response_time"] = (
                (current_avg * (total - 1) + duration * 1000) / total
            )
            
            # Add headers
            response.headers['X-Request-ID'] = g.request_id
            response.headers['X-Response-Time'] = f"{duration * 1000:.2f}ms"
            
            logger.info(f"Response {g.request_id}: {response.status_code} ({duration * 1000:.2f}ms)")
            
            return response
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Global exception handler"""
            logger.error(f"Unhandled exception in request {g.request_id}: {str(e)}")
            
            return jsonify({
                "error": "Internal server error",
                "request_id": g.request_id,
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def gateway_health():
            """API Gateway health check"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "api-gateway",
                "version": "1.0.0",
                "services": self.service_health,
                "stats": self.request_stats
            })
        
        # Service discovery endpoint
        @self.app.route('/services', methods=['GET'])
        def list_services():
            """List registered services"""
            return jsonify({
                "services": {
                    name: {
                        "url": config.url,
                        "status": self.service_health.get(name, "unknown"),
                        "health_endpoint": config.health_endpoint
                    }
                    for name, config in self.services.items()
                }
            })
        
        # Authentication endpoint
        @self.app.route('/auth/login', methods=['POST'])
        def login():
            """User authentication"""
            data = request.get_json()
            
            if not data or 'username' not in data or 'password' not in data:
                return jsonify({"error": "Username and password required"}), 400
            
            # Mock authentication - replace with real auth service
            if self._authenticate_user(data['username'], data['password']):
                token = self._generate_jwt_token(data['username'])
                return jsonify({
                    "token": token,
                    "expires_in": 3600,
                    "user": data['username']
                })
            else:
                return jsonify({"error": "Invalid credentials"}), 401
        
        # Dynamic route handler
        @self.app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
        def proxy_request(path):
            """Proxy requests to appropriate microservices"""
            return self._handle_proxy_request(path)
    
    def _register_services(self):
        """Register microservices"""
        
        # Cache Service
        self.register_service(ServiceConfig(
            name="cache",
            url="http://localhost:8001",
            health_endpoint="/health",
            timeout=5,
            retries=3
        ))
        
        # Data Ingestion Service
        self.register_service(ServiceConfig(
            name="data-ingestion",
            url="http://localhost:8002",
            health_endpoint="/health",
            timeout=10,
            retries=3
        ))
        
        # Analytics Service
        self.register_service(ServiceConfig(
            name="analytics",
            url="http://localhost:8003",
            health_endpoint="/health",
            timeout=15,
            retries=2
        ))
        
        # Signal Generation Service
        self.register_service(ServiceConfig(
            name="signals",
            url="http://localhost:8004",
            health_endpoint="/health",
            timeout=20,
            retries=2
        ))
        
        # Portfolio Service
        self.register_service(ServiceConfig(
            name="portfolio",
            url="http://localhost:8005",
            health_endpoint="/health",
            timeout=10,
            retries=3
        ))
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register service routes"""
        
        # Cache service routes
        cache_routes = [
            RouteConfig("api/v1/cache/<path:subpath>", "cache", ["GET", "POST", "PUT", "DELETE"], auth_required=False),
        ]
        
        # Data ingestion routes
        data_routes = [
            RouteConfig("api/v1/data/<path:subpath>", "data-ingestion", ["GET", "POST"], auth_required=True),
        ]
        
        # Analytics routes
        analytics_routes = [
            RouteConfig("api/v1/analytics/<path:subpath>", "analytics", ["GET", "POST"], auth_required=True),
        ]
        
        # Signal routes
        signal_routes = [
            RouteConfig("api/v1/signals/<path:subpath>", "signals", ["GET", "POST"], auth_required=True),
        ]
        
        # Portfolio routes
        portfolio_routes = [
            RouteConfig("api/v1/portfolio/<path:subpath>", "portfolio", ["GET", "POST", "PUT", "DELETE"], auth_required=True),
        ]
        
        # Register all routes
        for routes in [cache_routes, data_routes, analytics_routes, signal_routes, portfolio_routes]:
            for route in routes:
                self.routes[route.path] = route
    
    def register_service(self, config: ServiceConfig):
        """Register a microservice"""
        self.services[config.name] = config
        self.circuit_breakers[config.name] = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        self.service_health[config.name] = ServiceStatus.UNKNOWN.value
        
        logger.info(f"Registered service: {config.name} at {config.url}")
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def health_check_loop():
            while True:
                try:
                    for service_name, config in self.services.items():
                        self.health_executor.submit(self._check_service_health, service_name, config)
                    
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    time.sleep(5)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
        logger.info("Started health monitoring")
    
    def _check_service_health(self, service_name: str, config: ServiceConfig):
        """Check health of a specific service"""
        try:
            url = f"{config.url}{config.health_endpoint}"
            response = requests.get(url, timeout=config.timeout)
            
            if response.status_code == 200:
                self.service_health[service_name] = ServiceStatus.HEALTHY.value
            else:
                self.service_health[service_name] = ServiceStatus.DEGRADED.value
        
        except Exception as e:
            self.service_health[service_name] = ServiceStatus.UNHEALTHY.value
            logger.warning(f"Health check failed for {service_name}: {e}")
    
    def _authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user (mock implementation)"""
        # Mock authentication - replace with real authentication
        mock_users = {
            "admin": "admin123",
            "trader": "trader123",
            "analyst": "analyst123"
        }
        
        return mock_users.get(username) == password
    
    def _generate_jwt_token(self, username: str) -> str:
        """Generate JWT token"""
        payload = {
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "iss": "ai-ots-gateway"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def _require_auth(self, f):
        """Authentication decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Authentication required"}), 401
            
            token = auth_header.split(' ')[1]
            payload = self._verify_jwt_token(token)
            
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401
            
            g.user = payload
            return f(*args, **kwargs)
        
        return decorated_function
    
    def _find_matching_route(self, path: str) -> Optional[RouteConfig]:
        """Find matching route configuration"""
        # Simple path matching - can be enhanced with regex
        for route_path, route_config in self.routes.items():
            if path.startswith(route_path.replace('<path:subpath>', '')):
                return route_config
        
        return None
    
    def _handle_proxy_request(self, path: str):
        """Handle proxy request to microservice"""
        # Find matching route
        route_config = self._find_matching_route(path)
        
        if not route_config:
            return jsonify({"error": "Route not found"}), 404
        
        # Check authentication
        if route_config.auth_required:
            auth_header = request.headers.get('Authorization')
            
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Authentication required"}), 401
            
            token = auth_header.split(' ')[1]
            payload = self._verify_jwt_token(token)
            
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401
            
            g.user = payload
        
        # Check rate limiting
        if route_config.rate_limit:
            client_id = g.get('user', {}).get('username', request.remote_addr)
            rate_limiter_key = f"{route_config.service}:{client_id}"
            
            if rate_limiter_key not in self.rate_limiters:
                self.rate_limiters[rate_limiter_key] = RateLimiter(route_config.rate_limit)
            
            if not self.rate_limiters[rate_limiter_key].allow_request():
                return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Get service configuration
        service_config = self.services.get(route_config.service)
        if not service_config:
            return jsonify({"error": "Service not found"}), 503
        
        # Check service health
        if self.service_health.get(route_config.service) == ServiceStatus.UNHEALTHY.value:
            return jsonify({"error": "Service unavailable"}), 503
        
        # Proxy request with circuit breaker
        try:
            circuit_breaker = self.circuit_breakers[route_config.service]
            response = circuit_breaker.call(self._make_service_request, service_config, path, route_config)
            return response
        
        except Exception as e:
            logger.error(f"Service request failed: {e}")
            return jsonify({"error": "Service error", "details": str(e)}), 502
    
    def _make_service_request(self, service_config: ServiceConfig, path: str, route_config: RouteConfig):
        """Make request to microservice"""
        # Construct target URL
        target_url = f"{service_config.url}/{path}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers['X-Request-ID'] = g.request_id
        headers['X-Forwarded-For'] = request.remote_addr
        headers['X-Gateway'] = 'ai-ots-gateway'
        
        # Add user context if authenticated
        if hasattr(g, 'user'):
            headers['X-User'] = json.dumps(g.user)
        
        # Prepare request data
        data = None
        if request.method in ['POST', 'PUT', 'PATCH']:
            if request.is_json:
                data = request.get_json()
            else:
                data = request.get_data()
        
        # Make request
        try:
            response = requests.request(
                method=request.method,
                url=target_url,
                headers=headers,
                json=data if request.is_json else None,
                data=data if not request.is_json else None,
                params=request.args,
                timeout=route_config.timeout
            )
            
            # Return response
            return (response.content, response.status_code, response.headers.items())
        
        except requests.exceptions.Timeout:
            raise Exception("Service timeout")
        except requests.exceptions.ConnectionError:
            raise Exception("Service connection error")
        except Exception as e:
            raise Exception(f"Service request error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        return {
            "request_stats": self.request_stats,
            "service_health": self.service_health,
            "registered_services": len(self.services),
            "active_routes": len(self.routes),
            "circuit_breaker_states": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API Gateway"""
        logger.info(f"Starting API Gateway on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

# Factory function
def create_api_gateway() -> APIGateway:
    """Create and configure API Gateway"""
    return APIGateway()

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Options Trading System API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run gateway
    gateway = create_api_gateway()
    gateway.run(host=args.host, port=args.port, debug=args.debug)

