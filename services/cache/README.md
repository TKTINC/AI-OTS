# Cache Service - AI Options Trading System

## Overview

The Cache Service is a high-performance Redis-based caching layer designed specifically for the AI Options Trading System. It provides intelligent caching strategies for different types of trading data including stock prices, options chains, trading signals, user sessions, and analytics.

## Features

### Core Capabilities
- **Multi-Strategy Caching**: Different caching patterns optimized for each data type
- **High Performance**: Redis with connection pooling and compression
- **Real-time Data**: Optimized for high-frequency trading data updates
- **Session Management**: Secure user session handling
- **Rate Limiting**: Built-in API rate limiting
- **Monitoring**: Comprehensive metrics and health checks

### Caching Strategies

#### Stock Prices
- **Strategy**: Write-through with bulk operations
- **TTL**: 5 minutes (configurable)
- **Features**: Latest price tracking, bulk updates, compression

#### Options Data
- **Strategy**: Cache-aside with refresh-ahead
- **TTL**: 5 minutes (configurable)
- **Features**: Options chain caching, summary analytics, background refresh

#### Trading Signals
- **Strategy**: Write-through with read-through fallback
- **TTL**: 30 minutes (configurable)
- **Features**: Signal persistence, symbol-based indexing

#### User Sessions
- **Strategy**: Cache-only storage
- **TTL**: 1 hour (configurable)
- **Features**: Secure session management, automatic expiration

#### Analytics
- **Strategy**: Refresh-ahead with background computation
- **TTL**: 15 minutes (configurable)
- **Features**: Computed analytics caching, background refresh

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Cache Service  │    │     Redis       │
│                 │────│                 │────│                 │
│  Rate Limiting  │    │  Flask API      │    │  Data Storage   │
│  Load Balancer  │    │  Cache Manager  │    │  Pub/Sub        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                       ┌─────────────────┐
                       │    Database     │
                       │                 │
                       │  TimescaleDB    │
                       │  (Fallback)     │
                       └─────────────────┘
```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Redis 7+

### Development Setup

1. **Clone and Navigate**
   ```bash
   cd services/cache
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Services**
   ```bash
   # Start Redis and Cache Service
   docker-compose up -d
   
   # View logs
   docker-compose logs -f cache-service
   ```

4. **Verify Installation**
   ```bash
   # Health check
   curl http://localhost:8001/health
   
   # Detailed health check
   curl http://localhost:8001/health/detailed
   ```

### Production Deployment

1. **Build Production Image**
   ```bash
   docker build -t ai-ots-cache-service:latest .
   ```

2. **Deploy with Environment Variables**
   ```bash
   docker run -d \
     --name ai-ots-cache \
     -p 8001:8001 \
     -e REDIS_HOST=your-redis-host \
     -e REDIS_PASSWORD=your-redis-password \
     -e ENVIRONMENT=production \
     ai-ots-cache-service:latest
   ```

## API Documentation

### Base URL
```
http://localhost:8001/api/v1
```

### Authentication
- API Key: Include `X-API-Key` header for rate limiting
- No authentication required for development

### Stock Prices

#### Get Stock Price
```http
GET /cache/stock-prices/{symbol}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "close_price": 150.25,
    "volume": 1000000,
    "timestamp": "2023-12-01T15:30:00Z"
  },
  "cached": true,
  "timestamp": "2023-12-01T15:30:05Z"
}
```

#### Set Stock Price
```http
POST /cache/stock-prices/{symbol}
Content-Type: application/json

{
  "close_price": 150.25,
  "volume": 1000000,
  "high_price": 151.00,
  "low_price": 149.50
}
```

#### Get Latest Prices
```http
GET /cache/stock-prices/latest?symbols=AAPL,GOOGL,MSFT
```

#### Bulk Update Prices
```http
POST /cache/stock-prices/bulk
Content-Type: application/json

[
  {
    "symbol": "AAPL",
    "close_price": 150.25,
    "volume": 1000000
  },
  {
    "symbol": "GOOGL",
    "close_price": 2800.50,
    "volume": 500000
  }
]
```

### Options Data

#### Get Options Chain
```http
GET /cache/options/{symbol}/{expiry}
```

**Example:**
```http
GET /cache/options/AAPL/2023-12-15
```

#### Set Options Chain
```http
POST /cache/options/{symbol}/{expiry}
Content-Type: application/json

[
  {
    "option_symbol": "AAPL231215C00150000",
    "strike_price": 150.00,
    "option_type": "CALL",
    "bid_price": 2.50,
    "ask_price": 2.55,
    "volume": 1000,
    "implied_volatility": 0.25
  }
]
```

#### Get Options Summary
```http
GET /cache/options/{symbol}/summary
```

### Trading Signals

#### Get Signal
```http
GET /cache/signals/{signal_id}
```

#### Set Signal
```http
POST /cache/signals/{signal_id}
Content-Type: application/json

{
  "symbol": "AAPL",
  "signal_type": "BUY",
  "confidence": 0.85,
  "target_price": 155.00,
  "reasoning": "Strong momentum and volume"
}
```

#### Get Active Signals
```http
GET /cache/signals/active?symbol=AAPL
```

### Sessions

#### Create Session
```http
POST /cache/sessions
Content-Type: application/json

{
  "user_id": "user123",
  "role": "trader",
  "permissions": ["read", "write"]
}
```

#### Get Session
```http
GET /cache/sessions/{session_id}
```

#### Delete Session
```http
DELETE /cache/sessions/{session_id}
```

### Analytics

#### Get Analytics
```http
GET /cache/analytics/{key}
```

#### Set Analytics
```http
POST /cache/analytics/{key}?ttl=900
Content-Type: application/json

{
  "metric": "portfolio_performance",
  "value": 15.5,
  "period": "1d"
}
```

### Market Status

#### Get Market Status
```http
GET /cache/market-status
```

#### Set Market Status
```http
POST /cache/market-status
Content-Type: application/json

{
  "is_open": true,
  "next_close": "2023-12-01T16:00:00Z",
  "session": "regular"
}
```

### Cache Management

#### Get Cache Statistics
```http
GET /cache/stats
```

#### Invalidate Cache
```http
POST /cache/invalidate
Content-Type: application/json

{
  "pattern": "stock_price:AAPL*"
}
```

## Configuration

### Environment Variables

#### Redis Configuration
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password
REDIS_SSL=false
REDIS_MAX_CONNECTIONS=100
```

#### Service Configuration
```bash
CACHE_SERVICE_HOST=0.0.0.0
CACHE_SERVICE_PORT=8001
CACHE_SERVICE_WORKERS=4
DEBUG=false
LOG_LEVEL=INFO
```

#### TTL Settings (seconds)
```bash
TTL_STOCK_PRICES=300
TTL_OPTIONS_DATA=300
TTL_SIGNALS=1800
TTL_USER_SESSIONS=3600
TTL_ANALYTICS=900
TTL_MARKET_STATUS=60
```

#### Rate Limiting
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
```

### Redis Configuration

The service includes an optimized Redis configuration (`redis.conf`) with:
- Memory optimization for trading data
- Persistence settings for data durability
- Performance tuning for high-frequency updates
- Security configurations

## Monitoring

### Health Checks

#### Basic Health Check
```bash
curl http://localhost:8001/health
```

#### Detailed Health Check
```bash
curl http://localhost:8001/health/detailed
```

### Metrics

The service exposes Prometheus metrics at `/metrics` including:
- Cache hit/miss ratios
- Request latency
- Memory usage
- Connection counts
- Error rates

### Logging

Structured logging with configurable levels:
- Console output (development)
- File logging (production)
- JSON format (optional)
- Syslog integration (optional)

## Performance Optimization

### Redis Optimization
- Connection pooling with configurable limits
- Data compression for large payloads
- Memory-efficient data structures
- Optimized eviction policies

### Caching Strategies
- **Write-through**: Immediate consistency for critical data
- **Write-behind**: High performance for bulk operations
- **Cache-aside**: Flexible caching for complex data
- **Refresh-ahead**: Proactive cache warming

### Network Optimization
- Keep-alive connections
- Request/response compression
- Efficient serialization (JSON/Pickle)
- Batch operations support

## Security

### Access Control
- API key-based rate limiting
- IP-based restrictions (configurable)
- Redis password authentication
- SSL/TLS support

### Data Protection
- Session encryption
- Sensitive data masking in logs
- Secure configuration management
- Regular security updates

## Troubleshooting

### Common Issues

#### Redis Connection Errors
```bash
# Check Redis status
docker-compose ps redis

# View Redis logs
docker-compose logs redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping
```

#### High Memory Usage
```bash
# Check Redis memory usage
redis-cli info memory

# Monitor cache statistics
curl http://localhost:8001/cache/stats
```

#### Performance Issues
```bash
# Check slow queries
redis-cli slowlog get 10

# Monitor latency
redis-cli --latency-history -h localhost -p 6379
```

### Debug Mode

Enable debug mode for detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
docker-compose up cache-service
```

### Performance Monitoring

Use the included monitoring stack:
```bash
# Start monitoring services
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
# Default: admin/admin

# Access Prometheus
open http://localhost:9090
```

## Development

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis**
   ```bash
   docker-compose up -d redis
   ```

3. **Run Service**
   ```bash
   export FLASK_ENV=development
   python src/app.py
   ```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_redis_manager.py::test_stock_price_caching
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t ai-ots-cache-service .

# Run container
docker run -d \
  --name cache-service \
  -p 8001:8001 \
  -e REDIS_HOST=redis-host \
  ai-ots-cache-service
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cache-service
  template:
    metadata:
      labels:
        app: cache-service
    spec:
      containers:
      - name: cache-service
        image: ai-ots-cache-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: ENVIRONMENT
          value: "production"
```

### AWS ECS Deployment

Use the provided Terraform configuration in `infrastructure/terraform/ecs.tf` for automated ECS deployment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## License

This project is part of the AI Options Trading System and is proprietary software.

---

For more information, see the main project documentation or contact the development team.

