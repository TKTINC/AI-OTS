# Portfolio Management Service - AI Options Trading System

## Overview

The Portfolio Management Service is a comprehensive portfolio management system with IBKR (Interactive Brokers) integration, intelligent position sizing, portfolio optimization, risk budgeting, and real-time monitoring capabilities. This service provides runtime-configurable paper/live account switching and advanced portfolio analytics.

## Features

### 🔄 **IBKR Integration with Runtime Account Switching**
- **Seamless Account Switching** - Toggle between paper and live accounts without restart
- **TWS API Integration** - Real-time market data and order execution
- **Account Validation** - Comprehensive account status and permission checking
- **Risk Controls** - Pre-trade validation and position limits

### 📊 **Intelligent Position Sizing**
- **7 Sizing Methods** - Fixed dollar, fixed percent, Kelly criterion, risk parity, volatility adjusted, confidence weighted, optimal F
- **Multi-Factor Adjustment** - Signal confidence, risk, diversification, portfolio constraints
- **Real-time Validation** - Position size and constraint verification

### 🎯 **Portfolio Optimization**
- **6 Optimization Objectives** - Max Sharpe, min variance, max return, risk parity, max diversification, Black-Litterman
- **Advanced Constraints** - Weight limits, sector limits, risk constraints, turnover limits
- **Rebalancing Engine** - Intelligent rebalancing plan generation

### ⚖️ **Risk Budgeting System**
- **6 Allocation Types** - Equal risk, volatility weighted, confidence weighted, strategy weighted, sector weighted, custom
- **6 Risk Metrics** - Volatility, VaR 95%, VaR 99%, expected shortfall, max drawdown, beta adjusted exposure
- **Dynamic Monitoring** - Real-time risk utilization and violation detection

### 📈 **Real-time Portfolio Monitoring**
- **Live P&L Tracking** - Unrealized, realized, and daily P&L monitoring
- **Performance Analytics** - 12 comprehensive performance metrics
- **Risk Monitoring** - Real-time risk metrics and limit monitoring
- **Historical Analysis** - 30-day rolling history with trend analysis

### 📊 **Performance Attribution Analysis**
- **Multi-Level Attribution** - Strategy, sector, symbol, and factor attribution
- **5 Attribution Methods** - Brinson-Hood-Beebower, arithmetic, geometric, interaction, Fama-French
- **Comprehensive Effects** - Allocation, selection, interaction, timing effects
- **Top Contributors/Detractors** - Automatic identification of performance drivers

## Architecture

### Service Components

```
Portfolio Management Service (Port 8005)
├── Core Portfolio Service
├── IBKR Integration Layer
├── Position Sizing Engine
├── Portfolio Optimization Engine
├── Risk Budgeting System
├── Real-time Monitoring
└── Performance Attribution
```

### Dependencies

- **PostgreSQL/TimescaleDB** - Portfolio and position data storage
- **Redis** - High-performance caching and real-time data
- **IBKR TWS/Gateway** - Interactive Brokers API connection
- **Prometheus** - Metrics collection and monitoring
- **Grafana** - Visualization and dashboards

## Quick Start

### Development Setup

1. **Clone and navigate to portfolio service**:
```bash
cd AI-OTS/services/portfolio
```

2. **Start development environment**:
```bash
docker-compose up -d
```

3. **Verify service health**:
```bash
curl http://localhost:8005/health
```

### IBKR Account Configuration

1. **Configure accounts** in `config/accounts.yaml`:
```yaml
accounts:
  DU123456:  # Paper account
    account_type: paper
    account_name: "Paper Trading Account"
    status: active
    max_order_value: 50000.0
    max_daily_trades: 100
    allowed_symbols: ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
  
  U123456:   # Live account
    account_type: live
    account_name: "Live Trading Account"
    status: active
    max_order_value: 25000.0
    max_daily_trades: 50
    allowed_symbols: ["AAPL", "MSFT", "SPY"]
```

2. **Start TWS or IB Gateway** with API enabled

3. **Switch accounts at runtime**:
```bash
# Switch to paper account
curl -X POST http://localhost:8005/api/v1/accounts/DU123456/switch

# Switch to live account
curl -X POST http://localhost:8005/api/v1/accounts/U123456/switch
```

## API Reference

### Portfolio Management

#### Create Portfolio
```bash
POST /api/v1/portfolios
{
  "account_id": "DU123456",
  "portfolio_name": "Options Trading Portfolio",
  "initial_cash": 100000.0,
  "risk_tolerance": "moderate",
  "investment_objective": "growth"
}
```

#### Add Position
```bash
POST /api/v1/portfolios/{portfolio_id}/positions
{
  "symbol": "AAPL",
  "quantity": 100,
  "price": 150.50,
  "strategy_id": "momentum_breakout",
  "signal_id": "signal_001"
}
```

#### Get Portfolio Snapshot
```bash
GET /api/v1/monitoring/{portfolio_id}/snapshot
```

### Position Sizing

#### Calculate Position Size
```bash
POST /api/v1/position-sizing/calculate
{
  "signal_data": {
    "symbol": "AAPL",
    "signal_type": "BUY_CALL",
    "confidence": 0.8,
    "expected_return": 0.06,
    "max_loss": 0.03,
    "strategy_id": "momentum_breakout"
  },
  "portfolio_data": {
    "total_value": 100000.0,
    "cash_balance": 20000.0,
    "current_positions": [],
    "risk_metrics": {"portfolio_volatility": 0.15}
  },
  "method": "confidence_weighted"
}
```

### Portfolio Optimization

#### Optimize Portfolio
```bash
POST /api/v1/optimization/optimize
{
  "assets": [
    {
      "symbol": "AAPL",
      "expected_return": 0.08,
      "volatility": 0.25,
      "market_value": 15000.0,
      "sector": "technology"
    }
  ],
  "objective": "max_sharpe",
  "constraints": {
    "max_weight": 0.15,
    "min_weight": 0.01,
    "max_sector_weight": 0.30
  }
}
```

### Risk Budgeting

#### Create Risk Budget
```bash
POST /api/v1/risk-budgets
{
  "budget_id": "main_portfolio_budget",
  "total_budget": 0.02,
  "risk_metric": "var_95",
  "allocation_type": "confidence_weighted",
  "entities": ["momentum_breakout", "volatility_squeeze", "gamma_scalping"]
}
```

#### Update Risk Allocations
```bash
POST /api/v1/risk-budgets/{budget_id}/update
{
  "portfolio_data": {
    "total_value": 100000.0,
    "positions": [...]
  },
  "market_regime": "normal"
}
```

### Performance Attribution

#### Calculate Attribution
```bash
POST /api/v1/attribution/{portfolio_id}/calculate
{
  "portfolio_data": {
    "positions": [...]
  },
  "benchmark_data": {...},
  "period": "1M",
  "method": "brinson"
}
```

## Configuration

### Environment Variables

```bash
# Service Configuration
PORT=8005
ENVIRONMENT=development
SECRET_KEY=your-secret-key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/portfolio_db
REDIS_URL=redis://localhost:6379

# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Monitoring
PROMETHEUS_PORT=8006
```

### Position Sizing Configuration

```python
position_sizing_config = {
    'max_portfolio_risk': 0.02,    # 2% max portfolio risk
    'max_position_risk': 0.005,    # 0.5% max position risk
    'min_confidence': 0.6,         # 60% minimum confidence
    'base_position_size': 0.02,    # 2% base size
    'kelly_fraction': 0.25,        # 25% of Kelly
    'confidence_multiplier': 2.0   # Confidence scaling
}
```

### Risk Budget Configuration

```python
risk_budget_config = {
    'total_budget': 0.02,          # 2% total VaR budget
    'risk_metric': 'var_95',       # 95% VaR
    'allocation_type': 'confidence_weighted',
    'rebalance_threshold': 0.1,    # 10% threshold
    'violation_threshold': 1.2     # 120% of budget
}
```

## Monitoring and Observability

### Prometheus Metrics

The service exposes comprehensive metrics on port 8006:

- **Portfolio Metrics** - Value, P&L, returns, positions
- **Risk Metrics** - Sharpe ratio, drawdown, volatility, VaR
- **Position Sizing** - Calculations, duration, distribution
- **Optimization** - Calculations, duration, improvements
- **Risk Budgeting** - Utilization, violations, updates
- **IBKR Integration** - Connections, switches, orders
- **Attribution** - Calculations, effects, contributors
- **System Metrics** - API requests, uptime, errors

### Grafana Dashboards

Access Grafana at http://localhost:3001 (admin/admin) for:

- **Portfolio Overview** - Real-time portfolio performance
- **Risk Dashboard** - Risk metrics and budget utilization
- **IBKR Integration** - Connection status and order flow
- **Performance Attribution** - Strategy and sector attribution
- **System Health** - Service metrics and alerts

### Health Checks

```bash
# Service health
curl http://localhost:8005/health

# Component status
curl http://localhost:8005/api/v1/accounts/status
```

## Testing

### Run Test Suite

```bash
# Run all tests
python tests/test_comprehensive.py

# Run specific test class
python -m unittest tests.test_comprehensive.TestPortfolioService

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - Service integration testing
- **Performance Tests** - Load and performance testing
- **IBKR Tests** - Interactive Brokers integration testing

## Production Deployment

### Docker Deployment

```bash
# Build production image
docker build -t portfolio-service:latest .

# Run production container
docker run -d \
  --name portfolio-service \
  -p 8005:8005 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  portfolio-service:latest
```

### AWS ECS Deployment

The service includes complete Terraform configuration for AWS ECS deployment with:

- **Auto Scaling** - Dynamic scaling based on load
- **Load Balancing** - Application Load Balancer
- **Service Discovery** - AWS Cloud Map integration
- **Monitoring** - CloudWatch metrics and alarms
- **Security** - VPC, security groups, IAM roles

## Security Considerations

### IBKR Security

- **Account Isolation** - Separate paper and live account configurations
- **Permission Validation** - Account-specific trading permissions
- **Order Limits** - Maximum order value and daily trade limits
- **Symbol Restrictions** - Configurable allowed trading symbols

### API Security

- **Authentication** - JWT token-based authentication
- **Authorization** - Role-based access control
- **Rate Limiting** - API request rate limiting
- **Input Validation** - Comprehensive input validation

### Data Security

- **Encryption** - Data encryption at rest and in transit
- **Audit Logging** - Comprehensive audit trail
- **Access Control** - Database access controls
- **Backup** - Automated backup and recovery

## Troubleshooting

### Common Issues

#### IBKR Connection Issues
```bash
# Check TWS/Gateway status
curl http://localhost:8005/api/v1/accounts/status

# Verify account configuration
cat config/accounts.yaml

# Check IBKR logs
docker logs portfolio-service | grep IBKR
```

#### Position Sizing Errors
```bash
# Check position sizing configuration
curl http://localhost:8005/api/v1/position-sizing/methods

# Validate signal data format
# Ensure confidence is between 0.6-0.9
# Verify expected_return and max_loss are positive
```

#### Portfolio Optimization Failures
```bash
# Check asset data quality
# Ensure expected returns and volatilities are reasonable
# Verify constraint feasibility
# Check for correlation matrix issues
```

### Performance Optimization

#### Database Optimization
- **Indexing** - Proper database indexing for queries
- **Connection Pooling** - Database connection pooling
- **Query Optimization** - Optimized SQL queries

#### Caching Strategy
- **Redis Caching** - High-frequency data caching
- **Cache Invalidation** - Intelligent cache invalidation
- **Memory Management** - Efficient memory usage

#### Monitoring Optimization
- **Metric Sampling** - Appropriate metric sampling rates
- **Background Processing** - Non-blocking background tasks
- **Resource Limits** - Proper resource allocation

## Support and Documentation

### Additional Resources

- **API Documentation** - Complete API reference
- **Architecture Guide** - Detailed architecture documentation
- **Deployment Guide** - Production deployment instructions
- **Performance Guide** - Performance optimization guide

### Getting Help

For issues and questions:

1. **Check Health Endpoints** - Verify service status
2. **Review Logs** - Check application and container logs
3. **Monitor Metrics** - Use Prometheus/Grafana for insights
4. **Test Components** - Run test suite for validation

## License

This Portfolio Management Service is part of the AI Options Trading System and is proprietary software.

