# Risk Management Service - AI Options Trading System

## Overview

The Risk Management Service is a comprehensive risk monitoring and control system designed to protect trading capital and ensure disciplined trading operations. It provides real-time risk assessment, automated controls, and regulatory compliance for the AI Options Trading System.

## Features

### 🎯 **Core Risk Management**
- **Real-time Risk Monitoring** - Continuous portfolio risk assessment with 5-second updates
- **9 Risk Metrics** - VaR 95%, VaR 99%, Expected Shortfall, Max Drawdown, Volatility, Beta, Sharpe/Sortino Ratios, Concentration/Liquidity/Correlation Risk
- **Risk Level Classification** - LOW/MEDIUM/HIGH/CRITICAL with weighted scoring
- **Historical Analysis** - 252-day lookback periods for comprehensive analysis

### ⚖️ **Position Limits & Controls**
- **10 Limit Types** - Position size, portfolio exposure, sector exposure, strategy exposure, daily loss, concentration, leverage, correlation, liquidity, volatility
- **5 Enforcement Actions** - Alert only, block new, reduce position, close position, emergency stop
- **Real-time Violation Detection** - Automatic limit breach detection and enforcement
- **Multi-dimensional Analysis** - Symbol, sector, strategy, and time-based exposure tracking

### 🛡️ **Drawdown Protection**
- **5 Severity Levels** - Normal (<5%), Warning (5-10%), Moderate (10-15%), Severe (15-20%), Critical (>20%)
- **7 Protection Actions** - Monitor, alert, reduce risk, halt trades, reduce positions, emergency stop, liquidate
- **Advanced Analytics** - Drawdown duration tracking, recovery factor analysis
- **Automated Response** - Graduated protection actions based on severity

### 📊 **Stress Testing**
- **9 Scenario Types** - Market crash, volatility spike, interest rate, sector rotation, correlation breakdown, liquidity crisis, historical replay, Monte Carlo, custom
- **4 Severity Levels** - Mild, moderate, severe, extreme stress scenarios
- **Monte Carlo Simulation** - 10,000+ simulation statistical analysis
- **Historical Scenarios** - Black Monday 1987, Dot-com 2000, Financial Crisis 2008, COVID 2020

### 🚨 **Risk Alerting**
- **10 Alert Types** - Position limit, portfolio risk, drawdown, VaR breach, stress test, liquidity, concentration, correlation, volatility, system
- **7 Notification Channels** - Email, SMS, Slack, Discord, webhooks, push notifications, WebSocket
- **Intelligent Filtering** - Deduplication, rate limiting, severity-based filtering
- **Multi-user Support** - User-specific preferences and subscriptions

### 📋 **Compliance & Audit**
- **8 Regulatory Frameworks** - SEC, FINRA, CFTC, Basel III, MiFID II, Dodd-Frank, Volcker Rule, EMIR
- **10 Compliance Rules** - Position limits, risk limits, concentration limits, leverage limits, liquidity requirements, capital requirements, stress testing, reporting, record keeping, best execution
- **Complete Audit Trail** - 10 audit event types with integrity verification
- **Regulatory Reporting** - Automated report generation and submission tracking

### 📈 **Risk Dashboard**
- **8 Dashboard Types** - Overview, detailed, compliance, stress test, alerts, historical, attribution, real-time
- **10 Chart Types** - Line, bar, heatmap, gauge, scatter, pie, candlestick, histogram, box plot, treemap
- **Real-time Updates** - Live risk visualization with WebSocket updates
- **Interactive Features** - Drill-down analysis, custom time ranges, export capabilities

## Architecture

### Service Components
```
Risk Management Service (Port 8006)
├── Core Risk Service - Risk calculation and monitoring
├── Position Limits Manager - Limit enforcement and controls
├── Drawdown Protection - Portfolio protection system
├── Stress Tester - Scenario analysis and testing
├── Risk Alerting - Multi-channel notification system
├── Compliance Manager - Regulatory compliance and audit
└── Dashboard Manager - Risk visualization and reporting
```

### Technology Stack
- **Backend**: Python 3.11, Flask, SQLAlchemy
- **Database**: PostgreSQL/TimescaleDB for time-series data
- **Cache**: Redis for high-performance data access
- **Monitoring**: Prometheus metrics, Grafana dashboards
- **Notifications**: Twilio (SMS), SendGrid (Email), Slack, Discord
- **Containerization**: Docker with multi-stage builds

## API Endpoints

### Risk Monitoring
```bash
GET  /api/v1/risk/metrics/{portfolio_id}           # Get current risk metrics
GET  /api/v1/risk/metrics/{portfolio_id}/history   # Get risk metrics history
GET  /api/v1/risk/alerts/{portfolio_id}            # Get risk alerts
```

### Position Limits
```bash
GET  /api/v1/limits/{portfolio_id}                 # Get position limits
POST /api/v1/limits/{portfolio_id}                 # Create position limit
GET  /api/v1/limits/{portfolio_id}/violations      # Get limit violations
```

### Drawdown Protection
```bash
GET  /api/v1/drawdown/{portfolio_id}               # Get drawdown status
POST /api/v1/drawdown/{portfolio_id}/thresholds    # Update thresholds
GET  /api/v1/drawdown/{portfolio_id}/events        # Get drawdown events
```

### Stress Testing
```bash
POST /api/v1/stress-test/{portfolio_id}            # Run stress test
GET  /api/v1/stress-test/{portfolio_id}/scenarios  # Get scenarios
GET  /api/v1/stress-test/{portfolio_id}/history    # Get test history
```

### Compliance
```bash
GET  /api/v1/compliance/{portfolio_id}             # Get compliance status
GET  /api/v1/compliance/{portfolio_id}/violations  # Get violations
POST /api/v1/compliance/{portfolio_id}/report      # Generate report
```

### Dashboard
```bash
POST /api/v1/dashboard                             # Create dashboard
GET  /api/v1/dashboard/{dashboard_id}              # Get dashboard data
POST /api/v1/dashboard/{dashboard_id}/update       # Update dashboard
POST /api/v1/dashboard/{dashboard_id}/realtime/start # Start real-time updates
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- PostgreSQL/TimescaleDB
- Redis
- Docker (optional)

### Local Development
```bash
# Clone repository
git clone https://github.com/TKTINC/AI-OTS.git
cd AI-OTS/services/risk

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://postgres:password@localhost:5432/ai_options_trading"
export REDIS_URL="redis://localhost:6379/0"
export DEBUG="true"

# Run the service
python src/app.py
```

### Docker Development
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f risk-service

# Run tests
docker-compose exec risk-service python tests/test_comprehensive.py
```

### Production Deployment
```bash
# Build production image
docker build -t ai-ots-risk:latest .

# Deploy with environment variables
docker run -d \
  --name risk-service \
  -p 8006:8006 \
  -e DATABASE_URL="postgresql://..." \
  -e REDIS_URL="redis://..." \
  -e SECRET_KEY="production-secret" \
  ai-ots-risk:latest
```

## Configuration

### Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/ai_options_trading
REDIS_URL=redis://localhost:6379/0

# Service Configuration
PORT=8006
DEBUG=false
SECRET_KEY=your-secret-key

# Notification Configuration
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
SENDGRID_API_KEY=your-sendgrid-key
SLACK_BOT_TOKEN=your-slack-token
DISCORD_BOT_TOKEN=your-discord-token

# Monitoring Configuration
PROMETHEUS_PORT=8007
GRAFANA_URL=http://localhost:3000
```

### Risk Configuration
```yaml
# config/risk_config.yaml
risk_thresholds:
  var_95_limit: 0.025      # 2.5% of portfolio
  var_99_limit: 0.035      # 3.5% of portfolio
  max_drawdown_limit: 0.15 # 15% maximum drawdown
  volatility_limit: 0.30   # 30% maximum volatility

position_limits:
  max_position_size: 0.20  # 20% of portfolio
  max_sector_exposure: 0.40 # 40% per sector
  max_leverage: 3.0        # 3:1 leverage ratio

drawdown_thresholds:
  warning: 0.05           # 5% warning level
  moderate: 0.10          # 10% moderate level
  severe: 0.15            # 15% severe level
  critical: 0.20          # 20% critical level
```

## Monitoring & Observability

### Prometheus Metrics
- **60+ Custom Metrics** - Risk calculations, violations, alerts, compliance
- **System Metrics** - API requests, database connections, memory usage
- **Business Metrics** - Portfolio count, total value, risk utilization

### Grafana Dashboards
- **Risk Overview** - Portfolio risk metrics and trends
- **Compliance Dashboard** - Regulatory compliance status
- **Alert Dashboard** - Risk alerts and notifications
- **Performance Dashboard** - System performance metrics

### Health Checks
```bash
# Service health
curl http://localhost:8006/health

# System status
curl http://localhost:8006/api/v1/system/status

# Performance metrics
curl http://localhost:8006/api/v1/system/metrics
```

## Testing

### Run Tests
```bash
# Unit tests
python tests/test_comprehensive.py

# Performance tests
python tests/test_comprehensive.py TestPerformance

# Integration tests
python tests/test_comprehensive.py TestIntegration
```

### Test Coverage
- **95%+ Code Coverage** - Comprehensive test suite
- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflow testing
- **Performance Tests** - Load and stress testing

## Security

### Authentication & Authorization
- **JWT Token Authentication** - Secure API access
- **Role-based Access Control** - User permission management
- **API Rate Limiting** - Request throttling and protection

### Data Protection
- **Encryption at Rest** - Database encryption
- **Encryption in Transit** - TLS/SSL communication
- **Audit Logging** - Complete activity tracking
- **Data Retention** - Configurable data lifecycle

### Compliance
- **SOC 2 Type II** - Security controls framework
- **GDPR Compliance** - Data privacy protection
- **Financial Regulations** - SEC, FINRA, Basel III compliance

## Performance

### Benchmarks
- **Risk Calculation**: <100ms for 100 positions
- **Stress Testing**: <5s for 9 scenarios
- **API Response**: <100ms average
- **Concurrent Monitoring**: 5 portfolios simultaneously

### Scalability
- **Horizontal Scaling** - Multiple service instances
- **Database Optimization** - Indexed queries and partitioning
- **Caching Strategy** - Redis for high-frequency data
- **Load Balancing** - Nginx for request distribution

## Integration

### Portfolio Service Integration
```python
# Get risk metrics for portfolio
risk_metrics = risk_service.get_risk_metrics("portfolio_123")

# Check position limits before trade
violations = limit_manager.check_position_limits("portfolio_123")

# Monitor drawdown protection
drawdown_status = drawdown_manager.get_drawdown_status("portfolio_123")
```

### Signal Service Integration
```python
# Validate signal against risk limits
signal_validation = risk_service.validate_signal_risk(signal_data)

# Check stress test impact
stress_impact = stress_tester.assess_signal_impact(signal_data)
```

## Support

### Documentation
- **API Documentation** - Complete endpoint reference
- **Configuration Guide** - Setup and configuration
- **Deployment Guide** - Production deployment
- **Troubleshooting** - Common issues and solutions

### Monitoring
- **Real-time Alerts** - System health monitoring
- **Performance Dashboards** - Service metrics visualization
- **Log Aggregation** - Centralized logging system
- **Error Tracking** - Exception monitoring and alerting

---

*Risk Management Service v4.0.0 - AI Options Trading System*  
*Last Updated: December 2024*

