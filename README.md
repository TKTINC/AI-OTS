# AI Options Trading System (AI-OTS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)

A sophisticated algorithmic trading system designed to identify and capitalize on short-term options trading opportunities through systematic analysis of market data and advanced pattern recognition.

## 🎯 Project Overview

The AI Options Trading System is a comprehensive microservices-based platform that analyzes intraday options price movements to identify consistent 5-10% profit opportunities. The system focuses on the Magnificent 7 stocks (AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META) and major ETFs (SPY, QQQ), providing real-time analytics and signal generation for options trading strategies.

### Key Features

- **Real-time Market Data Ingestion** - High-frequency data collection from institutional sources
- **Advanced Technical Analysis** - Comprehensive suite of technical indicators and pattern recognition
- **Options Analytics** - Specialized calculations for options pricing, Greeks, and volatility analysis
- **Scalable Microservices Architecture** - Cloud-native design with independent service scaling
- **Comprehensive Monitoring** - Full observability with metrics, logging, and alerting
- **Production-Ready Infrastructure** - AWS-based deployment with Infrastructure as Code

## 🏗️ Architecture

The system implements a distributed microservices architecture optimized for high-frequency trading:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Data Ingestion │    │   Analytics     │
│     (8000)      │    │     (8002)      │    │     (8003)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │  Cache Service  │    │   TimescaleDB   │    │     Redis       │
         │     (8001)      │    │     (5432)      │    │     (6379)      │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Services

- **API Gateway** - Central routing, authentication, and rate limiting
- **Data Ingestion Service** - Market data collection and processing
- **Analytics Service** - Technical analysis and pattern recognition
- **Cache Service** - High-performance data caching and retrieval
- **TimescaleDB** - Time-series database for market data storage
- **Redis** - In-memory caching for real-time data access

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- AWS CLI (for production deployment)
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/TKTINC/AI-OTS.git
   cd AI-OTS
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the development environment**
   ```bash
   ./start-dev.sh
   ```

4. **Verify services are running**
   ```bash
   curl http://localhost:8000/health
   ```

### Service URLs

- **API Gateway**: http://localhost:8000
- **Cache Service**: http://localhost:8001
- **Data Ingestion**: http://localhost:8002
- **Analytics**: http://localhost:8003
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 📊 API Documentation

### Health Checks

```bash
# Check all services
curl http://localhost:8000/health

# Individual service health
curl http://localhost:8001/health  # Cache Service
curl http://localhost:8002/health  # Data Ingestion
curl http://localhost:8003/health  # Analytics
```

### Market Data APIs

```bash
# Get technical indicators
curl "http://localhost:8000/api/v1/analytics/indicators/AAPL?period=20&indicators=sma,rsi,macd"

# Get pattern analysis
curl "http://localhost:8000/api/v1/analytics/patterns/AAPL?days=30"

# Get options analytics
curl "http://localhost:8000/api/v1/analytics/options/AAPL"

# Start data collection
curl -X POST "http://localhost:8000/api/v1/data/collection/start" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "GOOGL"], "schemas": ["trades", "tbbo"]}'
```

### Cache Operations

```bash
# Get cached stock price
curl "http://localhost:8000/api/v1/cache/stock-prices/AAPL"

# Cache analytics result
curl -X POST "http://localhost:8000/api/v1/cache/analytics" \
  -H "Content-Type: application/json" \
  -d '{"key": "AAPL_indicators", "data": {...}, "ttl": 300}'
```

## 🛠️ Development

### Project Structure

```
AI-OTS/
├── services/                 # Microservices
│   ├── api-gateway/         # API Gateway service
│   ├── cache/               # Cache service
│   ├── data-ingestion/      # Data ingestion service
│   └── analytics/           # Analytics service
├── infrastructure/          # AWS infrastructure code
│   └── terraform/           # Terraform configurations
├── database/               # Database schemas and migrations
│   ├── schema/             # SQL schema files
│   └── migrations/         # Migration scripts
├── monitoring/             # Monitoring and observability
│   ├── prometheus/         # Prometheus configuration
│   ├── grafana/           # Grafana dashboards
│   └── logging/           # Logging configuration
├── tests/                 # Test suites
├── docs/                  # Documentation
└── docker-compose.yml     # Development environment
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_comprehensive.py::TestAPIEndpoints -v
python -m pytest tests/test_comprehensive.py::TestDataProcessing -v
python -m pytest tests/test_comprehensive.py::TestPerformance -v
```

### Adding New Services

1. Create service directory under `services/`
2. Implement Flask application with health check endpoint
3. Add Dockerfile and requirements.txt
4. Update docker-compose.yml
5. Add service to API Gateway routing
6. Create monitoring dashboards and alerts

### Code Quality

```bash
# Format code
black services/

# Lint code
flake8 services/

# Type checking
mypy services/
```

## 🚀 Production Deployment

### AWS Infrastructure

The system deploys to AWS using Terraform for Infrastructure as Code:

```bash
# Deploy to production
./deploy-prod.sh deploy

# Check deployment status
./deploy-prod.sh status

# Destroy infrastructure
./deploy-prod.sh destroy
```

### Infrastructure Components

- **VPC** - Multi-AZ network with public/private subnets
- **ECS Fargate** - Serverless container execution
- **RDS** - Managed PostgreSQL with TimescaleDB
- **ElastiCache** - Managed Redis clusters
- **ALB** - Application Load Balancer with SSL termination
- **CloudWatch** - Monitoring and logging
- **Secrets Manager** - Secure credential storage

### Environment Configuration

Production deployments require:

1. **AWS Credentials** - Configured via AWS CLI or IAM roles
2. **Databento API Key** - For real market data access
3. **SSL Certificates** - For HTTPS endpoints
4. **Monitoring Setup** - CloudWatch alarms and notifications

## 📈 Monitoring and Observability

### Metrics and Dashboards

The system includes comprehensive monitoring with:

- **System Metrics** - CPU, memory, disk, network usage
- **Application Metrics** - Request rates, response times, error rates
- **Business Metrics** - Signal generation, data quality, trading performance
- **Custom Dashboards** - Grafana dashboards for different user roles

### Logging

Structured logging with multiple output formats:

- **Application Logs** - Service-specific operational logs
- **Access Logs** - HTTP request/response logging
- **Audit Logs** - Security and compliance events
- **Performance Logs** - Detailed timing and throughput metrics

### Alerting

Prometheus alerting rules for:

- **Service Health** - Service outages and degraded performance
- **Data Pipeline** - Data ingestion failures and quality issues
- **Security** - Authentication failures and suspicious activity
- **Business Logic** - Trading signal anomalies and market data issues

## 🔒 Security

### Authentication and Authorization

- **JWT Authentication** - Secure API access with token-based auth
- **Role-Based Access Control** - Fine-grained permissions
- **API Rate Limiting** - Protection against abuse and DoS attacks

### Data Protection

- **Encryption at Rest** - All sensitive data encrypted in storage
- **Encryption in Transit** - TLS for all network communications
- **Secrets Management** - Secure storage of API keys and credentials

### Compliance

- **Audit Logging** - Comprehensive audit trails for compliance
- **Data Retention** - Automated data lifecycle management
- **Access Controls** - Strict access controls for sensitive operations

## 📚 Documentation

### Technical Documentation

- [Week 1 Implementation Summary](docs/WEEK1_IMPLEMENTATION_SUMMARY.md)
- [System Architecture Document](docs/SYSTEM_ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Operations Manual](docs/OPERATIONS_MANUAL.md)

### Development Guides

- [Development Setup](docs/DEVELOPMENT_SETUP.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)
- [Testing Guide](docs/TESTING_GUIDE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

We welcome contributions to the AI Options Trading System! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run quality checks
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- **Documentation** - Check the docs/ directory for comprehensive guides
- **Issues** - Report bugs and request features via GitHub Issues
- **Discussions** - Join community discussions for questions and ideas

### Common Issues

- **Service Startup** - Check Docker logs: `docker-compose logs [service-name]`
- **Database Connection** - Verify TimescaleDB is running and accessible
- **API Errors** - Check service health endpoints and logs
- **Performance** - Monitor Grafana dashboards for bottlenecks

### Health Check Commands

```bash
# Quick system health check
curl http://localhost:8000/health

# Detailed service status
docker-compose ps

# View service logs
docker-compose logs -f api-gateway
docker-compose logs -f data-ingestion
docker-compose logs -f analytics
```

## 🔮 Roadmap

### Current Status (Week 1 - Complete)

- ✅ Core infrastructure and microservices
- ✅ Data ingestion and analytics foundation
- ✅ Monitoring and observability
- ✅ Development and deployment automation

### Upcoming Features

- **Week 2** - Signal generation and trading algorithms
- **Week 3** - Portfolio management and position tracking
- **Week 4** - Risk management and safety controls
- **Week 5** - User interface and reporting dashboard

### Future Enhancements

- Machine learning integration for pattern recognition
- Advanced options strategies implementation
- Real-time streaming with Apache Kafka
- Mobile application for trading alerts
- Backtesting and strategy optimization tools

---

**Built with ❤️ by the AI-OTS Team**

For questions, suggestions, or support, please open an issue or contact the development team.

