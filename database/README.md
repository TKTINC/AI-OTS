# Database Documentation - AI Options Trading System

## Overview

The AI Options Trading System uses **TimescaleDB** (PostgreSQL extension) for high-performance time-series data storage and analysis. The database is optimized for handling real-time market data, options chains, trading signals, and performance analytics.

## Architecture

### Schema Structure

```
trading/
├── stock_prices          # Real-time stock price data (hypertable)
├── options_data          # Options chain data with Greeks (hypertable)
├── signals               # AI-generated trading signals (hypertable)
├── positions             # Portfolio positions and P&L tracking
├── performance_metrics   # Performance analytics (hypertable)
├── data_quality          # Data quality monitoring (hypertable)
└── system_events         # System events and logs (hypertable)
```

### Hypertables

TimescaleDB hypertables provide automatic partitioning and optimization for time-series data:

- **stock_prices**: Partitioned by 1-day chunks
- **options_data**: Partitioned by 1-day chunks
- **signals**: Partitioned by 1-day chunks
- **performance_metrics**: Partitioned by 1-week chunks
- **data_quality**: Partitioned by 1-day chunks
- **system_events**: Partitioned by 1-day chunks

### Continuous Aggregates

Pre-computed aggregations for common queries:

- **stock_prices_1m**: 1-minute OHLCV data
- **stock_prices_5m**: 5-minute OHLCV data
- **stock_prices_1h**: 1-hour OHLCV data
- **stock_prices_1d**: Daily OHLCV data
- **options_volume_1h**: Hourly options volume and IV
- **signal_performance_1d**: Daily signal performance metrics

## Setup and Migration

### Prerequisites

```bash
# Install PostgreSQL and TimescaleDB
sudo apt-get update
sudo apt-get install postgresql-15 postgresql-client-15
sudo apt-get install timescaledb-2-postgresql-15

# Configure TimescaleDB
sudo timescaledb-tune
sudo systemctl restart postgresql
```

### Database Creation

```bash
# Create database
sudo -u postgres createdb trading_db

# Create user
sudo -u postgres psql -c "CREATE USER trading_admin WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_admin;"
```

### Running Migrations

```bash
# Set environment variables
export DATABASE_URL="postgresql://trading_admin:password@localhost:5432/trading_db"

# Run all migrations
python3 database/migrations/migrate.py --database-url "$DATABASE_URL" --action migrate

# Check migration status
python3 database/migrations/migrate.py --database-url "$DATABASE_URL" --action status

# Verify database health
python3 database/migrations/migrate.py --database-url "$DATABASE_URL" --action health
```

## Data Models

### Stock Prices

```sql
CREATE TABLE trading.stock_prices (
    id BIGSERIAL,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    vwap DECIMAL(12,4),
    market_cap BIGINT,
    shares_outstanding BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);
```

### Options Data

```sql
CREATE TABLE trading.options_data (
    id BIGSERIAL,
    underlying_symbol VARCHAR(10) NOT NULL,
    option_symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    expiration_date DATE NOT NULL,
    strike_price DECIMAL(12,4) NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
    bid_price DECIMAL(12,4),
    ask_price DECIMAL(12,4),
    last_price DECIMAL(12,4),
    volume BIGINT DEFAULT 0,
    open_interest BIGINT DEFAULT 0,
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    intrinsic_value DECIMAL(12,4),
    time_value DECIMAL(12,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);
```

### Trading Signals

```sql
CREATE TABLE trading.signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    target_price DECIMAL(12,4),
    stop_loss DECIMAL(12,4),
    reasoning TEXT,
    model_version VARCHAR(50),
    features JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Query Examples

### Real-time Stock Prices

```sql
-- Get latest prices for all symbols
SELECT DISTINCT ON (symbol) 
    symbol, timestamp, close_price, volume
FROM trading.stock_prices 
ORDER BY symbol, timestamp DESC;

-- Get 5-minute OHLCV data for AAPL
SELECT * FROM trading.stock_prices_5m 
WHERE symbol = 'AAPL' 
AND bucket >= NOW() - INTERVAL '1 day'
ORDER BY bucket DESC;
```

### Options Analysis

```sql
-- Get options chain for AAPL expiring this week
SELECT 
    option_symbol, strike_price, option_type,
    last_price, volume, open_interest, implied_volatility
FROM trading.options_data 
WHERE underlying_symbol = 'AAPL'
AND expiration_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
AND timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY option_type, strike_price;

-- Calculate put/call ratio
SELECT 
    underlying_symbol,
    SUM(CASE WHEN option_type = 'PUT' THEN volume ELSE 0 END) as put_volume,
    SUM(CASE WHEN option_type = 'CALL' THEN volume ELSE 0 END) as call_volume,
    CASE 
        WHEN SUM(CASE WHEN option_type = 'CALL' THEN volume ELSE 0 END) > 0 
        THEN SUM(CASE WHEN option_type = 'PUT' THEN volume ELSE 0 END)::DECIMAL / 
             SUM(CASE WHEN option_type = 'CALL' THEN volume ELSE 0 END)::DECIMAL
        ELSE NULL 
    END as put_call_ratio
FROM trading.options_data 
WHERE timestamp >= NOW() - INTERVAL '1 day'
GROUP BY underlying_symbol;
```

### Technical Analysis

```sql
-- Calculate RSI for AAPL
SELECT * FROM trading.calculate_rsi('AAPL', 14, '5 minutes')
WHERE timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- Get Bollinger Bands
SELECT * FROM trading.calculate_bollinger_bands('AAPL', 20, 2.0, '5 minutes')
WHERE timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- Moving averages
SELECT * FROM trading.calculate_moving_average('AAPL', 50, '5 minutes')
WHERE timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;
```

### Signal Analysis

```sql
-- Get recent high-confidence signals
SELECT 
    symbol, signal_type, confidence, target_price, reasoning, timestamp
FROM trading.signals 
WHERE confidence > 0.8
AND timestamp >= NOW() - INTERVAL '1 day'
ORDER BY timestamp DESC;

-- Signal performance by symbol
SELECT 
    symbol,
    COUNT(*) as total_signals,
    AVG(confidence) as avg_confidence,
    COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END) as buy_signals,
    COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END) as sell_signals,
    COUNT(CASE WHEN signal_type = 'HOLD' THEN 1 END) as hold_signals
FROM trading.signals 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY symbol
ORDER BY total_signals DESC;
```

## Performance Optimization

### Indexing Strategy

The database includes optimized indexes for common query patterns:

- **Time-based queries**: Indexes on timestamp columns
- **Symbol lookups**: Composite indexes on (symbol, timestamp)
- **Volume analysis**: Partial indexes for high-volume data
- **Options filtering**: Indexes on strike price, expiration, Greeks
- **JSONB data**: GIN indexes for features and metadata

### Compression

TimescaleDB compression is enabled for data older than 7 days:

```sql
-- Check compression status
SELECT 
    hypertable_name,
    compression_enabled,
    compressed_chunks,
    uncompressed_chunks
FROM timescaledb_information.hypertables 
WHERE schema_name = 'trading';
```

### Data Retention

Automatic data retention policies:

- **Raw tick data**: 90 days
- **Options data**: 60 days
- **Signals**: 1 year
- **Performance metrics**: 2 years
- **System events**: 3 months (critical events: 2 years)

## Backup and Recovery

### Automated Backups

```bash
# Full backup
./database/scripts/backup.sh full

# Schema-only backup
./database/scripts/backup.sh schema

# Data backup for specific tables
./database/scripts/backup.sh data trading.stock_prices trading.signals

# Upload to S3
S3_BUCKET=my-backup-bucket ./database/scripts/backup.sh full
```

### Restore Operations

```bash
# Full restore
./database/scripts/restore.sh full /path/to/backup.sql

# Restore to different database
./database/scripts/restore.sh full /path/to/backup.sql trading_test

# Schema-only restore
./database/scripts/restore.sh schema /path/to/schema.sql

# Clone database
./database/scripts/restore.sh clone trading_db trading_test
```

### Point-in-Time Recovery

For production environments, configure continuous WAL archiving:

```bash
# In postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

## Monitoring

### Database Health Checks

```sql
-- Check hypertable health
SELECT * FROM timescaledb_information.hypertables;

-- Monitor chunk compression
SELECT 
    chunk_schema, chunk_name, compression_status,
    before_compression_total_bytes, after_compression_total_bytes
FROM timescaledb_information.chunks 
WHERE compression_status = 'Compressed';

-- Check continuous aggregate status
SELECT * FROM timescaledb_information.continuous_aggregates;
```

### Performance Monitoring

```sql
-- Query performance
SELECT 
    query, calls, total_time, mean_time, rows
FROM pg_stat_statements 
WHERE query LIKE '%trading.%'
ORDER BY total_time DESC;

-- Table statistics
SELECT 
    schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del,
    n_live_tup, n_dead_tup, last_vacuum, last_autovacuum
FROM pg_stat_user_tables 
WHERE schemaname = 'trading';
```

## Troubleshooting

### Common Issues

1. **Slow queries**: Check indexes and query plans
2. **High memory usage**: Tune shared_buffers and work_mem
3. **Compression issues**: Monitor chunk compression status
4. **Connection limits**: Adjust max_connections and connection pooling

### Useful Commands

```bash
# Check database size
SELECT pg_size_pretty(pg_database_size('trading_db'));

# Check table sizes
SELECT 
    schemaname, tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'trading'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'trading_db';
```

## Environment Variables

```bash
# Database connection
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=trading_db
export DB_USER=trading_admin
export DB_PASSWORD=your_password
export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

# Backup configuration
export BACKUP_DIR=/var/backups/ai-ots
export RETENTION_DAYS=30
export S3_BUCKET=my-backup-bucket
export S3_PREFIX=backups/ai-ots
```

## Security

### Access Control

```sql
-- Create read-only user for analytics
CREATE USER analytics_user WITH PASSWORD 'analytics_password';
GRANT CONNECT ON DATABASE trading_db TO analytics_user;
GRANT USAGE ON SCHEMA trading TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA trading TO analytics_user;

-- Create application user with limited permissions
CREATE USER app_user WITH PASSWORD 'app_password';
GRANT CONNECT ON DATABASE trading_db TO app_user;
GRANT USAGE ON SCHEMA trading TO app_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA trading TO app_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA trading TO app_user;
```

### SSL Configuration

```bash
# In postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
ssl_ca_file = 'ca.crt'
```

## Contact and Support

For database-related issues:

1. Check the logs: `/var/log/postgresql/`
2. Review query performance: `pg_stat_statements`
3. Monitor system resources: CPU, memory, disk I/O
4. Consult TimescaleDB documentation: https://docs.timescale.com/

---

*This documentation is part of the AI Options Trading System Week 1 implementation.*

