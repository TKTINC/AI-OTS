-- Initial schema for AI Options Trading System
-- TimescaleDB optimized for time-series data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for trading data
CREATE SCHEMA IF NOT EXISTS trading;

-- Stock prices hypertable
CREATE TABLE IF NOT EXISTS trading.stock_prices (
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

-- Convert to hypertable (partition by time)
SELECT create_hypertable(
    'trading.stock_prices', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Options data hypertable
CREATE TABLE IF NOT EXISTS trading.options_data (
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

-- Convert to hypertable
SELECT create_hypertable(
    'trading.options_data', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading.signals (
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

-- Convert to hypertable
SELECT create_hypertable(
    'trading.signals', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Portfolio positions table
CREATE TABLE IF NOT EXISTS trading.positions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    position_type VARCHAR(10) NOT NULL CHECK (position_type IN ('STOCK', 'OPTION')),
    side VARCHAR(5) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity DECIMAL(15,6) NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4),
    unrealized_pnl DECIMAL(15,4),
    realized_pnl DECIMAL(15,4) DEFAULT 0,
    option_details JSONB, -- For option-specific data
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS trading.performance_metrics (
    id BIGSERIAL,
    user_id VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable(
    'trading.performance_metrics', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Market data quality table
CREATE TABLE IF NOT EXISTS trading.data_quality (
    id BIGSERIAL,
    source VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    quality_score DECIMAL(5,4) CHECK (quality_score >= 0 AND quality_score <= 1),
    issues JSONB,
    records_processed BIGINT DEFAULT 0,
    records_failed BIGINT DEFAULT 0,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable(
    'trading.data_quality', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- System events table for monitoring
CREATE TABLE IF NOT EXISTS trading.system_events (
    id BIGSERIAL,
    event_type VARCHAR(50) NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    details JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable(
    'trading.system_events', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION trading.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_stock_prices_updated_at 
    BEFORE UPDATE ON trading.stock_prices 
    FOR EACH ROW EXECUTE FUNCTION trading.update_updated_at_column();

CREATE TRIGGER update_options_data_updated_at 
    BEFORE UPDATE ON trading.options_data 
    FOR EACH ROW EXECUTE FUNCTION trading.update_updated_at_column();

CREATE TRIGGER update_signals_updated_at 
    BEFORE UPDATE ON trading.signals 
    FOR EACH ROW EXECUTE FUNCTION trading.update_updated_at_column();

CREATE TRIGGER update_positions_updated_at 
    BEFORE UPDATE ON trading.positions 
    FOR EACH ROW EXECUTE FUNCTION trading.update_updated_at_column();

-- Add comments for documentation
COMMENT ON SCHEMA trading IS 'Schema for AI Options Trading System data';
COMMENT ON TABLE trading.stock_prices IS 'Real-time and historical stock price data';
COMMENT ON TABLE trading.options_data IS 'Options chain data with Greeks and pricing';
COMMENT ON TABLE trading.signals IS 'AI-generated trading signals with confidence scores';
COMMENT ON TABLE trading.positions IS 'Portfolio positions and P&L tracking';
COMMENT ON TABLE trading.performance_metrics IS 'Performance analytics and KPIs';
COMMENT ON TABLE trading.data_quality IS 'Data quality monitoring and validation';
COMMENT ON TABLE trading.system_events IS 'System events and monitoring logs';

