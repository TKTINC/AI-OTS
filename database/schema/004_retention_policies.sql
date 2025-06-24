-- Data retention policies for TimescaleDB
-- Automatically manage data lifecycle and storage costs

-- Retention policy for stock prices
-- Keep raw tick data for 90 days, then rely on continuous aggregates
SELECT add_retention_policy(
    'trading.stock_prices', 
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Retention policy for options data
-- Keep raw options data for 60 days (options expire quickly)
SELECT add_retention_policy(
    'trading.options_data', 
    INTERVAL '60 days',
    if_not_exists => TRUE
);

-- Retention policy for signals
-- Keep signals for 1 year for backtesting and model improvement
SELECT add_retention_policy(
    'trading.signals', 
    INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Retention policy for performance metrics
-- Keep performance data for 2 years for long-term analysis
SELECT add_retention_policy(
    'trading.performance_metrics', 
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- Retention policy for data quality
-- Keep data quality metrics for 6 months
SELECT add_retention_policy(
    'trading.data_quality', 
    INTERVAL '6 months',
    if_not_exists => TRUE
);

-- Retention policy for system events
-- Keep system events for 3 months (except critical events)
SELECT add_retention_policy(
    'trading.system_events', 
    INTERVAL '3 months',
    if_not_exists => TRUE
);

-- Create a separate table for critical events with longer retention
CREATE TABLE IF NOT EXISTS trading.critical_events (
    LIKE trading.system_events INCLUDING ALL
);

-- Convert to hypertable
SELECT create_hypertable(
    'trading.critical_events', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Retention policy for critical events (2 years)
SELECT add_retention_policy(
    'trading.critical_events', 
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- Create trigger to automatically copy critical events
CREATE OR REPLACE FUNCTION trading.copy_critical_events()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.severity IN ('ERROR', 'CRITICAL') THEN
        INSERT INTO trading.critical_events 
        SELECT NEW.*;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_copy_critical_events
    AFTER INSERT ON trading.system_events
    FOR EACH ROW
    EXECUTE FUNCTION trading.copy_critical_events();

-- Retention policies for continuous aggregates
-- Keep 1-minute aggregates for 1 year
SELECT add_retention_policy(
    'trading.stock_prices_1m', 
    INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Keep 5-minute aggregates for 2 years
SELECT add_retention_policy(
    'trading.stock_prices_5m', 
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- Keep 1-hour aggregates for 5 years
SELECT add_retention_policy(
    'trading.stock_prices_1h', 
    INTERVAL '5 years',
    if_not_exists => TRUE
);

-- Keep daily aggregates forever (no retention policy)
-- This provides long-term historical data for backtesting

-- Keep options volume aggregates for 1 year
SELECT add_retention_policy(
    'trading.options_volume_1h', 
    INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Keep signal performance aggregates for 3 years
SELECT add_retention_policy(
    'trading.signal_performance_1d', 
    INTERVAL '3 years',
    if_not_exists => TRUE
);

-- Create archival tables for long-term storage
-- These tables will store compressed historical data

CREATE TABLE IF NOT EXISTS trading.stock_prices_archive (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    avg_vwap DECIMAL(12,4),
    tick_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS trading.options_summary_archive (
    underlying_symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    total_call_volume BIGINT,
    total_put_volume BIGINT,
    avg_call_iv DECIMAL(8,6),
    avg_put_iv DECIMAL(8,6),
    put_call_ratio DECIMAL(8,6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (underlying_symbol, date)
);

-- Create functions for manual archival
CREATE OR REPLACE FUNCTION trading.archive_stock_data(p_date DATE)
RETURNS INTEGER AS $$
DECLARE
    rows_archived INTEGER;
BEGIN
    -- Archive daily stock data
    INSERT INTO trading.stock_prices_archive (
        symbol, date, open_price, high_price, low_price, 
        close_price, volume, avg_vwap, tick_count
    )
    SELECT 
        symbol,
        p_date,
        first(open_price, timestamp) AS open_price,
        max(high_price) AS high_price,
        min(low_price) AS low_price,
        last(close_price, timestamp) AS close_price,
        sum(volume) AS volume,
        avg(vwap) AS avg_vwap,
        count(*) AS tick_count
    FROM trading.stock_prices
    WHERE DATE(timestamp) = p_date
    GROUP BY symbol
    ON CONFLICT (symbol, date) DO UPDATE SET
        open_price = EXCLUDED.open_price,
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        volume = EXCLUDED.volume,
        avg_vwap = EXCLUDED.avg_vwap,
        tick_count = EXCLUDED.tick_count;
    
    GET DIAGNOSTICS rows_archived = ROW_COUNT;
    RETURN rows_archived;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trading.archive_options_data(p_date DATE)
RETURNS INTEGER AS $$
DECLARE
    rows_archived INTEGER;
BEGIN
    -- Archive daily options summary
    INSERT INTO trading.options_summary_archive (
        underlying_symbol, date, total_call_volume, total_put_volume,
        avg_call_iv, avg_put_iv, put_call_ratio
    )
    SELECT 
        underlying_symbol,
        p_date,
        sum(CASE WHEN option_type = 'CALL' THEN volume ELSE 0 END) AS total_call_volume,
        sum(CASE WHEN option_type = 'PUT' THEN volume ELSE 0 END) AS total_put_volume,
        avg(CASE WHEN option_type = 'CALL' THEN implied_volatility END) AS avg_call_iv,
        avg(CASE WHEN option_type = 'PUT' THEN implied_volatility END) AS avg_put_iv,
        CASE 
            WHEN sum(CASE WHEN option_type = 'CALL' THEN volume ELSE 0 END) > 0 
            THEN sum(CASE WHEN option_type = 'PUT' THEN volume ELSE 0 END)::DECIMAL / 
                 sum(CASE WHEN option_type = 'CALL' THEN volume ELSE 0 END)::DECIMAL
            ELSE NULL 
        END AS put_call_ratio
    FROM trading.options_data
    WHERE DATE(timestamp) = p_date
    GROUP BY underlying_symbol
    ON CONFLICT (underlying_symbol, date) DO UPDATE SET
        total_call_volume = EXCLUDED.total_call_volume,
        total_put_volume = EXCLUDED.total_put_volume,
        avg_call_iv = EXCLUDED.avg_call_iv,
        avg_put_iv = EXCLUDED.avg_put_iv,
        put_call_ratio = EXCLUDED.put_call_ratio;
    
    GET DIAGNOSTICS rows_archived = ROW_COUNT;
    RETURN rows_archived;
END;
$$ LANGUAGE plpgsql;

-- Create automated archival job (to be called by cron or scheduler)
CREATE OR REPLACE FUNCTION trading.daily_archival()
RETURNS TEXT AS $$
DECLARE
    archive_date DATE;
    stock_rows INTEGER;
    options_rows INTEGER;
    result_text TEXT;
BEGIN
    -- Archive data from 2 days ago (ensure all data is collected)
    archive_date := CURRENT_DATE - INTERVAL '2 days';
    
    -- Archive stock data
    SELECT trading.archive_stock_data(archive_date) INTO stock_rows;
    
    -- Archive options data
    SELECT trading.archive_options_data(archive_date) INTO options_rows;
    
    -- Log the archival
    INSERT INTO trading.system_events (
        event_type, service_name, severity, message, details
    ) VALUES (
        'DATA_ARCHIVAL',
        'database',
        'INFO',
        'Daily data archival completed',
        jsonb_build_object(
            'date', archive_date,
            'stock_rows_archived', stock_rows,
            'options_rows_archived', options_rows
        )
    );
    
    result_text := format(
        'Archived data for %s: %s stock rows, %s options rows',
        archive_date, stock_rows, options_rows
    );
    
    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- Create indexes on archive tables
CREATE INDEX IF NOT EXISTS idx_stock_archive_symbol_date 
    ON trading.stock_prices_archive (symbol, date DESC);

CREATE INDEX IF NOT EXISTS idx_options_archive_symbol_date 
    ON trading.options_summary_archive (underlying_symbol, date DESC);

-- Create view for unified historical data access
CREATE OR REPLACE VIEW trading.stock_prices_historical AS
-- Recent data from hypertable
SELECT 
    symbol,
    DATE(timestamp) as date,
    first(open_price, timestamp) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, timestamp) AS close_price,
    sum(volume) AS volume,
    avg(vwap) AS avg_vwap,
    count(*) AS tick_count,
    'live' AS source
FROM trading.stock_prices
WHERE timestamp >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY symbol, DATE(timestamp)

UNION ALL

-- Archived data
SELECT 
    symbol,
    date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    avg_vwap,
    tick_count,
    'archive' AS source
FROM trading.stock_prices_archive
WHERE date < CURRENT_DATE - INTERVAL '90 days';

-- Comments for documentation
COMMENT ON FUNCTION trading.archive_stock_data IS 'Archive daily stock price data to compressed storage';
COMMENT ON FUNCTION trading.archive_options_data IS 'Archive daily options summary data';
COMMENT ON FUNCTION trading.daily_archival IS 'Automated daily archival process';
COMMENT ON VIEW trading.stock_prices_historical IS 'Unified view of live and archived stock price data';
COMMENT ON TABLE trading.stock_prices_archive IS 'Compressed archive of historical stock price data';
COMMENT ON TABLE trading.options_summary_archive IS 'Compressed archive of historical options summary data';
COMMENT ON TABLE trading.critical_events IS 'Long-term storage for critical system events';

