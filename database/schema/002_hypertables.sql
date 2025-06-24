-- Hypertable configuration and optimization for TimescaleDB
-- This script configures chunk intervals, compression, and retention policies

-- Configure compression for stock_prices
ALTER TABLE trading.stock_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Configure compression for options_data
ALTER TABLE trading.options_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'underlying_symbol, option_type',
    timescaledb.compress_orderby = 'timestamp DESC, strike_price'
);

-- Configure compression for signals
ALTER TABLE trading.signals SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, signal_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Configure compression for performance_metrics
ALTER TABLE trading.performance_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id, metric_name',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Configure compression for data_quality
ALTER TABLE trading.data_quality SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'source, symbol, data_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Configure compression for system_events
ALTER TABLE trading.system_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'service_name, event_type, severity',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policies (compress data older than 7 days)
SELECT add_compression_policy('trading.stock_prices', INTERVAL '7 days');
SELECT add_compression_policy('trading.options_data', INTERVAL '7 days');
SELECT add_compression_policy('trading.signals', INTERVAL '7 days');
SELECT add_compression_policy('trading.performance_metrics', INTERVAL '7 days');
SELECT add_compression_policy('trading.data_quality', INTERVAL '7 days');
SELECT add_compression_policy('trading.system_events', INTERVAL '7 days');

-- Create continuous aggregates for common queries

-- 1-minute OHLCV aggregates for stock prices
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.stock_prices_1m
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 minute', timestamp) AS bucket,
    first(open_price, timestamp) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, timestamp) AS close_price,
    sum(volume) AS volume,
    avg(vwap) AS avg_vwap,
    count(*) AS tick_count
FROM trading.stock_prices
GROUP BY symbol, bucket;

-- 5-minute OHLCV aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.stock_prices_5m
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('5 minutes', timestamp) AS bucket,
    first(open_price, timestamp) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, timestamp) AS close_price,
    sum(volume) AS volume,
    avg(vwap) AS avg_vwap,
    count(*) AS tick_count
FROM trading.stock_prices
GROUP BY symbol, bucket;

-- 1-hour OHLCV aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.stock_prices_1h
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 hour', timestamp) AS bucket,
    first(open_price, timestamp) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, timestamp) AS close_price,
    sum(volume) AS volume,
    avg(vwap) AS avg_vwap,
    count(*) AS tick_count
FROM trading.stock_prices
GROUP BY symbol, bucket;

-- Daily OHLCV aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.stock_prices_1d
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', timestamp) AS bucket,
    first(open_price, timestamp) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, timestamp) AS close_price,
    sum(volume) AS volume,
    avg(vwap) AS avg_vwap,
    count(*) AS tick_count
FROM trading.stock_prices
GROUP BY symbol, bucket;

-- Options volume and open interest aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.options_volume_1h
WITH (timescaledb.continuous) AS
SELECT 
    underlying_symbol,
    option_type,
    time_bucket('1 hour', timestamp) AS bucket,
    sum(volume) AS total_volume,
    avg(open_interest) AS avg_open_interest,
    avg(implied_volatility) AS avg_iv,
    count(*) AS contract_count
FROM trading.options_data
GROUP BY underlying_symbol, option_type, bucket;

-- Signal performance aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.signal_performance_1d
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    signal_type,
    time_bucket('1 day', timestamp) AS bucket,
    count(*) AS signal_count,
    avg(confidence) AS avg_confidence,
    min(confidence) AS min_confidence,
    max(confidence) AS max_confidence
FROM trading.signals
GROUP BY symbol, signal_type, bucket;

-- Add refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('trading.stock_prices_1m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

SELECT add_continuous_aggregate_policy('trading.stock_prices_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('trading.stock_prices_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('trading.stock_prices_1d',
    start_offset => INTERVAL '1 week',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

SELECT add_continuous_aggregate_policy('trading.options_volume_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('trading.signal_performance_1d',
    start_offset => INTERVAL '1 week',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Create functions for common calculations

-- Calculate RSI
CREATE OR REPLACE FUNCTION trading.calculate_rsi(
    p_symbol VARCHAR(10),
    p_period INTEGER DEFAULT 14,
    p_timeframe INTERVAL DEFAULT '5 minutes'
)
RETURNS TABLE(timestamp TIMESTAMPTZ, rsi DECIMAL(5,2)) AS $$
BEGIN
    RETURN QUERY
    WITH price_changes AS (
        SELECT 
            time_bucket(p_timeframe, sp.timestamp) AS bucket,
            last(close_price, sp.timestamp) AS close_price,
            lag(last(close_price, sp.timestamp)) OVER (ORDER BY time_bucket(p_timeframe, sp.timestamp)) AS prev_close
        FROM trading.stock_prices sp
        WHERE sp.symbol = p_symbol
        GROUP BY bucket
        ORDER BY bucket
    ),
    gains_losses AS (
        SELECT 
            bucket,
            CASE WHEN close_price > prev_close THEN close_price - prev_close ELSE 0 END AS gain,
            CASE WHEN close_price < prev_close THEN prev_close - close_price ELSE 0 END AS loss
        FROM price_changes
        WHERE prev_close IS NOT NULL
    ),
    avg_gains_losses AS (
        SELECT 
            bucket,
            avg(gain) OVER (ORDER BY bucket ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW) AS avg_gain,
            avg(loss) OVER (ORDER BY bucket ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW) AS avg_loss,
            row_number() OVER (ORDER BY bucket) AS rn
        FROM gains_losses
    )
    SELECT 
        bucket::TIMESTAMPTZ,
        CASE 
            WHEN avg_loss = 0 THEN 100::DECIMAL(5,2)
            ELSE (100 - (100 / (1 + (avg_gain / avg_loss))))::DECIMAL(5,2)
        END AS rsi
    FROM avg_gains_losses
    WHERE rn >= p_period;
END;
$$ LANGUAGE plpgsql;

-- Calculate moving averages
CREATE OR REPLACE FUNCTION trading.calculate_moving_average(
    p_symbol VARCHAR(10),
    p_period INTEGER,
    p_timeframe INTERVAL DEFAULT '5 minutes'
)
RETURNS TABLE(timestamp TIMESTAMPTZ, ma DECIMAL(12,4)) AS $$
BEGIN
    RETURN QUERY
    WITH bucketed_prices AS (
        SELECT 
            time_bucket(p_timeframe, sp.timestamp) AS bucket,
            last(close_price, sp.timestamp) AS close_price
        FROM trading.stock_prices sp
        WHERE sp.symbol = p_symbol
        GROUP BY bucket
        ORDER BY bucket
    )
    SELECT 
        bucket::TIMESTAMPTZ,
        avg(close_price) OVER (ORDER BY bucket ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW)::DECIMAL(12,4) AS ma
    FROM bucketed_prices;
END;
$$ LANGUAGE plpgsql;

-- Calculate Bollinger Bands
CREATE OR REPLACE FUNCTION trading.calculate_bollinger_bands(
    p_symbol VARCHAR(10),
    p_period INTEGER DEFAULT 20,
    p_std_dev DECIMAL DEFAULT 2.0,
    p_timeframe INTERVAL DEFAULT '5 minutes'
)
RETURNS TABLE(
    timestamp TIMESTAMPTZ, 
    middle_band DECIMAL(12,4),
    upper_band DECIMAL(12,4),
    lower_band DECIMAL(12,4)
) AS $$
BEGIN
    RETURN QUERY
    WITH bucketed_prices AS (
        SELECT 
            time_bucket(p_timeframe, sp.timestamp) AS bucket,
            last(close_price, sp.timestamp) AS close_price
        FROM trading.stock_prices sp
        WHERE sp.symbol = p_symbol
        GROUP BY bucket
        ORDER BY bucket
    ),
    bb_calc AS (
        SELECT 
            bucket,
            close_price,
            avg(close_price) OVER (ORDER BY bucket ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW) AS sma,
            stddev(close_price) OVER (ORDER BY bucket ROWS BETWEEN p_period-1 PRECEDING AND CURRENT ROW) AS std_dev
        FROM bucketed_prices
    )
    SELECT 
        bucket::TIMESTAMPTZ,
        sma::DECIMAL(12,4) AS middle_band,
        (sma + (p_std_dev * std_dev))::DECIMAL(12,4) AS upper_band,
        (sma - (p_std_dev * std_dev))::DECIMAL(12,4) AS lower_band
    FROM bb_calc
    WHERE std_dev IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Add comments
COMMENT ON MATERIALIZED VIEW trading.stock_prices_1m IS '1-minute OHLCV aggregates for stock prices';
COMMENT ON MATERIALIZED VIEW trading.stock_prices_5m IS '5-minute OHLCV aggregates for stock prices';
COMMENT ON MATERIALIZED VIEW trading.stock_prices_1h IS '1-hour OHLCV aggregates for stock prices';
COMMENT ON MATERIALIZED VIEW trading.stock_prices_1d IS 'Daily OHLCV aggregates for stock prices';
COMMENT ON MATERIALIZED VIEW trading.options_volume_1h IS 'Hourly options volume and IV aggregates';
COMMENT ON MATERIALIZED VIEW trading.signal_performance_1d IS 'Daily signal performance metrics';

COMMENT ON FUNCTION trading.calculate_rsi IS 'Calculate RSI indicator for a given symbol and timeframe';
COMMENT ON FUNCTION trading.calculate_moving_average IS 'Calculate simple moving average for a given symbol and timeframe';
COMMENT ON FUNCTION trading.calculate_bollinger_bands IS 'Calculate Bollinger Bands for a given symbol and timeframe';

