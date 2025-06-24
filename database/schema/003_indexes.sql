-- Indexes for optimal query performance
-- Designed for common query patterns in the trading system

-- Stock prices indexes
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_time 
    ON trading.stock_prices (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_stock_prices_time 
    ON trading.stock_prices (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_volume 
    ON trading.stock_prices (symbol, volume DESC) 
    WHERE volume > 0;

CREATE INDEX IF NOT EXISTS idx_stock_prices_price_range 
    ON trading.stock_prices (symbol, close_price) 
    WHERE close_price IS NOT NULL;

-- Partial index for recent data (last 30 days)
CREATE INDEX IF NOT EXISTS idx_stock_prices_recent 
    ON trading.stock_prices (symbol, timestamp DESC, close_price) 
    WHERE timestamp >= NOW() - INTERVAL '30 days';

-- Options data indexes
CREATE INDEX IF NOT EXISTS idx_options_underlying_time 
    ON trading.options_data (underlying_symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_options_symbol_time 
    ON trading.options_data (option_symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_options_expiry_strike 
    ON trading.options_data (underlying_symbol, expiration_date, strike_price, option_type);

CREATE INDEX IF NOT EXISTS idx_options_volume 
    ON trading.options_data (underlying_symbol, volume DESC) 
    WHERE volume > 0;

CREATE INDEX IF NOT EXISTS idx_options_iv 
    ON trading.options_data (underlying_symbol, implied_volatility) 
    WHERE implied_volatility IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_options_greeks 
    ON trading.options_data (underlying_symbol, delta, gamma, theta, vega) 
    WHERE delta IS NOT NULL;

-- Partial index for active options (not expired)
CREATE INDEX IF NOT EXISTS idx_options_active 
    ON trading.options_data (underlying_symbol, option_type, strike_price, timestamp DESC) 
    WHERE expiration_date >= CURRENT_DATE;

-- Partial index for near-the-money options
CREATE INDEX IF NOT EXISTS idx_options_ntm 
    ON trading.options_data (underlying_symbol, timestamp DESC, strike_price) 
    WHERE abs(delta) BETWEEN 0.3 AND 0.7;

-- Trading signals indexes
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
    ON trading.signals (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_signals_type_confidence 
    ON trading.signals (signal_type, confidence DESC);

CREATE INDEX IF NOT EXISTS idx_signals_active 
    ON trading.signals (symbol, timestamp DESC) 
    WHERE expires_at IS NULL OR expires_at > NOW();

CREATE INDEX IF NOT EXISTS idx_signals_model_version 
    ON trading.signals (model_version, timestamp DESC);

-- GIN index for features JSONB column
CREATE INDEX IF NOT EXISTS idx_signals_features 
    ON trading.signals USING GIN (features);

-- Portfolio positions indexes
CREATE INDEX IF NOT EXISTS idx_positions_user_symbol 
    ON trading.positions (user_id, symbol);

CREATE INDEX IF NOT EXISTS idx_positions_open 
    ON trading.positions (user_id, symbol, opened_at DESC) 
    WHERE closed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_positions_pnl 
    ON trading.positions (user_id, unrealized_pnl DESC) 
    WHERE closed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_positions_type 
    ON trading.positions (position_type, side, opened_at DESC);

-- GIN index for option_details JSONB column
CREATE INDEX IF NOT EXISTS idx_positions_option_details 
    ON trading.positions USING GIN (option_details) 
    WHERE position_type = 'OPTION';

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_user_metric 
    ON trading.performance_metrics (user_id, metric_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_performance_period 
    ON trading.performance_metrics (user_id, period_start, period_end);

-- GIN index for metadata JSONB column
CREATE INDEX IF NOT EXISTS idx_performance_metadata 
    ON trading.performance_metrics USING GIN (metadata);

-- Data quality indexes
CREATE INDEX IF NOT EXISTS idx_data_quality_source_symbol 
    ON trading.data_quality (source, symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_data_quality_score 
    ON trading.data_quality (quality_score DESC, timestamp DESC) 
    WHERE quality_score < 0.95;

CREATE INDEX IF NOT EXISTS idx_data_quality_issues 
    ON trading.data_quality (source, timestamp DESC) 
    WHERE issues IS NOT NULL;

-- GIN index for issues JSONB column
CREATE INDEX IF NOT EXISTS idx_data_quality_issues_gin 
    ON trading.data_quality USING GIN (issues);

-- System events indexes
CREATE INDEX IF NOT EXISTS idx_system_events_service_time 
    ON trading.system_events (service_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_events_severity 
    ON trading.system_events (severity, timestamp DESC) 
    WHERE severity IN ('ERROR', 'CRITICAL');

CREATE INDEX IF NOT EXISTS idx_system_events_type 
    ON trading.system_events (event_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_events_unresolved 
    ON trading.system_events (service_name, timestamp DESC) 
    WHERE resolved_at IS NULL AND severity IN ('ERROR', 'CRITICAL');

-- GIN index for details JSONB column
CREATE INDEX IF NOT EXISTS idx_system_events_details 
    ON trading.system_events USING GIN (details);

-- Composite indexes for complex queries

-- Stock price momentum analysis
CREATE INDEX IF NOT EXISTS idx_stock_momentum 
    ON trading.stock_prices (symbol, timestamp DESC, close_price, volume) 
    WHERE volume > 1000000;

-- Options flow analysis
CREATE INDEX IF NOT EXISTS idx_options_flow 
    ON trading.options_data (underlying_symbol, timestamp DESC, volume, open_interest) 
    WHERE volume > 100;

-- Signal effectiveness analysis
CREATE INDEX IF NOT EXISTS idx_signal_effectiveness 
    ON trading.signals (symbol, signal_type, confidence DESC, timestamp DESC) 
    WHERE confidence > 0.7;

-- Portfolio performance analysis
CREATE INDEX IF NOT EXISTS idx_portfolio_performance 
    ON trading.positions (user_id, opened_at DESC, realized_pnl, unrealized_pnl) 
    WHERE realized_pnl IS NOT NULL OR unrealized_pnl IS NOT NULL;

-- Unique constraints for data integrity
ALTER TABLE trading.stock_prices 
    ADD CONSTRAINT uk_stock_prices_symbol_timestamp 
    UNIQUE (symbol, timestamp);

ALTER TABLE trading.options_data 
    ADD CONSTRAINT uk_options_data_symbol_timestamp 
    UNIQUE (option_symbol, timestamp);

-- Check constraints for data validation
ALTER TABLE trading.stock_prices 
    ADD CONSTRAINT chk_stock_prices_positive_volume 
    CHECK (volume IS NULL OR volume >= 0);

ALTER TABLE trading.stock_prices 
    ADD CONSTRAINT chk_stock_prices_positive_prices 
    CHECK (
        (open_price IS NULL OR open_price > 0) AND
        (high_price IS NULL OR high_price > 0) AND
        (low_price IS NULL OR low_price > 0) AND
        (close_price IS NULL OR close_price > 0)
    );

ALTER TABLE trading.stock_prices 
    ADD CONSTRAINT chk_stock_prices_high_low 
    CHECK (
        high_price IS NULL OR low_price IS NULL OR 
        high_price >= low_price
    );

ALTER TABLE trading.options_data 
    ADD CONSTRAINT chk_options_positive_prices 
    CHECK (
        (bid_price IS NULL OR bid_price >= 0) AND
        (ask_price IS NULL OR ask_price >= 0) AND
        (last_price IS NULL OR last_price >= 0) AND
        (strike_price > 0)
    );

ALTER TABLE trading.options_data 
    ADD CONSTRAINT chk_options_bid_ask 
    CHECK (
        bid_price IS NULL OR ask_price IS NULL OR 
        ask_price >= bid_price
    );

ALTER TABLE trading.options_data 
    ADD CONSTRAINT chk_options_expiry_future 
    CHECK (expiration_date >= CURRENT_DATE);

ALTER TABLE trading.positions 
    ADD CONSTRAINT chk_positions_quantity_nonzero 
    CHECK (quantity != 0);

ALTER TABLE trading.positions 
    ADD CONSTRAINT chk_positions_positive_entry_price 
    CHECK (entry_price > 0);

-- Statistics for query optimization
-- Update statistics on all tables
ANALYZE trading.stock_prices;
ANALYZE trading.options_data;
ANALYZE trading.signals;
ANALYZE trading.positions;
ANALYZE trading.performance_metrics;
ANALYZE trading.data_quality;
ANALYZE trading.system_events;

-- Create statistics on commonly filtered columns
CREATE STATISTICS IF NOT EXISTS stat_stock_prices_symbol_time 
    ON symbol, timestamp 
    FROM trading.stock_prices;

CREATE STATISTICS IF NOT EXISTS stat_options_underlying_expiry 
    ON underlying_symbol, expiration_date 
    FROM trading.options_data;

CREATE STATISTICS IF NOT EXISTS stat_signals_symbol_type 
    ON symbol, signal_type 
    FROM trading.signals;

-- Comments for documentation
COMMENT ON INDEX idx_stock_prices_symbol_time IS 'Primary index for stock price queries by symbol and time';
COMMENT ON INDEX idx_options_underlying_time IS 'Primary index for options data queries by underlying and time';
COMMENT ON INDEX idx_signals_symbol_time IS 'Primary index for signals queries by symbol and time';
COMMENT ON INDEX idx_positions_user_symbol IS 'Primary index for position queries by user and symbol';
COMMENT ON INDEX idx_performance_user_metric IS 'Primary index for performance metrics by user and metric type';
COMMENT ON INDEX idx_data_quality_source_symbol IS 'Primary index for data quality monitoring';
COMMENT ON INDEX idx_system_events_service_time IS 'Primary index for system event monitoring';

