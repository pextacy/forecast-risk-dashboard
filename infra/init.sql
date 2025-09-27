-- Treasury Dashboard Database Schema with TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Asset prices table for time-series data
CREATE TABLE asset_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    volume BIGINT,
    market_cap BIGINT,
    source VARCHAR(50) NOT NULL DEFAULT 'coingecko',
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('asset_prices', 'time');

-- Portfolio balances
CREATE TABLE portfolio_balances (
    time TIMESTAMPTZ NOT NULL,
    wallet_address VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    balance DECIMAL(30,18) NOT NULL,
    usd_value DECIMAL(20,8),
    source VARCHAR(50) NOT NULL DEFAULT 'manual',
    PRIMARY KEY (time, symbol, wallet_address)
);

SELECT create_hypertable('portfolio_balances', 'time');

-- Forecasts table
CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    forecast_type VARCHAR(50) NOT NULL, -- 'price', 'volatility', 'portfolio_value'
    forecast_horizon INTEGER NOT NULL, -- days
    model_name VARCHAR(100) NOT NULL,
    forecast_data JSONB NOT NULL, -- {dates: [], values: [], confidence_intervals: []}
    model_metrics JSONB, -- {rmse: 0.1, mae: 0.05, r_squared: 0.85}
    accuracy_score DECIMAL(5,4)
);

-- Risk metrics table
CREATE TABLE risk_metrics (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id VARCHAR(100) DEFAULT 'default',
    var_95 DECIMAL(20,8), -- Value at Risk 95%
    var_99 DECIMAL(20,8), -- Value at Risk 99%
    expected_shortfall DECIMAL(20,8),
    sharpe_ratio DECIMAL(10,6),
    volatility_30d DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    beta DECIMAL(10,6),
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable('risk_metrics', 'time');

-- Hedge suggestions table
CREATE TABLE hedge_suggestions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    portfolio_id VARCHAR(100) DEFAULT 'default',
    risk_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high'
    suggestion_type VARCHAR(50) NOT NULL, -- 'rebalance', 'hedge', 'reduce_exposure'
    current_allocation JSONB NOT NULL,
    suggested_allocation JSONB NOT NULL,
    rationale TEXT NOT NULL,
    expected_risk_reduction DECIMAL(5,4),
    implementation_cost DECIMAL(10,6),
    confidence_score DECIMAL(5,4)
);

-- Backtest results table
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    actual_vs_predicted JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    rolling_accuracy JSONB
);

-- Indexes for performance
CREATE INDEX idx_asset_prices_symbol_time ON asset_prices (symbol, time DESC);
CREATE INDEX idx_portfolio_balances_time ON portfolio_balances (time DESC);
CREATE INDEX idx_forecasts_symbol_created ON forecasts (symbol, created_at DESC);
CREATE INDEX idx_risk_metrics_time ON risk_metrics (time DESC);
CREATE INDEX idx_hedge_suggestions_created ON hedge_suggestions (created_at DESC);

-- Insert sample supported assets
INSERT INTO asset_prices (time, symbol, price, volume, market_cap, source) VALUES
(NOW(), 'BTC', 43500.00, 25000000000, 850000000000, 'initial'),
(NOW(), 'ETH', 2650.00, 15000000000, 320000000000, 'initial'),
(NOW(), 'USDC', 1.00, 5000000000, 32000000000, 'initial'),
(NOW(), 'USDT', 1.00, 45000000000, 83000000000, 'initial');