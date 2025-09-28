"""Application configuration management."""

import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation."""

    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://treasury_user:treasury_pass_2024@localhost:5432/treasury_dashboard")

    # API Keys
    dexscreener_api_key: Optional[str] = os.getenv("DEXSCREENER_API_KEY")
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    yahoo_finance_api_key: Optional[str] = os.getenv("YAHOO_FINANCE_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")

    # Application settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_rate_limit: int = int(os.getenv("API_RATE_LIMIT", "100"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    # DexScreener configuration
    dexscreener_base_url: str = "https://api.dexscreener.com/latest"
    dexscreener_pairs_url: str = "https://api.dexscreener.com/latest/dex"

    # Alpha Vantage configuration
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"

    # Supported assets for real data fetching (DexScreener token addresses)
    supported_crypto_symbols: List[str] = [
        "ethereum", "bitcoin", "solana", "polygon", "avalanche",
        "binancecoin", "cardano", "chainlink", "uniswap", "aave"
    ]

    # Popular token addresses for DexScreener (REAL CONTRACT ADDRESSES)
    dex_token_addresses: dict = {
        "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        "bitcoin": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",   # WBTC
        "usdc": "0xA0b86a33E6441c7C677AB43c16f0dB1CCa34884e",      # USDC (Ethereum mainnet)
        "usdt": "0xdAC17F958D2ee523a2206206994597C13D831ec7",      # USDT
        "uniswap": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",   # UNI
        "chainlink": "0x514910771AF9Ca656af840dff83E8264EcF986CA",  # LINK
        "aave": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9"       # AAVE
    }

    supported_stock_symbols: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JNJ", "V"
    ]

    # Forecasting model parameters
    forecast_horizon_days: int = 30
    confidence_intervals: List[float] = [0.80, 0.90, 0.95]
    min_historical_days: int = 90

    # Risk management
    var_confidence_levels: List[float] = [0.95, 0.99]
    portfolio_rebalance_threshold: float = 0.05  # 5%
    max_single_asset_weight: float = 0.40  # 40%

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_api_headers(service: str) -> dict:
    """Get API headers for different services."""
    headers = {"User-Agent": "TreasuryDashboard/1.0"}

    if service == "dexscreener" and settings.dexscreener_api_key:
        headers["X-API-KEY"] = settings.dexscreener_api_key

    elif service == "alpha_vantage":
        # Alpha Vantage uses API key as query parameter
        pass

    elif service == "groq" and settings.groq_api_key:
        headers["Authorization"] = f"Bearer {settings.groq_api_key}"
        headers["Content-Type"] = "application/json"

    return headers


def get_dexscreener_url() -> str:
    """Get DexScreener API URL."""
    return settings.dexscreener_base_url


def validate_environment():
    """Validate required environment variables."""
    required_for_production = [
        "DATABASE_URL",
        "GROQ_API_KEY"
    ]

    if settings.environment == "production":
        missing = [var for var in required_for_production if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Validate database URL format
    if not settings.database_url.startswith("postgresql://"):
        raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string")

    return True