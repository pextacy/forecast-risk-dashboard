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
    coingecko_api_key: Optional[str] = os.getenv("COINGECKO_API_KEY")
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    yahoo_finance_api_key: Optional[str] = os.getenv("YAHOO_FINANCE_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Application settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_rate_limit: int = int(os.getenv("API_RATE_LIMIT", "100"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    # CoinGecko configuration
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    coingecko_pro_url: str = "https://pro-api.coingecko.com/api/v3"

    # Alpha Vantage configuration
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"

    # Supported assets for real data fetching
    supported_crypto_symbols: List[str] = [
        "bitcoin", "ethereum", "tether", "usd-coin", "binancecoin",
        "solana", "cardano", "polygon", "avalanche-2", "chainlink"
    ]

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

    if service == "coingecko" and settings.coingecko_api_key:
        headers["x-cg-pro-api-key"] = settings.coingecko_api_key

    elif service == "alpha_vantage":
        # Alpha Vantage uses API key as query parameter
        pass

    return headers


def get_coingecko_url() -> str:
    """Get appropriate CoinGecko URL based on API key availability."""
    if settings.coingecko_api_key:
        return settings.coingecko_pro_url
    return settings.coingecko_base_url


def validate_environment():
    """Validate required environment variables."""
    required_for_production = [
        "DATABASE_URL",
        "COINGECKO_API_KEY"
    ]

    if settings.environment == "production":
        missing = [var for var in required_for_production if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Validate database URL format
    if not settings.database_url.startswith("postgresql://"):
        raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string")

    return True