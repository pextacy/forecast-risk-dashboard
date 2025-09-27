"""Real-time data ingestion service for cryptocurrency and financial data."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
import yfinance as yf
from app.db.connection import db_manager
from app.utils.config import settings, get_api_headers, get_coingecko_url

logger = logging.getLogger(__name__)


class CoinGeckoClient:
    """Real CoinGecko API client for cryptocurrency data."""

    def __init__(self):
        self.base_url = get_coingecko_url()
        self.headers = get_api_headers("coingecko")
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_current_prices(self, coin_ids: List[str]) -> Dict[str, dict]:
        """Fetch current prices for multiple cryptocurrencies."""
        url = f"{self.base_url}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Fetched current prices for {len(data)} coins")
                return data
            else:
                logger.error(f"CoinGecko API error: {response.status}")
                raise Exception(f"CoinGecko API returned {response.status}")

    async def get_historical_data(self, coin_id: str, days: int = 30) -> List[dict]:
        """Fetch historical price data for a single cryptocurrency."""
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "hourly" if days <= 90 else "daily"
        }

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()

                # Convert to structured format
                prices = data.get("prices", [])
                volumes = data.get("total_volumes", [])
                market_caps = data.get("market_caps", [])

                historical_data = []
                for i, (timestamp, price) in enumerate(prices):
                    historical_data.append({
                        "time": datetime.fromtimestamp(timestamp / 1000),
                        "symbol": coin_id,
                        "price": price,
                        "volume": volumes[i][1] if i < len(volumes) else None,
                        "market_cap": market_caps[i][1] if i < len(market_caps) else None,
                        "source": "coingecko"
                    })

                logger.info(f"Fetched {len(historical_data)} historical records for {coin_id}")
                return historical_data
            else:
                logger.error(f"CoinGecko historical API error: {response.status}")
                raise Exception(f"CoinGecko API returned {response.status}")

    async def get_market_data(self, coin_ids: List[str]) -> List[dict]:
        """Fetch comprehensive market data including volatility metrics."""
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": len(coin_ids),
            "page": 1,
            "sparkline": "true",
            "price_change_percentage": "1h,24h,7d,30d"
        }

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Fetched market data for {len(data)} coins")
                return data
            else:
                logger.error(f"CoinGecko market API error: {response.status}")
                raise Exception(f"CoinGecko API returned {response.status}")


class YahooFinanceClient:
    """Yahoo Finance client for traditional financial assets."""

    def __init__(self):
        self.symbols = settings.supported_stock_symbols

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetch current prices for stock symbols."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            tickers = yf.Tickers(" ".join(symbols))

            current_data = {}
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = await loop.run_in_executor(None, lambda: ticker.info)
                    hist = await loop.run_in_executor(None, lambda: ticker.history(period="1d"))

                    if not hist.empty:
                        latest = hist.iloc[-1]
                        current_data[symbol] = {
                            "usd": latest["Close"],
                            "usd_market_cap": info.get("marketCap"),
                            "usd_24h_vol": latest["Volume"] * latest["Close"],
                            "last_updated_at": int(datetime.now().timestamp())
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue

            logger.info(f"Fetched Yahoo Finance data for {len(current_data)} symbols")
            return current_data

        except Exception as e:
            logger.error(f"Yahoo Finance API error: {e}")
            raise

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[dict]:
        """Fetch historical data for a stock symbol."""
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)

            # Calculate period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            hist = await loop.run_in_executor(
                None,
                lambda: ticker.history(start=start_date, end=end_date, interval="1h" if days <= 30 else "1d")
            )

            historical_data = []
            for timestamp, row in hist.iterrows():
                historical_data.append({
                    "time": timestamp.to_pydatetime(),
                    "symbol": symbol,
                    "price": row["Close"],
                    "volume": row["Volume"],
                    "market_cap": None,  # Not available in Yahoo Finance
                    "source": "yahoo_finance"
                })

            logger.info(f"Fetched {len(historical_data)} historical records for {symbol}")
            return historical_data

        except Exception as e:
            logger.error(f"Yahoo Finance historical data error for {symbol}: {e}")
            raise


class DataIngestionService:
    """Main data ingestion service coordinating multiple data sources."""

    def __init__(self):
        self.coingecko_client = CoinGeckoClient()
        self.yahoo_client = YahooFinanceClient()

    async def ingest_current_market_data(self) -> Tuple[int, int]:
        """Ingest current market data from all sources."""
        crypto_count = 0
        stock_count = 0

        try:
            # Fetch cryptocurrency data
            async with CoinGeckoClient() as client:
                crypto_prices = await client.get_current_prices(settings.supported_crypto_symbols)

                crypto_records = []
                for coin_id, price_data in crypto_prices.items():
                    crypto_records.append({
                        "time": datetime.fromtimestamp(price_data["last_updated_at"]),
                        "symbol": coin_id,
                        "price": price_data["usd"],
                        "volume": price_data.get("usd_24h_vol"),
                        "market_cap": price_data.get("usd_market_cap"),
                        "source": "coingecko"
                    })

                if crypto_records:
                    await db_manager.bulk_insert_prices(crypto_records)
                    crypto_count = len(crypto_records)
                    logger.info(f"Ingested {crypto_count} cryptocurrency price records")

        except Exception as e:
            logger.error(f"Failed to ingest crypto data: {e}")

        try:
            # Fetch stock data
            stock_prices = await self.yahoo_client.get_current_prices(settings.supported_stock_symbols)

            stock_records = []
            for symbol, price_data in stock_prices.items():
                stock_records.append({
                    "time": datetime.fromtimestamp(price_data["last_updated_at"]),
                    "symbol": symbol,
                    "price": price_data["usd"],
                    "volume": price_data.get("usd_24h_vol"),
                    "market_cap": price_data.get("usd_market_cap"),
                    "source": "yahoo_finance"
                })

            if stock_records:
                await db_manager.bulk_insert_prices(stock_records)
                stock_count = len(stock_records)
                logger.info(f"Ingested {stock_count} stock price records")

        except Exception as e:
            logger.error(f"Failed to ingest stock data: {e}")

        return crypto_count, stock_count

    async def ingest_historical_data(self, symbol: str, days: int = 90, source: str = "auto") -> int:
        """Ingest historical data for a specific symbol."""
        try:
            if source == "auto":
                # Determine source based on symbol
                if symbol in settings.supported_crypto_symbols:
                    source = "coingecko"
                elif symbol in settings.supported_stock_symbols:
                    source = "yahoo_finance"
                else:
                    raise ValueError(f"Unknown symbol: {symbol}")

            historical_data = []

            if source == "coingecko":
                async with CoinGeckoClient() as client:
                    historical_data = await client.get_historical_data(symbol, days)

            elif source == "yahoo_finance":
                historical_data = await self.yahoo_client.get_historical_data(symbol, days)

            if historical_data:
                await db_manager.bulk_insert_prices(historical_data)
                logger.info(f"Ingested {len(historical_data)} historical records for {symbol}")
                return len(historical_data)

        except Exception as e:
            logger.error(f"Failed to ingest historical data for {symbol}: {e}")
            raise

        return 0

    async def calculate_real_volatility(self, symbol: str, window_days: int = 30) -> Optional[float]:
        """Calculate actual volatility from real price data."""
        try:
            price_history = await db_manager.get_price_history(symbol, window_days)

            if len(price_history) < 10:  # Need minimum data points
                return None

            # Convert to pandas for efficient calculation
            df = pd.DataFrame(price_history)
            df['returns'] = df['price'].pct_change()

            # Calculate annualized volatility
            volatility = df['returns'].std() * (365 ** 0.5)  # Annualized

            return float(volatility)

        except Exception as e:
            logger.error(f"Failed to calculate volatility for {symbol}: {e}")
            return None

    async def get_correlation_matrix(self, symbols: List[str], days: int = 30) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix from real price data."""
        try:
            price_data = {}

            for symbol in symbols:
                history = await db_manager.get_price_history(symbol, days)
                if history:
                    df = pd.DataFrame(history)
                    df['returns'] = df['price'].pct_change()
                    price_data[symbol] = df.set_index('time')['returns']

            if len(price_data) < 2:
                return None

            # Combine into single DataFrame and calculate correlation
            combined_df = pd.DataFrame(price_data)
            correlation_matrix = combined_df.corr()

            return correlation_matrix

        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            return None


# Global service instance
ingestion_service = DataIngestionService()