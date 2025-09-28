"""Real-time data ingestion service using DexScreener API ONLY - NO MOCK DATA."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
import yfinance as yf
from app.db.connection import db_manager
from app.utils.config import settings, get_api_headers, get_dexscreener_url
from app.services.dexscreener_client import DexScreenerClient

logger = logging.getLogger(__name__)


class HistoricalDataClient:
    """
    Real historical data client using multiple sources.
    Since DexScreener doesn't provide historical data, we use other real sources.
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_historical_data_from_binance(self, symbol: str, days: int = 30) -> List[dict]:
        """Fetch real historical data from Binance API (free, no API key required)."""
        # Convert symbol to Binance format
        if symbol.upper() == "ETHEREUM":
            symbol = "ETHUSDT"
        elif symbol.upper() == "BITCOIN":
            symbol = "BTCUSDT"
        elif symbol.upper() == "USDC":
            symbol = "USDCUSDT"
        elif symbol.upper() == "USDT":
            return []  # USDT is the base, skip
        else:
            symbol = f"{symbol.upper()}USDT"

        url = "https://api.binance.com/api/v3/klines"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    historical_data = []
                    for kline in data:
                        historical_data.append({
                            "time": datetime.fromtimestamp(int(kline[0]) / 1000),
                            "symbol": symbol.replace("USDT", "").lower(),
                            "price": float(kline[4]),  # Close price
                            "volume": float(kline[5]),  # Volume
                            "market_cap": None,  # Not available from Binance
                            "source": "binance"
                        })

                    logger.info(f"Fetched {len(historical_data)} real historical records for {symbol} from Binance")
                    return historical_data
                else:
                    logger.warning(f"Binance API error: {response.status} for {symbol}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching historical data from Binance for {symbol}: {e}")
            return []

    async def get_historical_data_from_coinbase(self, symbol: str, days: int = 30) -> List[dict]:
        """Fetch real historical data from Coinbase Pro API (free, no API key required)."""
        # Convert symbol to Coinbase format
        if symbol.upper() == "ETHEREUM":
            symbol = "ETH-USD"
        elif symbol.upper() == "BITCOIN":
            symbol = "BTC-USD"
        elif symbol.upper() == "USDC":
            symbol = "USDC-USD"
        else:
            symbol = f"{symbol.upper()}-USD"

        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        end_time = datetime.now().isoformat()
        start_time = (datetime.now() - timedelta(days=days)).isoformat()

        params = {
            "start": start_time,
            "end": end_time,
            "granularity": 3600  # 1 hour
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    historical_data = []
                    for candle in data:
                        historical_data.append({
                            "time": datetime.fromtimestamp(candle[0]),
                            "symbol": symbol.split("-")[0].lower(),
                            "price": float(candle[4]),  # Close price
                            "volume": float(candle[5]),  # Volume
                            "market_cap": None,  # Not available from Coinbase
                            "source": "coinbase"
                        })

                    # Sort by time (Coinbase returns reverse chronological)
                    historical_data.sort(key=lambda x: x["time"])

                    logger.info(f"Fetched {len(historical_data)} real historical records for {symbol} from Coinbase")
                    return historical_data
                else:
                    logger.warning(f"Coinbase API error: {response.status} for {symbol}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching historical data from Coinbase for {symbol}: {e}")
            return []


class YahooFinanceClient:
    """Yahoo Finance client for traditional financial assets - REAL DATA ONLY."""

    def __init__(self):
        self.symbols = settings.supported_stock_symbols

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetch current prices for stock symbols - REAL DATA ONLY."""
        try:
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
                    logger.warning(f"Failed to fetch real data for {symbol}: {e}")
                    continue

            logger.info(f"Fetched Yahoo Finance REAL data for {len(current_data)} symbols")
            return current_data

        except Exception as e:
            logger.error(f"Yahoo Finance API error: {e}")
            raise

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[dict]:
        """Fetch historical data for a stock symbol - REAL DATA ONLY."""
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)

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

            logger.info(f"Fetched {len(historical_data)} REAL historical records for {symbol}")
            return historical_data

        except Exception as e:
            logger.error(f"Yahoo Finance historical data error for {symbol}: {e}")
            raise


class DataIngestionService:
    """Main data ingestion service using DexScreener and real historical sources ONLY."""

    def __init__(self):
        self.dexscreener_client = DexScreenerClient()
        self.yahoo_client = YahooFinanceClient()
        self.historical_client = HistoricalDataClient()

    async def ingest_current_market_data(self) -> Tuple[int, int]:
        """Ingest current market data from DexScreener and Yahoo Finance - REAL DATA ONLY."""
        crypto_count = 0
        stock_count = 0

        try:
            # Fetch cryptocurrency data from DexScreener (REAL DATA)
            async with DexScreenerClient() as client:
                crypto_symbols = ["ethereum", "bitcoin", "usdc", "usdt", "uniswap", "chainlink", "aave"]
                crypto_prices = await client.get_current_prices(crypto_symbols)

                crypto_records = []
                for symbol, price_data in crypto_prices.items():
                    crypto_records.append({
                        "time": datetime.fromtimestamp(price_data["last_updated_at"]),
                        "symbol": symbol,
                        "price": price_data["usd"],
                        "volume": price_data.get("usd_24h_vol"),
                        "market_cap": price_data.get("usd_market_cap"),
                        "source": "dexscreener"
                    })

                if crypto_records:
                    await db_manager.bulk_insert_prices(crypto_records)
                    crypto_count = len(crypto_records)
                    logger.info(f"Ingested {crypto_count} REAL cryptocurrency price records from DexScreener")

        except Exception as e:
            logger.error(f"Failed to ingest crypto data from DexScreener: {e}")

        try:
            # Fetch stock data from Yahoo Finance (REAL DATA)
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
                logger.info(f"Ingested {stock_count} REAL stock price records from Yahoo Finance")

        except Exception as e:
            logger.error(f"Failed to ingest stock data from Yahoo Finance: {e}")

        return crypto_count, stock_count

    async def ingest_historical_data(self, symbol: str, days: int = 90, source: str = "auto") -> int:
        """Ingest historical data for a specific symbol - REAL DATA ONLY."""
        try:
            if source == "auto":
                # Determine source based on symbol
                crypto_symbols = ["ethereum", "bitcoin", "usdc", "usdt", "uniswap", "chainlink", "aave"]
                if symbol.lower() in crypto_symbols:
                    source = "crypto"
                elif symbol in settings.supported_stock_symbols:
                    source = "yahoo_finance"
                else:
                    raise ValueError(f"Unknown symbol: {symbol}")

            historical_data = []

            if source == "crypto":
                # Try multiple real sources for crypto historical data
                async with HistoricalDataClient() as client:
                    # Try Coinbase first
                    historical_data = await client.get_historical_data_from_coinbase(symbol, days)

                    # If Coinbase fails, try Binance
                    if not historical_data:
                        historical_data = await client.get_historical_data_from_binance(symbol, days)

            elif source == "yahoo_finance":
                historical_data = await self.yahoo_client.get_historical_data(symbol, days)

            if historical_data:
                await db_manager.bulk_insert_prices(historical_data)
                logger.info(f"Ingested {len(historical_data)} REAL historical records for {symbol}")
                return len(historical_data)
            else:
                logger.warning(f"No real historical data found for {symbol}")

        except Exception as e:
            logger.error(f"Failed to ingest real historical data for {symbol}: {e}")
            raise

        return 0

    async def calculate_real_volatility(self, symbol: str, window_days: int = 30) -> Optional[float]:
        """Calculate actual volatility from REAL price data ONLY."""
        try:
            price_history = await db_manager.get_price_history(symbol, window_days)

            if len(price_history) < 10:  # Need minimum data points
                logger.warning(f"Insufficient real data for volatility calculation: {symbol}")
                return None

            # Convert to pandas for efficient calculation
            df = pd.DataFrame(price_history)
            df['returns'] = df['price'].pct_change()

            # Calculate annualized volatility from REAL data
            volatility = df['returns'].std() * (365 ** 0.5)  # Annualized

            logger.info(f"Calculated real volatility for {symbol}: {volatility:.4f}")
            return float(volatility)

        except Exception as e:
            logger.error(f"Failed to calculate real volatility for {symbol}: {e}")
            return None

    async def get_correlation_matrix(self, symbols: List[str], days: int = 30) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix from REAL price data ONLY."""
        try:
            price_data = {}

            for symbol in symbols:
                history = await db_manager.get_price_history(symbol, days)
                if history and len(history) >= 10:  # Minimum real data points
                    df = pd.DataFrame(history)
                    df['returns'] = df['price'].pct_change()
                    price_data[symbol] = df.set_index('time')['returns']

            if len(price_data) < 2:
                logger.warning("Insufficient real data for correlation matrix calculation")
                return None

            # Combine into single DataFrame and calculate correlation from REAL data
            combined_df = pd.DataFrame(price_data)
            correlation_matrix = combined_df.corr()

            logger.info(f"Calculated real correlation matrix for {len(price_data)} symbols")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Failed to calculate real correlation matrix: {e}")
            return None

    async def get_dexscreener_market_data(self, symbols: List[str]) -> List[dict]:
        """Get comprehensive market data from DexScreener - REAL DATA ONLY."""
        try:
            async with DexScreenerClient() as client:
                market_data = await client.get_market_data(symbols)
                logger.info(f"Fetched real market data for {len(market_data)} symbols from DexScreener")
                return market_data
        except Exception as e:
            logger.error(f"Failed to fetch real market data from DexScreener: {e}")
            return []

    async def search_tokens(self, query: str) -> List[dict]:
        """Search for tokens using DexScreener - REAL DATA ONLY."""
        try:
            async with DexScreenerClient() as client:
                results = await client.search_tokens(query)
                logger.info(f"Found {len(results)} real tokens for query: {query}")
                return results
        except Exception as e:
            logger.error(f"Failed to search tokens: {e}")
            return []


# Global service instance
ingestion_service = DataIngestionService()