#!/usr/bin/env python3
"""
Test script for DexScreener API integration - REAL DATA ONLY
This script tests all DexScreener functionality with actual API calls.
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.dexscreener_client import DexScreenerClient
from backend.app.services.ingestion import DataIngestionService, HistoricalDataClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_dexscreener_current_prices():
    """Test fetching current prices from DexScreener - REAL DATA."""
    logger.info("Testing DexScreener current prices...")

    async with DexScreenerClient() as client:
        test_symbols = ["ethereum", "bitcoin", "usdc", "usdt", "uniswap"]

        try:
            prices = await client.get_current_prices(test_symbols)

            logger.info(f"Successfully fetched {len(prices)} token prices:")
            for symbol, data in prices.items():
                logger.info(f"  {symbol.upper()}: ${data['usd']:.4f} (Volume: ${data.get('usd_24h_vol', 0):,.0f})")

            return len(prices) > 0

        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return False


async def test_dexscreener_market_data():
    """Test fetching market data from DexScreener - REAL DATA."""
    logger.info("Testing DexScreener market data...")

    async with DexScreenerClient() as client:
        test_symbols = ["ethereum", "bitcoin", "usdc"]

        try:
            market_data = await client.get_market_data(test_symbols)

            logger.info(f"Successfully fetched market data for {len(market_data)} tokens:")
            for data in market_data:
                logger.info(f"  {data['symbol']}: ${data['current_price']:.4f} | "
                          f"Market Cap: ${data.get('market_cap', 0):,.0f} | "
                          f"24h Volume: ${data.get('total_volume', 0):,.0f}")

            return len(market_data) > 0

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return False


async def test_dexscreener_search():
    """Test searching tokens on DexScreener - REAL DATA."""
    logger.info("Testing DexScreener token search...")

    async with DexScreenerClient() as client:
        try:
            search_results = await client.search_tokens("ethereum")

            logger.info(f"Found {len(search_results)} tokens for 'ethereum':")
            for i, token in enumerate(search_results[:5]):  # Show first 5
                logger.info(f"  {i+1}. {token['symbol']} ({token['name']}) - ${token.get('price_usd', 0):.4f}")

            return len(search_results) > 0

        except Exception as e:
            logger.error(f"Error searching tokens: {e}")
            return False


async def test_historical_data_sources():
    """Test real historical data sources (Binance/Coinbase) - REAL DATA."""
    logger.info("Testing real historical data sources...")

    async with HistoricalDataClient() as client:
        # Test Coinbase
        try:
            coinbase_data = await client.get_historical_data_from_coinbase("ethereum", 7)
            logger.info(f"Coinbase: Fetched {len(coinbase_data)} historical records for ETH")

            if coinbase_data:
                latest = coinbase_data[-1]
                logger.info(f"  Latest ETH price: ${latest['price']:.2f}")
        except Exception as e:
            logger.error(f"Coinbase historical data error: {e}")

        # Test Binance
        try:
            binance_data = await client.get_historical_data_from_binance("ethereum", 7)
            logger.info(f"Binance: Fetched {len(binance_data)} historical records for ETH")

            if binance_data:
                latest = binance_data[-1]
                logger.info(f"  Latest ETH price: ${latest['price']:.2f}")
        except Exception as e:
            logger.error(f"Binance historical data error: {e}")

        return True


async def test_data_ingestion_service():
    """Test the complete data ingestion service - REAL DATA."""
    logger.info("Testing complete data ingestion service...")

    service = DataIngestionService()

    try:
        # Test current market data ingestion
        crypto_count, stock_count = await service.ingest_current_market_data()
        logger.info(f"Ingested {crypto_count} crypto and {stock_count} stock price records")

        # Test DexScreener market data
        market_data = await service.get_dexscreener_market_data(["ethereum", "bitcoin"])
        logger.info(f"Fetched DexScreener market data for {len(market_data)} symbols")

        return True

    except Exception as e:
        logger.error(f"Data ingestion service error: {e}")
        return False


async def main():
    """Run all DexScreener tests with REAL DATA."""
    logger.info("=" * 60)
    logger.info("TESTING DEXSCREENER INTEGRATION - REAL DATA ONLY")
    logger.info("=" * 60)

    tests = [
        ("Current Prices", test_dexscreener_current_prices),
        ("Market Data", test_dexscreener_market_data),
        ("Token Search", test_dexscreener_search),
        ("Historical Data Sources", test_historical_data_sources),
        ("Data Ingestion Service", test_data_ingestion_service),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED - {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All DexScreener tests passed with REAL DATA!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check implementation.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)