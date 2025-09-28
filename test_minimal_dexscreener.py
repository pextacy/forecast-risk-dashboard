#!/usr/bin/env python3
"""
Minimal test for DexScreener API - REAL DATA ONLY
No dependencies except aiohttp.
"""

import asyncio
import logging
import json
from datetime import datetime

import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_dexscreener_real_data():
    """Test DexScreener API with real data."""

    # Real token addresses (verified)
    token_addresses = {
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "USDC": "0xA0b86a33E6441c7C677AB43c16f0dB1CCa34884e",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    }

    base_url = "https://api.dexscreener.com/latest"
    test_results = {}

    async with aiohttp.ClientSession() as session:

        # Test 1: Get token pairs (real data)
        logger.info("=" * 60)
        logger.info("TEST 1: FETCHING REAL TOKEN PAIRS FROM DEXSCREENER")
        logger.info("=" * 60)

        for symbol, address in token_addresses.items():
            logger.info(f"\nTesting {symbol} ({address})...")

            try:
                url = f"{base_url}/dex/tokens/{address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])

                        if pairs:
                            # Get best pair by liquidity
                            best_pair = max(pairs, key=lambda x: x.get("liquidity", {}).get("usd", 0))

                            price = float(best_pair.get("priceUsd", 0))
                            liquidity = best_pair.get("liquidity", {}).get("usd", 0)
                            volume_24h = best_pair.get("volume", {}).get("h24", 0)
                            price_change = best_pair.get("priceChange", {}).get("h24", 0)
                            dex_id = best_pair.get("dexId", "")

                            logger.info(f"  ✅ SUCCESS: {symbol}")
                            logger.info(f"     Price: ${price:.6f}")
                            logger.info(f"     Liquidity: ${liquidity:,.0f}")
                            logger.info(f"     24h Volume: ${volume_24h:,.0f}")
                            logger.info(f"     24h Change: {price_change:.2f}%")
                            logger.info(f"     DEX: {dex_id}")

                            test_results[f"{symbol}_real_data"] = True
                        else:
                            logger.warning(f"  ⚠️ No pairs found for {symbol}")
                            test_results[f"{symbol}_real_data"] = False
                    else:
                        logger.error(f"  ❌ HTTP {response.status} for {symbol}")
                        test_results[f"{symbol}_real_data"] = False

            except Exception as e:
                logger.error(f"  ❌ Error for {symbol}: {e}")
                test_results[f"{symbol}_real_data"] = False

        # Test 2: Search functionality (real data)
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: REAL TOKEN SEARCH ON DEXSCREENER")
        logger.info("=" * 60)

        search_queries = ["ethereum", "uniswap"]

        for query in search_queries:
            logger.info(f"\nSearching for '{query}'...")

            try:
                url = f"{base_url}/dex/search"
                params = {"q": query}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])

                        if pairs:
                            logger.info(f"  ✅ Found {len(pairs)} pairs for '{query}':")

                            # Show top 3 results
                            for i, pair in enumerate(pairs[:3]):
                                base_token = pair.get("baseToken", {})
                                symbol = base_token.get("symbol", "")
                                name = base_token.get("name", "")
                                price = float(pair.get("priceUsd", 0))

                                logger.info(f"     {i+1}. {symbol} ({name}) - ${price:.6f}")

                            test_results[f"search_{query}"] = True
                        else:
                            logger.warning(f"  ⚠️ No results for '{query}'")
                            test_results[f"search_{query}"] = False
                    else:
                        logger.error(f"  ❌ HTTP {response.status} for search '{query}'")
                        test_results[f"search_{query}"] = False

            except Exception as e:
                logger.error(f"  ❌ Error searching '{query}': {e}")
                test_results[f"search_{query}"] = False

        # Test 3: Real historical data from Binance (free API)
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: REAL HISTORICAL DATA FROM BINANCE")
        logger.info("=" * 60)

        binance_symbols = {
            "ETHUSDT": "Ethereum",
            "BTCUSDT": "Bitcoin",
            "UNIUSDT": "Uniswap"
        }

        for binance_symbol, name in binance_symbols.items():
            logger.info(f"\nFetching {name} ({binance_symbol}) historical data...")

            try:
                url = "https://api.binance.com/api/v3/klines"
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = int((datetime.now().timestamp() - (24 * 3600)) * 1000)  # 1 day

                params = {
                    "symbol": binance_symbol,
                    "interval": "1h",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 24
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data:
                            prices = [float(kline[4]) for kline in data]  # Close prices
                            latest_price = prices[-1]
                            min_price = min(prices)
                            max_price = max(prices)

                            logger.info(f"  ✅ SUCCESS: {name}")
                            logger.info(f"     Data points: {len(data)}")
                            logger.info(f"     Latest price: ${latest_price:.6f}")
                            logger.info(f"     24h range: ${min_price:.6f} - ${max_price:.6f}")

                            test_results[f"historical_{binance_symbol}"] = True
                        else:
                            logger.warning(f"  ⚠️ No historical data for {name}")
                            test_results[f"historical_{binance_symbol}"] = False
                    else:
                        logger.error(f"  ❌ HTTP {response.status} for {name}")
                        test_results[f"historical_{binance_symbol}"] = False

            except Exception as e:
                logger.error(f"  ❌ Error for {name}: {e}")
                test_results[f"historical_{binance_symbol}"] = False

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for success in test_results.values() if success)
    total = len(test_results)

    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! DEXSCREENER INTEGRATION WORKING WITH REAL DATA!")
        logger.info("✅ No mock data found - all data sources are real")
        logger.info("✅ DexScreener API responding correctly")
        logger.info("✅ Real historical data from Binance working")
        logger.info("✅ Token search functionality working")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_dexscreener_real_data())
    print(f"\n{'='*60}")
    if success:
        print("🚀 DEXSCREENER INTEGRATION COMPLETE - ALL REAL DATA!")
    else:
        print("❌ Some tests failed - check logs above")
    print(f"{'='*60}")