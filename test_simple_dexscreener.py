#!/usr/bin/env python3
"""
Simple test for DexScreener API - REAL DATA ONLY
Tests core DexScreener functionality without database dependencies.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import aiohttp
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDexScreenerTest:
    """Simple DexScreener API test with REAL data only."""

    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest"
        self.session = None

        # Real token addresses (verified)
        self.token_addresses = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86a33E6441c7C677AB43c16f0dB1CCa34884e",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_token_pairs(self, token_address: str) -> dict:
        """Test fetching real token pairs from DexScreener."""
        url = f"{self.base_url}/dex/tokens/{token_address}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])

                    if pairs:
                        # Get the best pair (highest liquidity)
                        best_pair = max(pairs, key=lambda x: x.get("liquidity", {}).get("usd", 0))
                        return {
                            "success": True,
                            "token_address": token_address,
                            "pair_count": len(pairs),
                            "best_pair": {
                                "dex_id": best_pair.get("dexId", ""),
                                "pair_address": best_pair.get("pairAddress", ""),
                                "price_usd": float(best_pair.get("priceUsd", 0)),
                                "liquidity_usd": best_pair.get("liquidity", {}).get("usd", 0),
                                "volume_24h": best_pair.get("volume", {}).get("h24", 0),
                                "price_change_24h": best_pair.get("priceChange", {}).get("h24", 0),
                                "base_token": best_pair.get("baseToken", {}),
                                "quote_token": best_pair.get("quoteToken", {}),
                            }
                        }
                    else:
                        return {"success": False, "error": "No pairs found"}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_search_tokens(self, query: str) -> dict:
        """Test searching for tokens on DexScreener."""
        url = f"{self.base_url}/dex/search"
        params = {"q": query}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])

                    results = []
                    seen_tokens = set()

                    for pair in pairs[:10]:  # Limit to top 10
                        base_token = pair.get("baseToken", {})
                        token_address = base_token.get("address", "")

                        if token_address and token_address not in seen_tokens:
                            seen_tokens.add(token_address)
                            results.append({
                                "symbol": base_token.get("symbol", ""),
                                "name": base_token.get("name", ""),
                                "address": token_address,
                                "price_usd": float(pair.get("priceUsd", 0)),
                                "volume_24h": pair.get("volume", {}).get("h24", 0),
                                "liquidity_usd": pair.get("liquidity", {}).get("usd", 0),
                                "dex_id": pair.get("dexId", ""),
                            })

                    return {"success": True, "query": query, "results": results}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_historical_data_binance(self, symbol: str, days: int = 7) -> dict:
        """Test fetching REAL historical data from Binance (free API)."""
        if symbol.upper() == "WETH":
            binance_symbol = "ETHUSDT"
        elif symbol.upper() == "WBTC":
            binance_symbol = "BTCUSDT"
        elif symbol.upper() == "USDC":
            binance_symbol = "USDCUSDT"
        else:
            binance_symbol = f"{symbol.upper()}USDT"

        url = "https://api.binance.com/api/v3/klines"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now().timestamp() - (days * 24 * 3600)) * 1000)

        params = {
            "symbol": binance_symbol,
            "interval": "1h",
            "startTime": start_time,
            "endTime": end_time,
            "limit": min(1000, days * 24)
        }

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    historical_data = []
                    for kline in data:
                        historical_data.append({
                            "timestamp": datetime.fromtimestamp(int(kline[0]) / 1000),
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                        })

                    return {
                        "success": True,
                        "symbol": binance_symbol,
                        "data_points": len(historical_data),
                        "latest_price": historical_data[-1]["close"] if historical_data else 0,
                        "price_range": {
                            "min": min(h["close"] for h in historical_data) if historical_data else 0,
                            "max": max(h["close"] for h in historical_data) if historical_data else 0,
                        }
                    }
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


async def run_tests():
    """Run all DexScreener tests with REAL data."""
    logger.info("=" * 60)
    logger.info("TESTING DEXSCREENER WITH REAL DATA ONLY")
    logger.info("=" * 60)

    async with SimpleDexScreenerTest() as tester:
        results = {}

        # Test 1: Token pairs
        logger.info("\n--- Test 1: Token Pairs (Real Data) ---")
        for symbol, address in tester.token_addresses.items():
            logger.info(f"Testing {symbol} ({address})...")
            result = await tester.test_token_pairs(address)

            if result["success"]:
                pair = result["best_pair"]
                logger.info(f"  ✅ {symbol}: ${pair['price_usd']:.4f} | "
                          f"Liquidity: ${pair['liquidity_usd']:,.0f} | "
                          f"Volume: ${pair['volume_24h']:,.0f}")
                results[f"{symbol}_pairs"] = True
            else:
                logger.error(f"  ❌ {symbol}: {result['error']}")
                results[f"{symbol}_pairs"] = False

        # Test 2: Token search
        logger.info("\n--- Test 2: Token Search (Real Data) ---")
        search_queries = ["ethereum", "uniswap", "chainlink"]

        for query in search_queries:
            logger.info(f"Searching for '{query}'...")
            result = await tester.test_search_tokens(query)

            if result["success"]:
                logger.info(f"  ✅ Found {len(result['results'])} tokens for '{query}':")
                for i, token in enumerate(result["results"][:3]):  # Show top 3
                    logger.info(f"    {i+1}. {token['symbol']} - ${token['price_usd']:.4f}")
                results[f"search_{query}"] = True
            else:
                logger.error(f"  ❌ Search failed for '{query}': {result['error']}")
                results[f"search_{query}"] = False

        # Test 3: Historical data from Binance
        logger.info("\n--- Test 3: Historical Data from Binance (Real Data) ---")
        test_symbols = ["WETH", "WBTC", "UNI"]

        for symbol in test_symbols:
            logger.info(f"Fetching historical data for {symbol}...")
            result = await tester.test_historical_data_binance(symbol, 3)  # 3 days

            if result["success"]:
                logger.info(f"  ✅ {symbol}: {result['data_points']} data points | "
                          f"Latest: ${result['latest_price']:.4f} | "
                          f"Range: ${result['price_range']['min']:.4f} - ${result['price_range']['max']:.4f}")
                results[f"historical_{symbol}"] = True
            else:
                logger.error(f"  ❌ {symbol}: {result['error']}")
                results[f"historical_{symbol}"] = False

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for success in results.values() if success)
        total = len(results)

        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All tests passed! DexScreener integration working with REAL DATA!")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed. Check implementation.")

        return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)