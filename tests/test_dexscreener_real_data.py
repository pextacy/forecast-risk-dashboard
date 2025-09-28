#!/usr/bin/env python3
"""
Comprehensive DexScreener API Integration Tests - REAL DATA ONLY
Tests all DexScreener functionality with actual API calls and real market data.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DexScreenerRealDataTests:
    """Comprehensive tests for DexScreener with REAL data only."""

    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest"
        self.session = None
        self.test_results = {}

        # Real verified token addresses (Ethereum mainnet)
        self.verified_tokens = {
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
            "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
            "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_token_pairs_real_data(self):
        """Test fetching real token pairs with comprehensive validation."""
        logger.info("=" * 60)
        logger.info("TEST 1: REAL TOKEN PAIRS FROM DEXSCREENER")
        logger.info("=" * 60)

        for symbol, address in self.verified_tokens.items():
            logger.info(f"Testing {symbol} ({address})...")

            try:
                url = f"{self.base_url}/dex/tokens/{address}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])

                        if pairs:
                            # Validate data structure and content
                            best_pair = max(pairs, key=lambda x: x.get("liquidity", {}).get("usd", 0))

                            # Comprehensive validation
                            price = float(best_pair.get("priceUsd", 0))
                            liquidity = best_pair.get("liquidity", {}).get("usd", 0)
                            volume_24h = best_pair.get("volume", {}).get("h24", 0)
                            dex_id = best_pair.get("dexId", "")
                            pair_address = best_pair.get("pairAddress", "")

                            # Validate data quality
                            if price > 0 and liquidity > 0 and dex_id and pair_address:
                                logger.info(f"  ✅ {symbol}: REAL DATA VALIDATED")
                                logger.info(f"     Price: ${price:.6f}")
                                logger.info(f"     Liquidity: ${liquidity:,.0f}")
                                logger.info(f"     Volume 24h: ${volume_24h:,.0f}")
                                logger.info(f"     DEX: {dex_id}")
                                logger.info(f"     Pair: {pair_address}")

                                self.test_results[f"{symbol}_pairs"] = True
                            else:
                                logger.warning(f"  ⚠️ {symbol}: Incomplete data")
                                self.test_results[f"{symbol}_pairs"] = False
                        else:
                            logger.warning(f"  ⚠️ {symbol}: No pairs found")
                            self.test_results[f"{symbol}_pairs"] = False
                    else:
                        logger.error(f"  ❌ {symbol}: HTTP {response.status}")
                        self.test_results[f"{symbol}_pairs"] = False

            except Exception as e:
                logger.error(f"  ❌ {symbol}: {e}")
                self.test_results[f"{symbol}_pairs"] = False

        return True

    async def test_market_data_comprehensive(self):
        """Test comprehensive market data with validation."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: COMPREHENSIVE MARKET DATA VALIDATION")
        logger.info("=" * 60)

        test_symbols = ["WETH", "WBTC", "UNI"]

        for symbol in test_symbols:
            address = self.verified_tokens[symbol]
            logger.info(f"Testing market data for {symbol}...")

            try:
                url = f"{self.base_url}/dex/tokens/{address}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])

                        if pairs:
                            best_pair = pairs[0]

                            # Extract and validate all available data
                            market_data = {
                                "symbol": symbol,
                                "price_usd": float(best_pair.get("priceUsd", 0)),
                                "liquidity_usd": best_pair.get("liquidity", {}).get("usd", 0),
                                "volume_24h": best_pair.get("volume", {}).get("h24", 0),
                                "volume_1h": best_pair.get("volume", {}).get("h1", 0),
                                "price_change_24h": best_pair.get("priceChange", {}).get("h24", 0),
                                "price_change_1h": best_pair.get("priceChange", {}).get("h1", 0),
                                "dex_id": best_pair.get("dexId", ""),
                                "pair_address": best_pair.get("pairAddress", ""),
                                "chain_id": best_pair.get("chainId", ""),
                                "base_token": best_pair.get("baseToken", {}),
                                "quote_token": best_pair.get("quoteToken", {}),
                            }

                            # Validate completeness
                            required_fields = ["price_usd", "liquidity_usd", "dex_id", "pair_address"]
                            valid = all(market_data.get(field) for field in required_fields)

                            if valid and market_data["price_usd"] > 0:
                                logger.info(f"  ✅ {symbol}: COMPLETE MARKET DATA")
                                logger.info(f"     Price: ${market_data['price_usd']:.6f}")
                                logger.info(f"     Liquidity: ${market_data['liquidity_usd']:,.0f}")
                                logger.info(f"     24h Volume: ${market_data['volume_24h']:,.0f}")
                                logger.info(f"     24h Change: {market_data['price_change_24h']:.2f}%")
                                logger.info(f"     Chain: {market_data['chain_id']}")

                                self.test_results[f"{symbol}_market_data"] = True
                            else:
                                logger.warning(f"  ⚠️ {symbol}: Incomplete market data")
                                self.test_results[f"{symbol}_market_data"] = False
                        else:
                            logger.warning(f"  ⚠️ {symbol}: No market data")
                            self.test_results[f"{symbol}_market_data"] = False
                    else:
                        logger.error(f"  ❌ {symbol}: HTTP {response.status}")
                        self.test_results[f"{symbol}_market_data"] = False

            except Exception as e:
                logger.error(f"  ❌ {symbol}: {e}")
                self.test_results[f"{symbol}_market_data"] = False

        return True

    async def test_search_functionality(self):
        """Test token search with comprehensive validation."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: SEARCH FUNCTIONALITY VALIDATION")
        logger.info("=" * 60)

        search_tests = [
            ("ethereum", 5),
            ("uniswap", 3),
            ("bitcoin", 3),
            ("chainlink", 2)
        ]

        for query, min_results in search_tests:
            logger.info(f"Searching for '{query}' (expecting ≥{min_results} results)...")

            try:
                url = f"{self.base_url}/dex/search"
                params = {"q": query}

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])

                        if len(pairs) >= min_results:
                            logger.info(f"  ✅ Found {len(pairs)} pairs for '{query}':")

                            # Validate search results
                            valid_results = 0
                            for i, pair in enumerate(pairs[:5]):
                                base_token = pair.get("baseToken", {})
                                symbol = base_token.get("symbol", "")
                                price = float(pair.get("priceUsd", 0))

                                if symbol and price > 0:
                                    valid_results += 1
                                    logger.info(f"     {i+1}. {symbol} - ${price:.6f}")

                            if valid_results >= min_results:
                                self.test_results[f"search_{query}"] = True
                            else:
                                logger.warning(f"  ⚠️ Only {valid_results} valid results")
                                self.test_results[f"search_{query}"] = False
                        else:
                            logger.warning(f"  ⚠️ Only {len(pairs)} results for '{query}'")
                            self.test_results[f"search_{query}"] = False
                    else:
                        logger.error(f"  ❌ HTTP {response.status} for '{query}'")
                        self.test_results[f"search_{query}"] = False

            except Exception as e:
                logger.error(f"  ❌ Search error for '{query}': {e}")
                self.test_results[f"search_{query}"] = False

        return True

    async def test_historical_data_sources(self):
        """Test real historical data sources."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: REAL HISTORICAL DATA SOURCES")
        logger.info("=" * 60)

        # Test Binance API
        binance_symbols = {
            "ETHUSDT": "Ethereum",
            "BTCUSDT": "Bitcoin",
            "UNIUSDT": "Uniswap"
        }

        for binance_symbol, name in binance_symbols.items():
            logger.info(f"Testing Binance historical data for {name}...")

            try:
                url = "https://api.binance.com/api/v3/klines"
                end_time = int(datetime.now().timestamp() * 1000)
                start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

                params = {
                    "symbol": binance_symbol,
                    "interval": "1h",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 24
                }

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data and len(data) >= 10:  # Minimum data points
                            prices = [float(kline[4]) for kline in data]
                            volumes = [float(kline[5]) for kline in data]

                            # Validate data quality
                            valid_prices = all(p > 0 for p in prices)
                            valid_volumes = all(v >= 0 for v in volumes)

                            if valid_prices and valid_volumes:
                                logger.info(f"  ✅ {name}: REAL HISTORICAL DATA VALIDATED")
                                logger.info(f"     Data points: {len(data)}")
                                logger.info(f"     Latest price: ${prices[-1]:.6f}")
                                logger.info(f"     Price range: ${min(prices):.6f} - ${max(prices):.6f}")
                                logger.info(f"     Avg volume: {sum(volumes)/len(volumes):,.0f}")

                                self.test_results[f"historical_{binance_symbol}"] = True
                            else:
                                logger.warning(f"  ⚠️ {name}: Invalid price/volume data")
                                self.test_results[f"historical_{binance_symbol}"] = False
                        else:
                            logger.warning(f"  ⚠️ {name}: Insufficient data points")
                            self.test_results[f"historical_{binance_symbol}"] = False
                    else:
                        logger.error(f"  ❌ {name}: HTTP {response.status}")
                        self.test_results[f"historical_{binance_symbol}"] = False

            except Exception as e:
                logger.error(f"  ❌ {name}: {e}")
                self.test_results[f"historical_{binance_symbol}"] = False

        return True

    async def test_data_consistency(self):
        """Test data consistency across multiple calls."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: DATA CONSISTENCY VALIDATION")
        logger.info("=" * 60)

        test_token = "WETH"
        address = self.verified_tokens[test_token]
        logger.info(f"Testing data consistency for {test_token}...")

        try:
            # Make multiple calls and compare
            prices = []
            for i in range(3):
                url = f"{self.base_url}/dex/tokens/{address}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get("pairs", [])
                        if pairs:
                            price = float(pairs[0].get("priceUsd", 0))
                            prices.append(price)
                            logger.info(f"  Call {i+1}: ${price:.6f}")

                await asyncio.sleep(1)  # Small delay between calls

            if len(prices) == 3:
                # Check price consistency (should be very similar)
                max_price = max(prices)
                min_price = min(prices)
                variation = ((max_price - min_price) / min_price) * 100

                if variation < 5.0:  # Less than 5% variation
                    logger.info(f"  ✅ Data consistency validated (variation: {variation:.2f}%)")
                    self.test_results["data_consistency"] = True
                else:
                    logger.warning(f"  ⚠️ High price variation: {variation:.2f}%")
                    self.test_results["data_consistency"] = False
            else:
                logger.warning("  ⚠️ Could not complete consistency test")
                self.test_results["data_consistency"] = False

        except Exception as e:
            logger.error(f"  ❌ Consistency test error: {e}")
            self.test_results["data_consistency"] = False

        return True

    async def run_all_tests(self):
        """Run all comprehensive tests."""
        logger.info("🚀 STARTING COMPREHENSIVE DEXSCREENER TESTS - REAL DATA ONLY")
        logger.info("=" * 80)

        test_functions = [
            ("Token Pairs Real Data", self.test_token_pairs_real_data),
            ("Market Data Comprehensive", self.test_market_data_comprehensive),
            ("Search Functionality", self.test_search_functionality),
            ("Historical Data Sources", self.test_historical_data_sources),
            ("Data Consistency", self.test_data_consistency),
        ]

        for test_name, test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Test '{test_name}' failed with error: {e}")

        # Final summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 80)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        categories = {
            "Token Pairs": [k for k in self.test_results.keys() if "_pairs" in k],
            "Market Data": [k for k in self.test_results.keys() if "_market_data" in k],
            "Search": [k for k in self.test_results.keys() if "search_" in k],
            "Historical": [k for k in self.test_results.keys() if "historical_" in k],
            "Consistency": [k for k in self.test_results.keys() if "consistency" in k],
        }

        for category, tests in categories.items():
            if tests:
                passed_in_category = sum(1 for test in tests if self.test_results.get(test, False))
                logger.info(f"\n{category}: {passed_in_category}/{len(tests)} passed")
                for test in tests:
                    status = "✅ PASSED" if self.test_results.get(test, False) else "❌ FAILED"
                    logger.info(f"  {test}: {status}")

        logger.info(f"\nOVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! DEXSCREENER INTEGRATION FULLY VALIDATED!")
            logger.info("✅ All data sources using REAL market data")
            logger.info("✅ No mock or synthetic data found")
            logger.info("✅ API responses validated and consistent")
            logger.info("✅ Historical data sources working correctly")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed - review implementation")

        return passed == total


async def main():
    """Run comprehensive DexScreener tests."""
    async with DexScreenerRealDataTests() as tester:
        success = await tester.run_all_tests()
        return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)