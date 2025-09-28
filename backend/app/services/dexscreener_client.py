"""
DexScreener API client for real-time DEX token data.
Uses DexScreener for accurate DeFi pricing from decentralized exchanges.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
from app.utils.config import settings, get_api_headers, get_dexscreener_url

logger = logging.getLogger(__name__)


class DexScreenerClient:
    """
    Real DexScreener API client for DeFi token data.
    Provides accurate pricing, volume, and liquidity data from DEX aggregators.
    """

    def __init__(self):
        self.base_url = get_dexscreener_url()
        self.headers = get_api_headers("dexscreener")
        self.session: Optional[aiohttp.ClientSession] = None

        # Popular DEX platforms and their chain IDs
        self.popular_dexes = {
            "ethereum": ["uniswap", "sushiswap", "curve"],
            "bsc": ["pancakeswap", "biswap"],
            "polygon": ["quickswap", "sushiswap"],
            "arbitrum": ["uniswap", "sushiswap"],
            "avalanche": ["traderjoe", "pangolin"]
        }

        # Token symbols to contract address mapping
        self.token_addresses = {
            # Ethereum mainnet tokens
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "USDC": "0xA0b86a33E6441c7C677AB43c16f0dB1CCa34884e",
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
            "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
            "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
            "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
            "COMP": "0xc00e94Cb662C3520282E6f5717214004A7f26888",
            "SUSHI": "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
            "CRV": "0xD533a949740bb3306d119CC777fa900bA034cd52",

            # Additional popular tokens
            "PEPE": "0x6982508145454Ce325dDbE47a25d4ec3d2311933",
            "SHIB": "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE",
            "MATIC": "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_token_info(self, token_address: str) -> Optional[Dict]:
        """Get basic token information."""
        url = f"{self.base_url}/dex/tokens/{token_address}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.warning(f"DexScreener token info failed: {response.status} for {token_address}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching token info for {token_address}: {e}")
            return None

    async def get_token_pairs(self, token_address: str, limit: int = 10) -> List[Dict]:
        """Get trading pairs for a specific token."""
        url = f"{self.base_url}/dex/tokens/{token_address}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])

                    # Sort by liquidity and volume
                    sorted_pairs = sorted(
                        pairs,
                        key=lambda x: (x.get("liquidity", {}).get("usd", 0), x.get("volume", {}).get("h24", 0)),
                        reverse=True
                    )

                    return sorted_pairs[:limit]
                else:
                    logger.warning(f"DexScreener pairs failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching pairs for {token_address}: {e}")
            return []

    async def get_current_prices(self, token_symbols: List[str]) -> Dict[str, dict]:
        """Fetch current prices for multiple tokens."""
        if not self.session:
            await self.__aenter__()

        prices = {}

        # Convert symbols to addresses
        addresses = []
        symbol_to_address = {}

        for symbol in token_symbols:
            # Handle different symbol formats
            clean_symbol = symbol.upper().replace("ETHEREUM", "WETH").replace("BITCOIN", "WBTC")

            if clean_symbol in self.token_addresses:
                address = self.token_addresses[clean_symbol]
                addresses.append(address)
                symbol_to_address[address] = symbol
            else:
                logger.warning(f"Token address not found for symbol: {symbol}")

        # Fetch data for each token
        for address in addresses:
            try:
                pairs = await self.get_token_pairs(address, 1)
                if pairs:
                    best_pair = pairs[0]
                    original_symbol = symbol_to_address[address]

                    prices[original_symbol] = {
                        "usd": best_pair.get("priceUsd", 0),
                        "usd_market_cap": best_pair.get("marketCap", 0),
                        "usd_24h_vol": best_pair.get("volume", {}).get("h24", 0),
                        "usd_24h_change": best_pair.get("priceChange", {}).get("h24", 0),
                        "last_updated_at": int(datetime.now().timestamp()),
                        "liquidity_usd": best_pair.get("liquidity", {}).get("usd", 0),
                        "dex_info": {
                            "dex_id": best_pair.get("dexId", ""),
                            "pair_address": best_pair.get("pairAddress", ""),
                            "chain_id": best_pair.get("chainId", "")
                        }
                    }
                else:
                    logger.warning(f"No pairs found for token: {original_symbol}")

            except Exception as e:
                logger.error(f"Error fetching price for {address}: {e}")

        logger.info(f"Fetched current prices for {len(prices)} tokens via DexScreener")
        return prices

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[dict]:
        """
        DexScreener doesn't provide historical data.
        This method returns empty list - use external historical data sources instead.
        """
        logger.warning(f"DexScreener does not provide historical data for {symbol}. Use Binance/Coinbase APIs instead.")
        return []

    async def get_trending_tokens(self, chain: str = "ethereum", limit: int = 10) -> List[Dict]:
        """Get trending tokens on a specific chain."""
        # DexScreener doesn't have a direct trending endpoint,
        # so we'll get popular pairs and extract trending tokens
        url = f"{self.base_url}/dex/search?q={chain}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])

                    # Sort by volume and recent activity
                    trending = sorted(
                        pairs,
                        key=lambda x: x.get("volume", {}).get("h24", 0),
                        reverse=True
                    )[:limit]

                    return trending
                else:
                    logger.warning(f"DexScreener trending failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching trending tokens: {e}")
            return []

    async def search_tokens(self, query: str) -> List[Dict]:
        """Search for tokens by name or symbol."""
        url = f"{self.base_url}/dex/search?q={query}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])

                    # Filter and format results
                    results = []
                    seen_tokens = set()

                    for pair in pairs[:20]:  # Limit results
                        base_token = pair.get("baseToken", {})
                        token_symbol = base_token.get("symbol", "")
                        token_address = base_token.get("address", "")

                        if token_address not in seen_tokens and token_symbol:
                            seen_tokens.add(token_address)
                            results.append({
                                "symbol": token_symbol,
                                "name": base_token.get("name", ""),
                                "address": token_address,
                                "price_usd": pair.get("priceUsd", 0),
                                "volume_24h": pair.get("volume", {}).get("h24", 0),
                                "liquidity_usd": pair.get("liquidity", {}).get("usd", 0),
                                "dex_id": pair.get("dexId", ""),
                                "chain_id": pair.get("chainId", "")
                            })

                    return results[:10]  # Return top 10 matches
                else:
                    logger.warning(f"DexScreener search failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error searching tokens for '{query}': {e}")
            return []

    async def get_pair_info(self, pair_address: str) -> Optional[Dict]:
        """Get detailed information about a specific trading pair."""
        url = f"{self.base_url}/dex/pairs/{pair_address}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pair = data.get("pair")
                    return pair
                else:
                    logger.warning(f"DexScreener pair info failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching pair info for {pair_address}: {e}")
            return None

    def get_symbol_address(self, symbol: str) -> Optional[str]:
        """Get contract address for a given symbol."""
        clean_symbol = symbol.upper().replace("ETHEREUM", "WETH").replace("BITCOIN", "WBTC")
        return self.token_addresses.get(clean_symbol)

    async def get_market_data(self, token_symbols: List[str]) -> List[dict]:
        """Fetch comprehensive market data including volatility metrics."""
        if not self.session:
            await self.__aenter__()

        market_data = []

        for symbol in token_symbols:
            try:
                pairs = await self.get_token_pairs(
                    self.get_symbol_address(symbol) or symbol, 1
                )

                if pairs:
                    pair = pairs[0]

                    # Extract REAL market data only (no estimates or mock data)
                    data = {
                        "id": symbol.lower(),
                        "symbol": symbol.upper(),
                        "name": pair.get("baseToken", {}).get("name", symbol),
                        "current_price": float(pair.get("priceUsd", 0)),
                        "market_cap": pair.get("marketCap", 0),
                        "total_volume": pair.get("volume", {}).get("h24", 0),
                        "high_24h": None,  # Not available from DexScreener
                        "low_24h": None,   # Not available from DexScreener
                        "price_change_24h": pair.get("priceChange", {}).get("h24", 0),
                        "price_change_percentage_24h": pair.get("priceChange", {}).get("h24", 0),
                        "market_cap_rank": None,  # Not available from DexScreener
                        "fully_diluted_valuation": pair.get("marketCap", 0),
                        "total_supply": None,  # Not available from DexScreener
                        "max_supply": None,    # Not available from DexScreener
                        "circulating_supply": None,  # Not available from DexScreener
                        "last_updated": datetime.now().isoformat(),

                        # DexScreener specific REAL data
                        "liquidity_usd": pair.get("liquidity", {}).get("usd", 0),
                        "dex_id": pair.get("dexId", ""),
                        "pair_address": pair.get("pairAddress", ""),
                        "chain_id": pair.get("chainId", ""),

                        # Real price change percentages (only what's available)
                        "price_change_percentage_1h": pair.get("priceChange", {}).get("h1", 0),
                        "price_change_percentage_7d": None,   # Not available from DexScreener
                        "price_change_percentage_30d": None,  # Not available from DexScreener
                    }

                    market_data.append(data)

            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")

        logger.info(f"Fetched market data for {len(market_data)} tokens")
        return market_data


# For backward compatibility with existing client interfaces
class DexScreenerAdapter:
    """Adapter to make DexScreener client compatible with existing client interface."""

    def __init__(self):
        self.client = DexScreenerClient()

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def get_current_prices(self, token_symbols: List[str]) -> Dict[str, dict]:
        """Get current prices using DexScreener."""
        return await self.client.get_current_prices(token_symbols)

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[dict]:
        """Get historical data (returns empty - use external sources)."""
        return await self.client.get_historical_data(symbol, days)

    async def get_market_data(self, token_symbols: List[str]) -> List[dict]:
        """Get market data using DexScreener."""
        return await self.client.get_market_data(token_symbols)