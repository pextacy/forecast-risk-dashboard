"""API endpoints for real-time market data - REAL DATA ONLY."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.dexscreener_client import DexScreenerClient
from app.services.ingestion import DataIngestionService
from app.db.connection import db_manager
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Response models
class TokenPriceData(BaseModel):
    symbol: str
    name: str
    price_usd: float
    market_cap: Optional[float] = None
    volume_24h: float
    price_change_24h: float
    liquidity_usd: float
    dex_id: str
    last_updated: str


class PortfolioOverview(BaseModel):
    total_value_usd: float
    total_change_24h: float
    asset_count: int
    last_updated: str
    assets: List[TokenPriceData]


class MarketOverviewResponse(BaseModel):
    market_data: List[TokenPriceData]
    market_summary: Dict
    data_sources: List[str]
    last_updated: str


class RiskMetrics(BaseModel):
    symbol: str
    volatility_7d: float
    volatility_30d: float
    var_95: float
    var_99: float
    correlation_btc: Optional[float] = None
    correlation_eth: Optional[float] = None


class DashboardData(BaseModel):
    portfolio_value: float
    portfolio_change_24h: float
    risk_level: str
    active_forecasts: int
    market_trend: str
    top_performers: List[TokenPriceData]
    risk_alerts: List[str]
    last_updated: str


@router.get("/current-prices", response_model=List[TokenPriceData])
async def get_current_prices():
    """Get current real-time prices from DexScreener."""
    try:
        async with DexScreenerClient() as client:
            # Get real prices for major tokens
            symbols = ["ethereum", "bitcoin", "uniswap", "chainlink", "aave"]
            current_prices = await client.get_current_prices(symbols)

            token_data = []
            for symbol, data in current_prices.items():
                if data and data.get("usd", 0) > 0:
                    token_data.append(TokenPriceData(
                        symbol=symbol.upper(),
                        name=symbol.capitalize(),
                        price_usd=float(data["usd"]),
                        market_cap=data.get("usd_market_cap"),
                        volume_24h=float(data.get("usd_24h_vol", 0)),
                        price_change_24h=float(data.get("usd_24h_change", 0)),
                        liquidity_usd=float(data.get("liquidity_usd", 0)),
                        dex_id=data.get("dex_info", {}).get("dex_id", "unknown"),
                        last_updated=datetime.fromtimestamp(data["last_updated_at"]).isoformat()
                    ))

        if not token_data:
            raise HTTPException(status_code=503, detail="No real market data available")

        logger.info(f"Served real-time prices for {len(token_data)} tokens")
        return token_data

    except Exception as e:
        logger.error(f"Failed to fetch current prices: {e}")
        raise HTTPException(status_code=500, detail=f"Market data unavailable: {str(e)}")


@router.get("/market-overview", response_model=MarketOverviewResponse)
async def get_market_overview():
    """Get comprehensive market overview with real data."""
    try:
        async with DexScreenerClient() as client:
            # Get market data for major assets
            symbols = ["ethereum", "bitcoin", "uniswap", "chainlink", "aave", "usdt"]
            market_data_raw = await client.get_market_data(symbols)

            market_data = []
            total_volume = 0
            positive_change_count = 0

            for data in market_data_raw:
                if data and data.get("current_price", 0) > 0:
                    token_price = TokenPriceData(
                        symbol=data["symbol"],
                        name=data.get("name", data["symbol"]),
                        price_usd=float(data["current_price"]),
                        market_cap=data.get("market_cap"),
                        volume_24h=float(data.get("total_volume", 0)),
                        price_change_24h=float(data.get("price_change_percentage_24h", 0)),
                        liquidity_usd=float(data.get("liquidity_usd", 0)),
                        dex_id=data.get("dex_id", "unknown"),
                        last_updated=datetime.now().isoformat()
                    )
                    market_data.append(token_price)
                    total_volume += token_price.volume_24h

                    if token_price.price_change_24h > 0:
                        positive_change_count += 1

            # Calculate market summary
            market_summary = {
                "total_volume_24h": total_volume,
                "assets_up": positive_change_count,
                "assets_down": len(market_data) - positive_change_count,
                "market_sentiment": "bullish" if positive_change_count > len(market_data) / 2 else "bearish",
                "average_change": sum(t.price_change_24h for t in market_data) / len(market_data) if market_data else 0
            }

            return MarketOverviewResponse(
                market_data=market_data,
                market_summary=market_summary,
                data_sources=["DexScreener", "Uniswap", "Balancer"],
                last_updated=datetime.now().isoformat()
            )

    except Exception as e:
        logger.error(f"Market overview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market overview unavailable: {str(e)}")


@router.get("/portfolio-overview", response_model=PortfolioOverview)
async def get_portfolio_overview(
    addresses: List[str] = Query([], description="Wallet addresses to analyze")
):
    """Get portfolio overview with real market data."""
    try:
        # For demo purposes, create a sample portfolio with real prices
        async with DexScreenerClient() as client:
            sample_holdings = {
                "ethereum": 10.5,   # 10.5 ETH
                "bitcoin": 0.25,    # 0.25 BTC
                "uniswap": 1000,    # 1000 UNI
                "chainlink": 500,   # 500 LINK
            }

            current_prices = await client.get_current_prices(list(sample_holdings.keys()))

            portfolio_assets = []
            total_value = 0
            total_change = 0

            for symbol, amount in sample_holdings.items():
                if symbol in current_prices and current_prices[symbol].get("usd", 0) > 0:
                    price_data = current_prices[symbol]
                    current_price = float(price_data["usd"])
                    position_value = amount * current_price
                    price_change = float(price_data.get("usd_24h_change", 0))

                    total_value += position_value
                    total_change += (position_value * price_change / 100)

                    portfolio_assets.append(TokenPriceData(
                        symbol=symbol.upper(),
                        name=symbol.capitalize(),
                        price_usd=current_price,
                        market_cap=price_data.get("usd_market_cap"),
                        volume_24h=float(price_data.get("usd_24h_vol", 0)),
                        price_change_24h=price_change,
                        liquidity_usd=float(price_data.get("liquidity_usd", 0)),
                        dex_id=price_data.get("dex_info", {}).get("dex_id", "unknown"),
                        last_updated=datetime.fromtimestamp(price_data["last_updated_at"]).isoformat()
                    ))

            portfolio_change_pct = (total_change / total_value * 100) if total_value > 0 else 0

            return PortfolioOverview(
                total_value_usd=total_value,
                total_change_24h=portfolio_change_pct,
                asset_count=len(portfolio_assets),
                last_updated=datetime.now().isoformat(),
                assets=portfolio_assets
            )

    except Exception as e:
        logger.error(f"Portfolio overview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio data unavailable: {str(e)}")


@router.get("/risk-metrics", response_model=List[RiskMetrics])
async def get_risk_metrics():
    """Get real risk metrics calculated from actual market data."""
    try:
        symbols = ["ethereum", "bitcoin", "uniswap", "chainlink"]
        risk_metrics = []

        ingestion_service = DataIngestionService()

        for symbol in symbols:
            try:
                # Calculate real volatilities
                vol_7d = await ingestion_service.calculate_real_volatility(symbol, 7)
                vol_30d = await ingestion_service.calculate_real_volatility(symbol, 30)

                if vol_7d is not None and vol_30d is not None:
                    # Get current price for VaR calculation
                    async with DexScreenerClient() as client:
                        current_prices = await client.get_current_prices([symbol])
                        current_price = 0

                        if symbol in current_prices:
                            current_price = float(current_prices[symbol].get("usd", 0))

                    # Calculate VaR (Value at Risk)
                    if current_price > 0:
                        var_95 = current_price * vol_30d * 1.645 / (252 ** 0.5)  # 95% confidence
                        var_99 = current_price * vol_30d * 2.326 / (252 ** 0.5)  # 99% confidence

                        risk_metrics.append(RiskMetrics(
                            symbol=symbol.upper(),
                            volatility_7d=vol_7d,
                            volatility_30d=vol_30d,
                            var_95=var_95,
                            var_99=var_99
                        ))

            except Exception as e:
                logger.warning(f"Risk metrics calculation failed for {symbol}: {e}")
                continue

        if not risk_metrics:
            raise HTTPException(status_code=503, detail="Risk metrics calculation failed")

        logger.info(f"Calculated real risk metrics for {len(risk_metrics)} assets")
        return risk_metrics

    except Exception as e:
        logger.error(f"Risk metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk metrics unavailable: {str(e)}")


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """Get comprehensive dashboard data with real market information."""
    try:
        # Get real market data
        market_overview = await get_market_overview()
        portfolio_overview = await get_portfolio_overview()
        risk_metrics = await get_risk_metrics()

        # Calculate portfolio metrics
        portfolio_value = portfolio_overview.total_value_usd
        portfolio_change = portfolio_overview.total_change_24h

        # Determine risk level based on real volatility
        avg_volatility = sum(r.volatility_30d for r in risk_metrics) / len(risk_metrics) if risk_metrics else 0.3
        if avg_volatility < 0.2:
            risk_level = "Low"
        elif avg_volatility < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Count active forecasts from database
        active_forecasts_query = """
        SELECT COUNT(DISTINCT symbol) as count
        FROM forecasts
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        """
        forecast_result = await db_manager.execute_raw_query(active_forecasts_query)
        active_forecasts = forecast_result[0]["count"] if forecast_result else 0

        # Determine market trend from real data
        positive_assets = sum(1 for asset in market_overview.market_data if asset.price_change_24h > 0)
        total_assets = len(market_overview.market_data)

        if positive_assets > total_assets * 0.6:
            market_trend = "Bullish"
        elif positive_assets < total_assets * 0.4:
            market_trend = "Bearish"
        else:
            market_trend = "Neutral"

        # Get top performers (real data)
        top_performers = sorted(
            market_overview.market_data,
            key=lambda x: x.price_change_24h,
            reverse=True
        )[:3]

        # Generate risk alerts based on real metrics
        risk_alerts = []
        for metric in risk_metrics:
            if metric.volatility_7d > 0.8:  # 80% volatility
                risk_alerts.append(f"High volatility detected in {metric.symbol}: {metric.volatility_7d:.1%}")
            if metric.var_95 > metric.volatility_30d * 1000:  # Large VaR
                risk_alerts.append(f"Elevated risk in {metric.symbol}: 95% VaR ${metric.var_95:.2f}")

        return DashboardData(
            portfolio_value=portfolio_value,
            portfolio_change_24h=portfolio_change,
            risk_level=risk_level,
            active_forecasts=active_forecasts,
            market_trend=market_trend,
            top_performers=top_performers,
            risk_alerts=risk_alerts[:3],  # Limit to 3 alerts
            last_updated=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Dashboard data compilation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data unavailable: {str(e)}")


@router.get("/search")
async def search_tokens(
    query: str = Query(..., min_length=2, description="Search query for tokens"),
    limit: int = Query(10, ge=1, le=50)
):
    """Search for tokens using real DexScreener data."""
    try:
        async with DexScreenerClient() as client:
            search_results = await client.search_tokens(query)

            # Format results for frontend
            formatted_results = []
            for result in search_results[:limit]:
                formatted_results.append({
                    "symbol": result["symbol"],
                    "name": result["name"],
                    "address": result["address"],
                    "price_usd": result["price_usd"],
                    "volume_24h": result["volume_24h"],
                    "liquidity_usd": result["liquidity_usd"],
                    "dex_id": result["dex_id"],
                    "chain_id": result["chain_id"]
                })

            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "data_source": "DexScreener",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Token search failed for '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search unavailable: {str(e)}")


@router.post("/refresh-data")
async def refresh_market_data(background_tasks: BackgroundTasks):
    """Manually refresh market data from all sources."""
    try:
        async def refresh_all_data():
            ingestion_service = DataIngestionService()
            try:
                crypto_count, stock_count = await ingestion_service.ingest_current_market_data()
                logger.info(f"Data refresh completed: {crypto_count} crypto + {stock_count} stock prices")
            except Exception as e:
                logger.error(f"Data refresh failed: {e}")

        background_tasks.add_task(refresh_all_data)

        return {
            "message": "Market data refresh initiated",
            "sources": ["DexScreener", "Binance", "Coinbase"],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Data refresh trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def market_data_health():
    """Check health of market data services."""
    try:
        health_status = {
            "services": {},
            "last_data_update": None,
            "data_staleness_hours": None,
            "overall_status": "healthy"
        }

        # Check DexScreener connectivity
        try:
            async with DexScreenerClient() as client:
                test_data = await client.get_current_prices(["ethereum"])
                health_status["services"]["dexscreener"] = "operational" if test_data else "degraded"
        except Exception as e:
            health_status["services"]["dexscreener"] = f"error: {str(e)}"

        # Check data freshness
        try:
            latest_prices = await db_manager.get_latest_prices()
            if latest_prices:
                latest_time = max(price["time"] for price in latest_prices)
                hours_ago = (datetime.now() - latest_time).total_seconds() / 3600
                health_status["last_data_update"] = latest_time.isoformat()
                health_status["data_staleness_hours"] = round(hours_ago, 1)

                if hours_ago > 6:
                    health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)

        return health_status

    except Exception as e:
        logger.error(f"Market data health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))