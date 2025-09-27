"""API endpoints for portfolio balance and risk management."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator

from app.services.risk_metrics import risk_calculator
from app.services.explainability import explainability_engine
from app.db.connection import db_manager
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class PortfolioWeights(BaseModel):
    weights: Dict[str, float] = Field(..., description="Asset weights as symbol: weight pairs")

    @validator('weights')
    def validate_weights(cls, v):
        if not v:
            raise ValueError("Portfolio weights cannot be empty")

        # Check weight sum
        total_weight = sum(v.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight:.3f}")

        # Check individual weights
        for symbol, weight in v.items():
            if weight < 0:
                raise ValueError(f"Negative weight not allowed for {symbol}")
            if weight > 1:
                raise ValueError(f"Weight cannot exceed 1.0 for {symbol}")

        return v


class RiskMetricsRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    lookback_days: int = Field(90, ge=30, le=365, description="Historical data lookback period")
    confidence_levels: Optional[List[float]] = Field(None, description="VaR confidence levels")
    include_explanation: bool = Field(True, description="Include AI-generated risk explanation")


class RiskMetricsResponse(BaseModel):
    portfolio_id: str
    calculation_date: str
    lookback_days: int
    portfolio_weights: Dict[str, float]
    metrics: Dict
    data_quality: Dict
    explanation: Optional[Dict] = None


class PortfolioBalanceEntry(BaseModel):
    symbol: str
    balance: float
    usd_value: Optional[float] = None
    wallet_address: Optional[str] = None
    source: str = "manual"


class UpdateBalancesRequest(BaseModel):
    balances: List[PortfolioBalanceEntry]
    timestamp: Optional[str] = None


class OptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=2, description="Assets to include in optimization")
    lookback_days: int = Field(90, ge=30, le=365)
    target_return: Optional[float] = Field(None, description="Target annual return")
    max_single_weight: Optional[float] = Field(None, ge=0, le=1, description="Maximum weight per asset")


class OptimizationResponse(BaseModel):
    optimization_success: bool
    optimal_weights: Optional[Dict[str, float]] = None
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    error: Optional[str] = None


class PortfolioPerformanceResponse(BaseModel):
    portfolio_weights: Dict[str, float]
    performance_period_days: int
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    daily_returns: List[Dict]


@router.post("/risk-metrics", response_model=RiskMetricsResponse)
async def calculate_portfolio_risk(request: RiskMetricsRequest):
    """Calculate comprehensive risk metrics for a portfolio."""
    try:
        # Validate symbols
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        invalid_symbols = [s for s in request.portfolio_weights.keys() if s not in all_symbols]
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported symbols: {invalid_symbols}"
            )

        # Calculate risk metrics
        risk_result = await risk_calculator.calculate_portfolio_metrics(
            portfolio_weights=request.portfolio_weights,
            lookback_days=request.lookback_days,
            confidence_levels=request.confidence_levels
        )

        response = RiskMetricsResponse(**risk_result)

        # Add explanation if requested
        if request.include_explanation:
            try:
                explanation = await explainability_engine.explain_risk_metrics(
                    portfolio_weights=request.portfolio_weights,
                    risk_metrics=risk_result["metrics"]
                )
                response.explanation = explanation

            except Exception as e:
                logger.warning(f"Risk explanation generation failed: {e}")
                response.explanation = {"error": "Explanation unavailable", "details": str(e)}

        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Risk calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")


@router.post("/update-balances")
async def update_portfolio_balances(request: UpdateBalancesRequest):
    """Update portfolio balances."""
    try:
        # Parse timestamp or use current time
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()

        # Validate balances
        for balance in request.balances:
            if balance.balance < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Negative balance not allowed for {balance.symbol}"
                )

        # Get current prices for USD valuation
        latest_prices = await db_manager.get_latest_prices()
        price_map = {price["symbol"]: price["price"] for price in latest_prices}

        # Insert balance records
        balance_records = []
        for balance in request.balances:
            usd_value = balance.usd_value
            if usd_value is None and balance.symbol in price_map:
                usd_value = balance.balance * price_map[balance.symbol]

            balance_records.append({
                "time": timestamp,
                "wallet_address": balance.wallet_address or "default",
                "symbol": balance.symbol,
                "balance": balance.balance,
                "usd_value": usd_value,
                "source": balance.source
            })

        # Bulk insert using raw query
        if balance_records:
            query = """
            INSERT INTO portfolio_balances (time, wallet_address, symbol, balance, usd_value, source)
            VALUES ($1, $2, $3, $4, $5, $6)
            """

            for record in balance_records:
                await db_manager.execute_raw_query(query, record)

        return {
            "message": f"Updated {len(balance_records)} balance records",
            "timestamp": timestamp.isoformat(),
            "total_usd_value": sum(r.get("usd_value", 0) or 0 for r in balance_records)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Balance update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_portfolio():
    """Get current portfolio balances and weights."""
    try:
        # Get latest balances
        query = """
        SELECT DISTINCT ON (symbol) symbol, balance, usd_value, time, wallet_address
        FROM portfolio_balances
        ORDER BY symbol, time DESC
        """

        balances = await db_manager.execute_raw_query(query)

        if not balances:
            return {
                "portfolio": {},
                "total_usd_value": 0,
                "last_updated": None,
                "message": "No portfolio data found"
            }

        # Calculate total USD value and weights
        total_usd_value = sum(balance.get("usd_value", 0) or 0 for balance in balances)

        portfolio_weights = {}
        portfolio_balances = {}

        for balance in balances:
            symbol = balance["symbol"]
            usd_value = balance.get("usd_value", 0) or 0

            portfolio_balances[symbol] = {
                "balance": balance["balance"],
                "usd_value": usd_value,
                "last_updated": balance["time"].isoformat()
            }

            if total_usd_value > 0:
                portfolio_weights[symbol] = usd_value / total_usd_value

        # Get latest update time
        latest_update = max(balance["time"] for balance in balances) if balances else None

        return {
            "portfolio_weights": portfolio_weights,
            "portfolio_balances": portfolio_balances,
            "total_usd_value": total_usd_value,
            "last_updated": latest_update.isoformat() if latest_update else None,
            "asset_count": len(balances)
        }

    except Exception as e:
        logger.error(f"Current portfolio query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_portfolio_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    wallet_address: Optional[str] = Query(None, description="Specific wallet address")
):
    """Get historical portfolio balances."""
    try:
        where_clause = "WHERE time >= NOW() - INTERVAL '%s days'" % days
        params = {"days": days}

        if wallet_address:
            where_clause += " AND wallet_address = $2"
            params["wallet_address"] = wallet_address

        query = f"""
        SELECT time, symbol, balance, usd_value, wallet_address
        FROM portfolio_balances
        {where_clause}
        ORDER BY time DESC, symbol
        """

        history = await db_manager.execute_raw_query(query, params)

        # Group by date for easier frontend consumption
        grouped_history = {}
        for record in history:
            date_key = record["time"].date().isoformat()
            if date_key not in grouped_history:
                grouped_history[date_key] = []

            grouped_history[date_key].append({
                "symbol": record["symbol"],
                "balance": record["balance"],
                "usd_value": record.get("usd_value"),
                "wallet_address": record.get("wallet_address")
            })

        return {
            "history_days": days,
            "wallet_address": wallet_address,
            "portfolio_history": grouped_history,
            "total_records": len(history)
        }

    except Exception as e:
        logger.error(f"Portfolio history query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """Calculate optimal portfolio weights using Modern Portfolio Theory."""
    try:
        # Validate symbols
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        invalid_symbols = [s for s in request.symbols if s not in all_symbols]
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported symbols: {invalid_symbols}"
            )

        # Perform optimization
        optimization_result = await risk_calculator.calculate_optimal_portfolio(
            symbols=request.symbols,
            lookback_days=request.lookback_days,
            target_return=request.target_return
        )

        return OptimizationResponse(**optimization_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_portfolio_performance(
    portfolio_weights: str = Query(..., description="JSON string of portfolio weights"),
    days: int = Query(30, ge=7, le=365, description="Performance analysis period")
) -> PortfolioPerformanceResponse:
    """Calculate portfolio performance metrics over a specified period."""
    try:
        import json
        import pandas as pd
        import numpy as np

        # Parse portfolio weights
        try:
            weights = json.loads(portfolio_weights)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for portfolio_weights")

        # Validate weights
        if not isinstance(weights, dict) or not weights:
            raise HTTPException(status_code=400, detail="Portfolio weights must be a non-empty dictionary")

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Portfolio weights must sum to 1.0")

        # Get price data for all assets
        price_data = {}
        for symbol, weight in weights.items():
            if weight > 0:
                history = await db_manager.get_price_history(symbol, days + 10)  # Extra buffer
                if history:
                    df = pd.DataFrame(history)
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.sort_values('time').set_index('time')
                    price_data[symbol] = df['price']

        if not price_data:
            raise HTTPException(status_code=404, detail="No price data available for portfolio assets")

        # Combine price data and calculate returns
        combined_prices = pd.DataFrame(price_data).dropna()

        if len(combined_prices) < days:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: only {len(combined_prices)} days available, {days} requested"
            )

        # Take last N days
        combined_prices = combined_prices.tail(days)
        returns = combined_prices.pct_change().dropna()

        # Calculate portfolio returns
        weights_series = pd.Series(weights)
        portfolio_returns = (returns * weights_series).sum(axis=1)

        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
        volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = portfolio_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Daily returns for charting
        daily_returns = [
            {
                "date": date.isoformat(),
                "return": float(ret),
                "cumulative_return": float(cum_ret - 1)
            }
            for date, ret, cum_ret in zip(
                portfolio_returns.index,
                portfolio_returns.values,
                cumulative_returns.values
            )
        ]

        return PortfolioPerformanceResponse(
            portfolio_weights=weights,
            performance_period_days=days,
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            daily_returns=daily_returns
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-summary")
async def get_risk_summary(
    portfolio_weights: str = Query(..., description="JSON string of portfolio weights")
):
    """Get a quick risk summary for a portfolio."""
    try:
        import json

        # Parse portfolio weights
        try:
            weights = json.loads(portfolio_weights)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for portfolio_weights")

        # Quick risk calculation with default parameters
        risk_result = await risk_calculator.calculate_portfolio_metrics(
            portfolio_weights=weights,
            lookback_days=30,  # Quick analysis
            confidence_levels=[0.95]
        )

        # Extract key risk indicators
        metrics = risk_result["metrics"]
        risk_summary = {
            "overall_risk_level": "medium",
            "key_metrics": {
                "volatility": metrics.get("annualized_volatility", 0),
                "var_95": metrics.get("var_95", {}).get("historical", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "concentration": metrics.get("concentration_ratio", 0)
            },
            "risk_alerts": []
        }

        # Determine risk level and alerts
        if metrics.get("annualized_volatility", 0) > 0.4:
            risk_summary["overall_risk_level"] = "high"
            risk_summary["risk_alerts"].append("High volatility detected")

        if metrics.get("concentration_ratio", 0) > 0.5:
            risk_summary["risk_alerts"].append("High concentration risk")

        var_95 = metrics.get("var_95", {}).get("historical", 0)
        if var_95 < -0.05:
            risk_summary["risk_alerts"].append("High Value at Risk")

        if metrics.get("sharpe_ratio", 0) < 0.5:
            risk_summary["risk_alerts"].append("Poor risk-adjusted returns")

        if not risk_summary["risk_alerts"]:
            risk_summary["overall_risk_level"] = "low"

        return risk_summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk summary calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/balances")
async def clear_portfolio_balances(
    confirm: bool = Query(False, description="Confirmation flag"),
    wallet_address: Optional[str] = Query(None, description="Specific wallet to clear")
):
    """Clear portfolio balances (admin function)."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear balances"
        )

    try:
        if wallet_address:
            query = "DELETE FROM portfolio_balances WHERE wallet_address = $1"
            params = {"wallet_address": wallet_address}
        else:
            query = "DELETE FROM portfolio_balances"
            params = {}

        await db_manager.execute_raw_query(query, params)

        return {
            "message": f"Portfolio balances cleared{'for wallet ' + wallet_address if wallet_address else ''}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Balance clearing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))