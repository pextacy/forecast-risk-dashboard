"""API endpoints for hedge suggestions and risk management."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator

from app.services.hedge import hedge_engine
from app.services.explainability import explainability_engine
from app.db.connection import db_manager
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class HedgeRequest(BaseModel):
    portfolio_weights: Dict[str, float] = Field(..., description="Current portfolio allocation")
    risk_tolerance: str = Field("medium", description="Risk tolerance: conservative, medium, aggressive")
    include_explanation: bool = Field(True, description="Include AI-generated explanation")

    @validator('portfolio_weights')
    def validate_weights(cls, v):
        if not v:
            raise ValueError("Portfolio weights cannot be empty")

        total_weight = sum(v.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight:.3f}")

        return v

    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        if v not in ["conservative", "medium", "aggressive"]:
            raise ValueError("Risk tolerance must be: conservative, medium, or aggressive")
        return v


class HedgeResponse(BaseModel):
    portfolio_id: str
    generated_at: str
    current_allocation: Dict[str, float]
    risk_assessment: Dict
    recommended_action: Dict
    alternative_strategies: List[Dict]
    implementation_cost: float
    confidence_score: float
    market_context: Dict
    explanation: Optional[Dict] = None


class SimulateHedgeRequest(BaseModel):
    current_allocation: Dict[str, float]
    suggested_allocation: Dict[str, float]
    simulation_days: int = Field(30, ge=7, le=90, description="Days to simulate")


class SimulateHedgeResponse(BaseModel):
    simulation_period: int
    current_portfolio_performance: Dict
    suggested_portfolio_performance: Dict
    improvement_metrics: Dict
    risk_comparison: Dict


class BacktestRequest(BaseModel):
    lookback_days: int = Field(90, ge=30, le=365, description="Backtesting period")
    portfolio_symbols: List[str] = Field(..., description="Symbols to include in backtest")


class BacktestResponse(BaseModel):
    backtest_period: int
    total_suggestions: int
    average_improvement: float
    success_rate: float
    detailed_results: List[Dict]


class RebalanceAlert(BaseModel):
    alert_type: str
    severity: str  # low, medium, high
    message: str
    recommended_action: Optional[str] = None
    trigger_metric: Dict


@router.post("/suggest", response_model=HedgeResponse)
async def generate_hedge_suggestion(request: HedgeRequest):
    """Generate hedge suggestion for a portfolio."""
    try:
        # Validate symbols
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        invalid_symbols = [s for s in request.portfolio_weights.keys() if s not in all_symbols]
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported symbols: {invalid_symbols}"
            )

        # Generate hedge suggestion
        hedge_result = await hedge_engine.generate_hedge_suggestions(
            portfolio_weights=request.portfolio_weights,
            risk_tolerance=request.risk_tolerance
        )

        response = HedgeResponse(**hedge_result)

        # Add explanation if requested
        if request.include_explanation:
            try:
                explanation = await explainability_engine.explain_hedge_suggestion(
                    current_allocation=request.portfolio_weights,
                    suggested_allocation=hedge_result["recommended_action"]["suggested_allocation"],
                    rationale_data={
                        "expected_risk_reduction": hedge_result.get("implementation_cost", 0),
                        "implementation_cost": hedge_result.get("implementation_cost", 0),
                        "confidence_score": hedge_result.get("confidence_score", 0)
                    }
                )
                response.explanation = explanation

            except Exception as e:
                logger.warning(f"Hedge explanation generation failed: {e}")
                response.explanation = {"error": "Explanation unavailable", "details": str(e)}

        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Hedge suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hedge suggestion failed: {str(e)}")


@router.post("/simulate", response_model=SimulateHedgeResponse)
async def simulate_hedge_impact(request: SimulateHedgeRequest):
    """Simulate the impact of a hedge suggestion."""
    try:
        import pandas as pd
        import numpy as np

        # Get historical data for simulation
        all_symbols = set(request.current_allocation.keys()) | set(request.suggested_allocation.keys())

        price_data = {}
        for symbol in all_symbols:
            history = await db_manager.get_price_history(symbol, request.simulation_days + 10)
            if history:
                df = pd.DataFrame(history)
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').set_index('time')
                price_data[symbol] = df['price']

        if not price_data:
            raise HTTPException(status_code=404, detail="No price data available for simulation")

        # Combine price data
        combined_prices = pd.DataFrame(price_data).dropna()

        if len(combined_prices) < request.simulation_days:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {request.simulation_days}-day simulation"
            )

        # Take last N days for simulation
        simulation_prices = combined_prices.tail(request.simulation_days)
        returns = simulation_prices.pct_change().dropna()

        # Calculate performance for both portfolios
        current_performance = _calculate_portfolio_simulation(
            returns, request.current_allocation, "Current Portfolio"
        )

        suggested_performance = _calculate_portfolio_simulation(
            returns, request.suggested_allocation, "Suggested Portfolio"
        )

        # Calculate improvements
        improvement_metrics = {
            "return_improvement": suggested_performance["total_return"] - current_performance["total_return"],
            "volatility_improvement": current_performance["volatility"] - suggested_performance["volatility"],
            "sharpe_improvement": suggested_performance["sharpe_ratio"] - current_performance["sharpe_ratio"],
            "max_drawdown_improvement": current_performance["max_drawdown"] - suggested_performance["max_drawdown"]
        }

        # Risk comparison
        risk_comparison = {
            "current_risk_score": _calculate_risk_score(current_performance),
            "suggested_risk_score": _calculate_risk_score(suggested_performance),
            "risk_reduction": _calculate_risk_score(current_performance) - _calculate_risk_score(suggested_performance)
        }

        return SimulateHedgeResponse(
            simulation_period=request.simulation_days,
            current_portfolio_performance=current_performance,
            suggested_portfolio_performance=suggested_performance,
            improvement_metrics=improvement_metrics,
            risk_comparison=risk_comparison
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hedge simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_rebalance_alerts(
    portfolio_weights: str = Query(..., description="JSON string of portfolio weights"),
    risk_tolerance: str = Query("medium", description="Risk tolerance level")
) -> List[RebalanceAlert]:
    """Get rebalancing alerts based on current portfolio state."""
    try:
        import json

        # Parse portfolio weights
        try:
            weights = json.loads(portfolio_weights)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for portfolio_weights")

        alerts = []

        # Calculate current risk metrics
        from app.services.risk_metrics import risk_calculator
        risk_result = await risk_calculator.calculate_portfolio_metrics(
            portfolio_weights=weights,
            lookback_days=30
        )

        metrics = risk_result["metrics"]

        # Check for various alert conditions

        # High volatility alert
        volatility = metrics.get("annualized_volatility", 0)
        if volatility > 0.4:
            alerts.append(RebalanceAlert(
                alert_type="high_volatility",
                severity="high" if volatility > 0.6 else "medium",
                message=f"Portfolio volatility ({volatility*100:.1f}%) exceeds recommended levels",
                recommended_action="Consider reducing exposure to volatile assets",
                trigger_metric={"volatility": volatility, "threshold": 0.4}
            ))

        # Concentration risk alert
        concentration = metrics.get("concentration_ratio", 0)
        if concentration > 0.5:
            alerts.append(RebalanceAlert(
                alert_type="concentration_risk",
                severity="high" if concentration > 0.7 else "medium",
                message=f"High concentration in single asset ({concentration*100:.1f}%)",
                recommended_action="Diversify holdings across more assets",
                trigger_metric={"concentration": concentration, "threshold": 0.5}
            ))

        # VaR alert
        var_95 = metrics.get("var_95", {}).get("historical", 0)
        if var_95 < -0.05:
            alerts.append(RebalanceAlert(
                alert_type="high_var",
                severity="high" if var_95 < -0.1 else "medium",
                message=f"High Value at Risk ({abs(var_95)*100:.1f}% daily VaR)",
                recommended_action="Implement risk reduction strategies",
                trigger_metric={"var_95": var_95, "threshold": -0.05}
            ))

        # Poor Sharpe ratio alert
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < 0.5:
            alerts.append(RebalanceAlert(
                alert_type="poor_risk_adjusted_returns",
                severity="medium" if sharpe < 0.2 else "low",
                message=f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})",
                recommended_action="Review asset selection and risk management",
                trigger_metric={"sharpe_ratio": sharpe, "threshold": 0.5}
            ))

        # High correlation alert
        avg_correlation = metrics.get("average_correlation", 0)
        if avg_correlation > 0.7:
            alerts.append(RebalanceAlert(
                alert_type="high_correlation",
                severity="medium",
                message=f"High average correlation ({avg_correlation:.1%}) reduces diversification benefits",
                recommended_action="Add uncorrelated assets to portfolio",
                trigger_metric={"correlation": avg_correlation, "threshold": 0.7}
            ))

        # Drawdown alert
        max_drawdown = metrics.get("max_drawdown", 0)
        if max_drawdown < -0.2:
            alerts.append(RebalanceAlert(
                alert_type="high_drawdown",
                severity="high" if max_drawdown < -0.3 else "medium",
                message=f"Significant maximum drawdown ({abs(max_drawdown)*100:.1f}%)",
                recommended_action="Implement drawdown protection measures",
                trigger_metric={"max_drawdown": max_drawdown, "threshold": -0.2}
            ))

        return alerts

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_hedge_suggestion_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get historical hedge suggestions."""
    try:
        query = """
        SELECT id, created_at, risk_level, suggestion_type, current_allocation,
               suggested_allocation, rationale, expected_risk_reduction,
               implementation_cost, confidence_score
        FROM hedge_suggestions
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """

        suggestions = await db_manager.execute_raw_query(query, {
            "limit": limit,
            "offset": offset
        })

        return {
            "hedge_suggestions": suggestions,
            "total_returned": len(suggestions),
            "query_parameters": {"limit": limit, "offset": offset}
        }

    except Exception as e:
        logger.error(f"Hedge history query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
async def backtest_hedge_performance(request: BacktestRequest):
    """Backtest hedge suggestion performance."""
    try:
        # Validate symbols
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        invalid_symbols = [s for s in request.portfolio_symbols if s not in all_symbols]
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported symbols: {invalid_symbols}"
            )

        # Get historical suggestions for backtesting
        cutoff_date = datetime.now() - timedelta(days=request.lookback_days)
        query = """
        SELECT created_at, current_allocation, suggested_allocation,
               expected_risk_reduction, implementation_cost
        FROM hedge_suggestions
        WHERE created_at >= $1
        ORDER BY created_at ASC
        """

        historical_suggestions = await db_manager.execute_raw_query(query, {
            "cutoff_date": cutoff_date
        })

        if not historical_suggestions:
            raise HTTPException(
                status_code=404,
                detail="No historical hedge suggestions found for backtesting period"
            )

        # Perform backtesting
        backtest_result = await hedge_engine.backtest_hedge_performance(
            historical_suggestions, request.lookback_days
        )

        if "error" in backtest_result:
            raise HTTPException(status_code=400, detail=backtest_result["error"])

        return BacktestResponse(**backtest_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hedge backtesting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_available_strategies():
    """Get information about available hedge strategies."""
    strategies = [
        {
            "name": "Volatility Targeting",
            "description": "Maintain portfolio volatility around target level",
            "parameters": {"target_volatility": "15% annual"},
            "use_case": "Stable risk management"
        },
        {
            "name": "Concentration Risk Management",
            "description": "Limit single asset exposure",
            "parameters": {"max_single_weight": "35%"},
            "use_case": "Diversification enforcement"
        },
        {
            "name": "Correlation-Based Hedging",
            "description": "Reduce exposure during high correlation periods",
            "parameters": {"correlation_threshold": "70%"},
            "use_case": "Crisis period protection"
        },
        {
            "name": "VaR-Based Risk Management",
            "description": "Reduce exposure when VaR exceeds threshold",
            "parameters": {"var_threshold": "3% daily"},
            "use_case": "Downside risk control"
        }
    ]

    return {
        "available_strategies": strategies,
        "default_risk_tolerances": ["conservative", "medium", "aggressive"],
        "supported_actions": ["maintain", "rebalance", "diversify", "risk_reduction"]
    }


@router.post("/execute-suggestion")
async def execute_hedge_suggestion(
    suggestion_id: int,
    dry_run: bool = Query(True, description="If true, only simulate execution"),
    background_tasks: BackgroundTasks = None
):
    """Execute a hedge suggestion (simulation or actual)."""
    try:
        # Get suggestion details
        query = """
        SELECT id, current_allocation, suggested_allocation, rationale,
               implementation_cost, confidence_score
        FROM hedge_suggestions
        WHERE id = $1
        """

        suggestion = await db_manager.execute_raw_query(query, {"suggestion_id": suggestion_id})

        if not suggestion:
            raise HTTPException(status_code=404, detail="Hedge suggestion not found")

        suggestion = suggestion[0]

        if dry_run:
            # Simulate execution
            execution_plan = _create_execution_plan(
                suggestion["current_allocation"],
                suggestion["suggested_allocation"]
            )

            return {
                "execution_type": "simulation",
                "suggestion_id": suggestion_id,
                "execution_plan": execution_plan,
                "estimated_cost": suggestion["implementation_cost"],
                "confidence_score": suggestion["confidence_score"],
                "message": "Dry run completed successfully"
            }

        else:
            # Actual execution would go here
            # For now, just log the execution
            async def log_execution():
                logger.info(f"Executing hedge suggestion {suggestion_id}")
                # In a real system, this would integrate with trading APIs

            if background_tasks:
                background_tasks.add_task(log_execution)

            return {
                "execution_type": "actual",
                "suggestion_id": suggestion_id,
                "status": "queued",
                "message": "Hedge suggestion execution queued",
                "timestamp": datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hedge execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _calculate_portfolio_simulation(returns, weights, portfolio_name):
    """Calculate portfolio simulation metrics."""
    import pandas as pd
    import numpy as np

    # Calculate portfolio returns
    weights_series = pd.Series(weights)

    # Only use symbols that exist in returns data
    valid_weights = {k: v for k, v in weights.items() if k in returns.columns}
    if not valid_weights:
        raise ValueError(f"No valid symbols found in returns data for {portfolio_name}")

    # Normalize weights for valid symbols
    total_valid_weight = sum(valid_weights.values())
    if total_valid_weight > 0:
        valid_weights = {k: v/total_valid_weight for k, v in valid_weights.items()}

    weights_series = pd.Series(valid_weights)
    portfolio_returns = (returns[list(valid_weights.keys())] * weights_series).sum(axis=1)

    # Calculate metrics
    total_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)

    # Sharpe ratio
    risk_free_rate = 0.02 / 252  # Daily risk-free rate
    excess_returns = portfolio_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

    # Max drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "portfolio_name": portfolio_name,
        "total_return": float(total_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "final_value": float(cumulative_returns.iloc[-1])
    }


def _calculate_risk_score(performance_metrics):
    """Calculate a composite risk score."""
    volatility = performance_metrics.get("volatility", 0)
    max_drawdown = abs(performance_metrics.get("max_drawdown", 0))
    sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)

    # Higher score = higher risk
    risk_score = (volatility * 100) + (max_drawdown * 50) - (sharpe_ratio * 20)
    return max(0, min(100, risk_score))  # Clamp between 0-100


def _create_execution_plan(current_allocation, suggested_allocation):
    """Create detailed execution plan for portfolio rebalancing."""
    execution_steps = []

    all_symbols = set(current_allocation.keys()) | set(suggested_allocation.keys())

    for symbol in all_symbols:
        current_weight = current_allocation.get(symbol, 0)
        suggested_weight = suggested_allocation.get(symbol, 0)
        change = suggested_weight - current_weight

        if abs(change) > 0.01:  # 1% threshold
            action = "increase" if change > 0 else "decrease"
            execution_steps.append({
                "symbol": symbol,
                "action": action,
                "current_weight": current_weight,
                "target_weight": suggested_weight,
                "change": change,
                "change_percent": change * 100
            })

    return {
        "total_steps": len(execution_steps),
        "steps": execution_steps,
        "estimated_duration": "5-10 minutes",
        "risk_level": "low"
    }