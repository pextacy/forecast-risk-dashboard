"""Intelligent hedge suggestion engine for portfolio risk management."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.db.connection import db_manager
from app.services.risk_metrics import risk_calculator
from app.services.forecasting import forecasting_service
from app.utils.config import settings

logger = logging.getLogger(__name__)


class HedgeStrategy:
    """Base class for hedge strategies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def generate_suggestion(
        self,
        current_portfolio: Dict[str, float],
        risk_metrics: Dict,
        market_data: Dict
    ) -> Dict:
        """Generate hedge suggestion based on strategy logic."""
        raise NotImplementedError


class VolatilityTargetingStrategy(HedgeStrategy):
    """Strategy to maintain target portfolio volatility."""

    def __init__(self, target_volatility: float = 0.15):
        super().__init__(
            "Volatility Targeting",
            f"Maintain portfolio volatility around {target_volatility*100:.0f}% annually"
        )
        self.target_volatility = target_volatility

    async def generate_suggestion(
        self,
        current_portfolio: Dict[str, float],
        risk_metrics: Dict,
        market_data: Dict
    ) -> Dict:
        """Generate volatility-targeting hedge suggestion."""
        current_volatility = risk_metrics.get("annualized_volatility", 0)

        if abs(current_volatility - self.target_volatility) < 0.02:  # 2% tolerance
            return {
                "action": "maintain",
                "rationale": f"Portfolio volatility ({current_volatility*100:.1f}%) is within target range",
                "suggested_allocation": current_portfolio.copy(),
                "expected_risk_reduction": 0.0
            }

        # Calculate adjustment needed
        volatility_ratio = self.target_volatility / current_volatility
        risk_adjustment = 1 - volatility_ratio

        # Generate new allocation
        suggested_allocation = {}
        stable_assets = ["USDC", "USDT", "usd-coin", "tether"]
        volatile_assets = [asset for asset in current_portfolio.keys() if asset not in stable_assets]

        if current_volatility > self.target_volatility:
            # Reduce volatility by moving to stable assets
            total_stable_weight = sum(current_portfolio.get(asset, 0) for asset in stable_assets)
            stable_increase = risk_adjustment * 0.5  # Move half the excess to stable assets

            for asset, weight in current_portfolio.items():
                if asset in stable_assets:
                    # Increase stable asset weights proportionally
                    if total_stable_weight > 0:
                        suggested_allocation[asset] = weight + (weight / total_stable_weight) * stable_increase
                    else:
                        suggested_allocation[asset] = weight + stable_increase / len(stable_assets)
                else:
                    # Reduce volatile asset weights
                    suggested_allocation[asset] = weight * (1 - stable_increase / len(volatile_assets))

        else:
            # Increase volatility by moving away from stable assets
            for asset, weight in current_portfolio.items():
                if asset in stable_assets:
                    suggested_allocation[asset] = weight * volatility_ratio
                else:
                    suggested_allocation[asset] = weight + (1 - volatility_ratio) * weight

        # Normalize weights
        total_weight = sum(suggested_allocation.values())
        if total_weight > 0:
            suggested_allocation = {k: v/total_weight for k, v in suggested_allocation.items()}

        expected_reduction = abs(current_volatility - self.target_volatility) / current_volatility

        return {
            "action": "rebalance",
            "rationale": f"Adjust volatility from {current_volatility*100:.1f}% to target {self.target_volatility*100:.1f}%",
            "suggested_allocation": suggested_allocation,
            "expected_risk_reduction": expected_reduction
        }


class ConcentrationRiskStrategy(HedgeStrategy):
    """Strategy to reduce concentration risk."""

    def __init__(self, max_single_weight: float = 0.35):
        super().__init__(
            "Concentration Risk Management",
            f"Limit single asset exposure to {max_single_weight*100:.0f}%"
        )
        self.max_single_weight = max_single_weight

    async def generate_suggestion(
        self,
        current_portfolio: Dict[str, float],
        risk_metrics: Dict,
        market_data: Dict
    ) -> Dict:
        """Generate concentration risk reduction suggestion."""
        max_weight = max(current_portfolio.values())
        max_asset = max(current_portfolio, key=current_portfolio.get)

        if max_weight <= self.max_single_weight:
            return {
                "action": "maintain",
                "rationale": f"Concentration risk is acceptable (max: {max_weight*100:.1f}%)",
                "suggested_allocation": current_portfolio.copy(),
                "expected_risk_reduction": 0.0
            }

        # Calculate excess concentration
        excess_weight = max_weight - self.max_single_weight

        # Redistribute excess weight to other assets
        suggested_allocation = current_portfolio.copy()
        suggested_allocation[max_asset] = self.max_single_weight

        # Distribute excess to other assets proportionally
        other_assets = {k: v for k, v in current_portfolio.items() if k != max_asset}
        total_other_weight = sum(other_assets.values())

        if total_other_weight > 0:
            for asset, weight in other_assets.items():
                proportion = weight / total_other_weight
                suggested_allocation[asset] = weight + excess_weight * proportion

        concentration_reduction = (max_weight - self.max_single_weight) / max_weight

        return {
            "action": "rebalance",
            "rationale": f"Reduce {max_asset} concentration from {max_weight*100:.1f}% to {self.max_single_weight*100:.1f}%",
            "suggested_allocation": suggested_allocation,
            "expected_risk_reduction": concentration_reduction * 0.5  # Partial risk reduction
        }


class CorrelationBasedStrategy(HedgeStrategy):
    """Strategy to reduce portfolio correlation during high correlation periods."""

    def __init__(self, correlation_threshold: float = 0.7):
        super().__init__(
            "Correlation-Based Hedging",
            f"Reduce exposure when average correlation exceeds {correlation_threshold:.0%}"
        )
        self.correlation_threshold = correlation_threshold

    async def generate_suggestion(
        self,
        current_portfolio: Dict[str, float],
        risk_metrics: Dict,
        market_data: Dict
    ) -> Dict:
        """Generate correlation-based hedge suggestion."""
        avg_correlation = risk_metrics.get("average_correlation", 0)

        if avg_correlation < self.correlation_threshold:
            return {
                "action": "maintain",
                "rationale": f"Portfolio correlation ({avg_correlation:.1%}) is within acceptable range",
                "suggested_allocation": current_portfolio.copy(),
                "expected_risk_reduction": 0.0
            }

        # High correlation detected - add uncorrelated assets
        suggested_allocation = current_portfolio.copy()

        # Define uncorrelated asset classes
        uncorrelated_assets = {
            "bonds": 0.1,  # Government bonds
            "gold": 0.05,  # Gold or gold-backed assets
            "stablecoins": 0.1  # Stablecoins for crypto portfolios
        }

        # Scale down existing positions
        scale_factor = 0.85  # Reduce by 15%
        for asset in suggested_allocation:
            suggested_allocation[asset] *= scale_factor

        # Add uncorrelated assets
        remaining_weight = 1 - sum(suggested_allocation.values())
        for asset_type, target_weight in uncorrelated_assets.items():
            if asset_type == "stablecoins":
                # Add stablecoin allocation
                stable_weight = min(target_weight, remaining_weight)
                if "USDC" in suggested_allocation:
                    suggested_allocation["USDC"] += stable_weight
                else:
                    suggested_allocation["USDC"] = stable_weight
                remaining_weight -= stable_weight

        # Normalize if needed
        total_weight = sum(suggested_allocation.values())
        if total_weight > 1.01:  # Allow small tolerance
            suggested_allocation = {k: v/total_weight for k, v in suggested_allocation.items()}

        correlation_reduction = (avg_correlation - self.correlation_threshold) / avg_correlation

        return {
            "action": "diversify",
            "rationale": f"Add uncorrelated assets to reduce correlation from {avg_correlation:.1%}",
            "suggested_allocation": suggested_allocation,
            "expected_risk_reduction": correlation_reduction * 0.3
        }


class VaRBasedStrategy(HedgeStrategy):
    """Strategy based on Value at Risk thresholds."""

    def __init__(self, var_threshold: float = -0.03):
        super().__init__(
            "VaR-Based Risk Management",
            f"Reduce exposure when daily VaR exceeds {abs(var_threshold)*100:.0f}%"
        )
        self.var_threshold = var_threshold

    async def generate_suggestion(
        self,
        current_portfolio: Dict[str, float],
        risk_metrics: Dict,
        market_data: Dict
    ) -> Dict:
        """Generate VaR-based hedge suggestion."""
        var_95 = risk_metrics.get("var_95", {}).get("historical", 0)

        if var_95 > self.var_threshold:
            return {
                "action": "maintain",
                "rationale": f"VaR ({abs(var_95)*100:.1f}%) is within acceptable limits",
                "suggested_allocation": current_portfolio.copy(),
                "expected_risk_reduction": 0.0
            }

        # VaR exceeds threshold - reduce risk
        excess_var = abs(var_95) - abs(self.var_threshold)
        risk_reduction_needed = excess_var / abs(var_95)

        # Scale down risky positions and increase cash/stable positions
        suggested_allocation = {}
        stable_assets = ["USDC", "USDT", "usd-coin", "tether"]

        # Calculate current stable allocation
        current_stable = sum(current_portfolio.get(asset, 0) for asset in stable_assets)
        target_stable_increase = risk_reduction_needed * 0.5

        for asset, weight in current_portfolio.items():
            if asset in stable_assets:
                # Increase stable asset weights
                if current_stable > 0:
                    proportion = weight / current_stable
                    suggested_allocation[asset] = weight + target_stable_increase * proportion
                else:
                    suggested_allocation[asset] = weight + target_stable_increase / len(stable_assets)
            else:
                # Reduce risky asset weights
                suggested_allocation[asset] = weight * (1 - target_stable_increase)

        # Normalize weights
        total_weight = sum(suggested_allocation.values())
        if total_weight > 0:
            suggested_allocation = {k: v/total_weight for k, v in suggested_allocation.items()}

        return {
            "action": "risk_reduction",
            "rationale": f"Reduce VaR from {abs(var_95)*100:.1f}% to target {abs(self.var_threshold)*100:.1f}%",
            "suggested_allocation": suggested_allocation,
            "expected_risk_reduction": risk_reduction_needed
        }


class HedgeSuggestionEngine:
    """Main engine for generating hedge suggestions."""

    def __init__(self):
        self.strategies = [
            VolatilityTargetingStrategy(target_volatility=0.15),
            ConcentrationRiskStrategy(max_single_weight=0.35),
            CorrelationBasedStrategy(correlation_threshold=0.7),
            VaRBasedStrategy(var_threshold=-0.03)
        ]

    async def generate_hedge_suggestions(
        self,
        portfolio_weights: Dict[str, float],
        risk_tolerance: str = "medium",
        market_context: Dict = None
    ) -> Dict:
        """Generate comprehensive hedge suggestions based on current portfolio."""
        try:
            # Calculate current risk metrics
            risk_metrics = await risk_calculator.calculate_portfolio_metrics(portfolio_weights)

            # Get market data context
            if market_context is None:
                market_context = await self._get_market_context(list(portfolio_weights.keys()))

            # Evaluate each strategy
            strategy_suggestions = []
            for strategy in self.strategies:
                try:
                    suggestion = await strategy.generate_suggestion(
                        portfolio_weights, risk_metrics["metrics"], market_context
                    )
                    suggestion["strategy_name"] = strategy.name
                    suggestion["strategy_description"] = strategy.description
                    strategy_suggestions.append(suggestion)
                except Exception as e:
                    logger.warning(f"Strategy {strategy.name} failed: {e}")

            # Select best suggestion based on risk tolerance and expected impact
            best_suggestion = self._select_best_suggestion(
                strategy_suggestions, risk_tolerance, risk_metrics["metrics"]
            )

            # Calculate implementation details
            implementation_cost = self._calculate_implementation_cost(
                portfolio_weights, best_suggestion["suggested_allocation"]
            )

            # Generate confidence score
            confidence_score = self._calculate_confidence_score(
                best_suggestion, risk_metrics["data_quality"]
            )

            # Store suggestion in database
            await self._store_hedge_suggestion(
                portfolio_weights, best_suggestion, implementation_cost, confidence_score
            )

            return {
                "portfolio_id": "default",
                "generated_at": datetime.now().isoformat(),
                "current_allocation": portfolio_weights,
                "risk_assessment": self._assess_current_risk(risk_metrics["metrics"]),
                "recommended_action": best_suggestion,
                "alternative_strategies": [s for s in strategy_suggestions if s != best_suggestion],
                "implementation_cost": implementation_cost,
                "confidence_score": confidence_score,
                "market_context": market_context
            }

        except Exception as e:
            logger.error(f"Hedge suggestion generation failed: {e}")
            raise

    def _select_best_suggestion(
        self,
        suggestions: List[Dict],
        risk_tolerance: str,
        risk_metrics: Dict
    ) -> Dict:
        """Select the best hedge suggestion based on risk tolerance and impact."""
        if not suggestions:
            # Return default "maintain" suggestion
            return {
                "action": "maintain",
                "rationale": "No significant portfolio adjustments needed",
                "suggested_allocation": {},
                "expected_risk_reduction": 0.0,
                "strategy_name": "No Action",
                "strategy_description": "Maintain current allocation"
            }

        # Filter out "maintain" suggestions if active action is needed
        active_suggestions = [s for s in suggestions if s["action"] != "maintain"]

        # Score suggestions based on risk tolerance
        scored_suggestions = []
        for suggestion in suggestions:
            score = self._score_suggestion(suggestion, risk_tolerance, risk_metrics)
            scored_suggestions.append((score, suggestion))

        # Return highest scored suggestion
        best_score, best_suggestion = max(scored_suggestions, key=lambda x: x[0])

        return best_suggestion

    def _score_suggestion(self, suggestion: Dict, risk_tolerance: str, risk_metrics: Dict) -> float:
        """Score a hedge suggestion based on multiple factors."""
        score = 0.0

        # Risk reduction benefit
        risk_reduction = suggestion.get("expected_risk_reduction", 0)
        score += risk_reduction * 100  # Base score from risk reduction

        # Adjust based on risk tolerance
        if risk_tolerance == "conservative":
            # Prefer suggestions that reduce risk
            if suggestion["action"] in ["risk_reduction", "diversify"]:
                score += 50
        elif risk_tolerance == "aggressive":
            # Prefer maintaining higher risk if justified
            if suggestion["action"] == "maintain":
                score += 30
        else:  # medium risk tolerance
            # Balanced approach
            if suggestion["action"] in ["rebalance", "diversify"]:
                score += 25

        # Penalize if current risk is acceptable
        current_volatility = risk_metrics.get("annualized_volatility", 0)
        if current_volatility < 0.2 and suggestion["action"] != "maintain":
            score -= 20

        # Bonus for addressing specific high-risk conditions
        if risk_metrics.get("concentration_ratio", 0) > 0.5 and "concentration" in suggestion.get("strategy_name", "").lower():
            score += 30

        if risk_metrics.get("average_correlation", 0) > 0.7 and "correlation" in suggestion.get("strategy_name", "").lower():
            score += 25

        return score

    async def _get_market_context(self, symbols: List[str]) -> Dict:
        """Get current market context for the portfolio symbols."""
        try:
            context = {
                "market_volatility": "medium",
                "correlation_regime": "normal",
                "trend_direction": "neutral"
            }

            # Calculate average volatility across symbols
            volatilities = []
            for symbol in symbols[:5]:  # Limit to avoid too many API calls
                try:
                    vol = await self._calculate_symbol_volatility(symbol)
                    if vol is not None:
                        volatilities.append(vol)
                except Exception as e:
                    logger.warning(f"Failed to get volatility for {symbol}: {e}")

            if volatilities:
                avg_volatility = np.mean(volatilities)
                if avg_volatility > 0.4:
                    context["market_volatility"] = "high"
                elif avg_volatility < 0.15:
                    context["market_volatility"] = "low"

            return context

        except Exception as e:
            logger.warning(f"Failed to get market context: {e}")
            return {"market_volatility": "unknown", "correlation_regime": "unknown"}

    async def _calculate_symbol_volatility(self, symbol: str, days: int = 30) -> Optional[float]:
        """Calculate recent volatility for a symbol."""
        try:
            price_history = await db_manager.get_price_history(symbol, days)
            if len(price_history) < 10:
                return None

            df = pd.DataFrame(price_history)
            returns = df['price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            return float(volatility)

        except Exception as e:
            logger.warning(f"Volatility calculation failed for {symbol}: {e}")
            return None

    def _calculate_implementation_cost(
        self,
        current_allocation: Dict[str, float],
        suggested_allocation: Dict[str, float]
    ) -> float:
        """Calculate estimated implementation cost as percentage of portfolio."""
        total_turnover = 0.0

        all_symbols = set(current_allocation.keys()) | set(suggested_allocation.keys())

        for symbol in all_symbols:
            current_weight = current_allocation.get(symbol, 0)
            suggested_weight = suggested_allocation.get(symbol, 0)
            turnover = abs(suggested_weight - current_weight)
            total_turnover += turnover

        # Assume 0.1% transaction cost per 1% turnover
        implementation_cost = total_turnover * 0.001

        return implementation_cost

    def _calculate_confidence_score(self, suggestion: Dict, data_quality: Dict) -> float:
        """Calculate confidence score for the suggestion."""
        confidence = 0.5  # Base confidence

        # Adjust based on data quality
        total_observations = data_quality.get("total_observations", 0)
        if total_observations >= 90:
            confidence += 0.3
        elif total_observations >= 30:
            confidence += 0.2

        assets_with_data = data_quality.get("assets_with_data", 0)
        if assets_with_data >= 3:
            confidence += 0.1

        # Adjust based on expected risk reduction
        risk_reduction = suggestion.get("expected_risk_reduction", 0)
        if risk_reduction > 0.1:
            confidence += 0.1

        # Penalize if action is maintain (might indicate lack of data)
        if suggestion["action"] == "maintain":
            confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def _assess_current_risk(self, risk_metrics: Dict) -> Dict:
        """Assess current portfolio risk level."""
        assessment = {
            "overall_level": "medium",
            "specific_risks": [],
            "risk_score": 50  # 0-100 scale
        }

        risk_score = 50

        # Assess volatility
        volatility = risk_metrics.get("annualized_volatility", 0)
        if volatility > 0.3:
            assessment["specific_risks"].append("High volatility")
            risk_score += 20
        elif volatility < 0.1:
            risk_score -= 10

        # Assess concentration
        concentration = risk_metrics.get("concentration_ratio", 0)
        if concentration > 0.5:
            assessment["specific_risks"].append("High concentration")
            risk_score += 15

        # Assess VaR
        var_95 = risk_metrics.get("var_95", {}).get("historical", 0)
        if var_95 < -0.05:
            assessment["specific_risks"].append("High Value at Risk")
            risk_score += 15

        # Assess Sharpe ratio
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        if sharpe < 0.5:
            assessment["specific_risks"].append("Poor risk-adjusted returns")
            risk_score += 10

        # Determine overall level
        if risk_score >= 70:
            assessment["overall_level"] = "high"
        elif risk_score <= 40:
            assessment["overall_level"] = "low"

        assessment["risk_score"] = min(100, max(0, risk_score))

        return assessment

    async def _store_hedge_suggestion(
        self,
        current_allocation: Dict[str, float],
        suggestion: Dict,
        implementation_cost: float,
        confidence_score: float
    ):
        """Store hedge suggestion in database."""
        try:
            # Determine risk level based on suggestion
            risk_level = "medium"
            if suggestion.get("expected_risk_reduction", 0) > 0.2:
                risk_level = "high"
            elif suggestion["action"] == "maintain":
                risk_level = "low"

            # Store in database
            query = """
            INSERT INTO hedge_suggestions (portfolio_id, risk_level, suggestion_type,
                                         current_allocation, suggested_allocation, rationale,
                                         expected_risk_reduction, implementation_cost, confidence_score)
            VALUES ('default', $1, $2, $3, $4, $5, $6, $7, $8)
            """

            await db_manager.execute_raw_query(query, {
                "risk_level": risk_level,
                "suggestion_type": suggestion["action"],
                "current_allocation": current_allocation,
                "suggested_allocation": suggestion["suggested_allocation"],
                "rationale": suggestion["rationale"],
                "expected_risk_reduction": suggestion.get("expected_risk_reduction", 0),
                "implementation_cost": implementation_cost,
                "confidence_score": confidence_score
            })

        except Exception as e:
            logger.error(f"Failed to store hedge suggestion: {e}")

    async def backtest_hedge_performance(
        self,
        historical_suggestions: List[Dict],
        lookback_days: int = 90
    ) -> Dict:
        """Backtest historical hedge suggestion performance."""
        try:
            performance_results = []

            for suggestion in historical_suggestions:
                suggestion_date = suggestion.get("created_at")
                if not suggestion_date:
                    continue

                # Get actual portfolio performance after suggestion
                current_allocation = suggestion["current_allocation"]
                suggested_allocation = suggestion["suggested_allocation"]

                # Calculate hypothetical performance (simplified)
                actual_performance = await self._calculate_portfolio_performance(
                    current_allocation, suggestion_date, lookback_days
                )

                suggested_performance = await self._calculate_portfolio_performance(
                    suggested_allocation, suggestion_date, lookback_days
                )

                if actual_performance is not None and suggested_performance is not None:
                    performance_results.append({
                        "suggestion_date": suggestion_date,
                        "actual_return": actual_performance["return"],
                        "suggested_return": suggested_performance["return"],
                        "actual_volatility": actual_performance["volatility"],
                        "suggested_volatility": suggested_performance["volatility"],
                        "improvement": suggested_performance["return"] - actual_performance["return"]
                    })

            if performance_results:
                avg_improvement = np.mean([r["improvement"] for r in performance_results])
                success_rate = sum(1 for r in performance_results if r["improvement"] > 0) / len(performance_results)

                return {
                    "total_suggestions": len(performance_results),
                    "average_improvement": avg_improvement,
                    "success_rate": success_rate,
                    "detailed_results": performance_results
                }
            else:
                return {"error": "No valid backtesting data available"}

        except Exception as e:
            logger.error(f"Hedge backtesting failed: {e}")
            return {"error": str(e)}

    async def _calculate_portfolio_performance(
        self,
        allocation: Dict[str, float],
        start_date: datetime,
        days: int
    ) -> Optional[Dict]:
        """Calculate portfolio performance over a period."""
        try:
            portfolio_returns = []

            # Get price data for each asset
            for symbol, weight in allocation.items():
                if weight > 0:
                    price_history = await db_manager.get_price_history(symbol, days)
                    if price_history:
                        # Filter to start from suggestion date
                        filtered_history = [
                            p for p in price_history
                            if pd.to_datetime(p["time"]) >= start_date
                        ]

                        if len(filtered_history) > 1:
                            df = pd.DataFrame(filtered_history)
                            returns = df['price'].pct_change().dropna()
                            weighted_returns = returns * weight
                            portfolio_returns.append(weighted_returns)

            if portfolio_returns:
                # Sum weighted returns
                combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
                total_return = (1 + combined_returns).prod() - 1
                volatility = combined_returns.std() * np.sqrt(252)

                return {
                    "return": float(total_return),
                    "volatility": float(volatility)
                }

        except Exception as e:
            logger.warning(f"Performance calculation failed: {e}")

        return None


# Global hedge suggestion engine instance
hedge_engine = HedgeSuggestionEngine()