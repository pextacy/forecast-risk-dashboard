"""AI-powered explainability engine for generating financial narratives and insights."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import openai
from app.utils.config import settings

logger = logging.getLogger(__name__)


class FinancialExplainabilityEngine:
    """Generate clear, actionable explanations for financial analysis results."""

    def __init__(self):
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
        else:
            logger.warning("OpenAI API key not configured, explanations will be rule-based")

    async def explain_forecast_results(
        self,
        symbol: str,
        forecast_data: Dict,
        current_price: float,
        historical_volatility: float
    ) -> Dict:
        """Generate comprehensive explanation for forecast results."""
        try:
            # Extract key forecast metrics
            forecast_summary = self._extract_forecast_summary(forecast_data, current_price)

            if settings.openai_api_key:
                narrative = await self._generate_ai_forecast_narrative(
                    symbol, forecast_summary, historical_volatility
                )
            else:
                narrative = self._generate_rule_based_forecast_narrative(
                    symbol, forecast_summary, historical_volatility
                )

            return {
                "symbol": symbol,
                "explanation_type": "forecast",
                "generated_at": datetime.now().isoformat(),
                "narrative": narrative,
                "key_insights": self._extract_forecast_insights(forecast_summary),
                "risk_level": self._assess_forecast_risk_level(forecast_summary),
                "confidence_assessment": self._assess_forecast_confidence(forecast_data)
            }

        except Exception as e:
            logger.error(f"Forecast explanation failed for {symbol}: {e}")
            return self._generate_fallback_explanation("forecast", symbol, str(e))

    async def explain_risk_metrics(
        self,
        portfolio_weights: Dict[str, float],
        risk_metrics: Dict,
        market_context: Dict = None
    ) -> Dict:
        """Generate comprehensive explanation for portfolio risk metrics."""
        try:
            # Analyze risk levels and key concerns
            risk_analysis = self._analyze_risk_levels(risk_metrics)

            if settings.openai_api_key:
                narrative = await self._generate_ai_risk_narrative(
                    portfolio_weights, risk_analysis, market_context
                )
            else:
                narrative = self._generate_rule_based_risk_narrative(
                    portfolio_weights, risk_analysis
                )

            return {
                "portfolio_weights": portfolio_weights,
                "explanation_type": "risk_analysis",
                "generated_at": datetime.now().isoformat(),
                "narrative": narrative,
                "key_concerns": risk_analysis["key_concerns"],
                "risk_level": risk_analysis["overall_risk_level"],
                "recommendations": self._generate_risk_recommendations(risk_analysis)
            }

        except Exception as e:
            logger.error(f"Risk explanation failed: {e}")
            return self._generate_fallback_explanation("risk", "portfolio", str(e))

    async def explain_hedge_suggestion(
        self,
        current_allocation: Dict[str, float],
        suggested_allocation: Dict[str, float],
        rationale_data: Dict
    ) -> Dict:
        """Generate explanation for hedge/rebalancing suggestions."""
        try:
            # Calculate allocation changes
            allocation_changes = self._calculate_allocation_changes(
                current_allocation, suggested_allocation
            )

            if settings.openai_api_key:
                narrative = await self._generate_ai_hedge_narrative(
                    allocation_changes, rationale_data
                )
            else:
                narrative = self._generate_rule_based_hedge_narrative(
                    allocation_changes, rationale_data
                )

            return {
                "explanation_type": "hedge_suggestion",
                "generated_at": datetime.now().isoformat(),
                "narrative": narrative,
                "allocation_changes": allocation_changes,
                "expected_impact": self._calculate_expected_impact(rationale_data),
                "implementation_steps": self._generate_implementation_steps(allocation_changes)
            }

        except Exception as e:
            logger.error(f"Hedge explanation failed: {e}")
            return self._generate_fallback_explanation("hedge", "rebalancing", str(e))

    def _extract_forecast_summary(self, forecast_data: Dict, current_price: float) -> Dict:
        """Extract key metrics from forecast results."""
        summary = {
            "current_price": current_price,
            "models_used": list(forecast_data.get("forecasts", {}).keys()),
            "forecast_horizon": forecast_data.get("forecast_horizon", 30)
        }

        # Get ensemble or best forecast
        best_forecast = None
        if "ensemble" in forecast_data.get("forecasts", {}):
            best_forecast = forecast_data["forecasts"]["ensemble"]
        elif "arima" in forecast_data.get("forecasts", {}):
            best_forecast = forecast_data["forecasts"]["arima"]
        elif "prophet" in forecast_data.get("forecasts", {}):
            best_forecast = forecast_data["forecasts"]["prophet"]

        if best_forecast and "forecast" in best_forecast:
            forecast_values = best_forecast["forecast"]
            final_price = forecast_values[-1]
            max_price = max(forecast_values)
            min_price = min(forecast_values)

            summary.update({
                "predicted_final_price": final_price,
                "predicted_max_price": max_price,
                "predicted_min_price": min_price,
                "expected_return": (final_price - current_price) / current_price,
                "price_range": (max_price - min_price) / current_price,
                "trend_direction": "bullish" if final_price > current_price else "bearish"
            })

            # Extract confidence intervals if available
            conf_intervals = best_forecast.get("confidence_intervals", {})
            if "95%" in conf_intervals:
                summary["confidence_95"] = {
                    "lower": conf_intervals["95%"]["lower"][-1],
                    "upper": conf_intervals["95%"]["upper"][-1]
                }

        return summary

    def _analyze_risk_levels(self, risk_metrics: Dict) -> Dict:
        """Analyze risk metrics and categorize risk levels."""
        analysis = {
            "key_concerns": [],
            "risk_indicators": {},
            "overall_risk_level": "medium"
        }

        # Analyze volatility
        volatility = risk_metrics.get("annualized_volatility", 0)
        if volatility > 0.4:  # 40% annual volatility
            analysis["key_concerns"].append("High volatility detected")
            analysis["risk_indicators"]["volatility"] = "high"
        elif volatility > 0.2:
            analysis["risk_indicators"]["volatility"] = "medium"
        else:
            analysis["risk_indicators"]["volatility"] = "low"

        # Analyze VaR
        var_95 = risk_metrics.get("var_95", {}).get("historical", 0)
        if var_95 < -0.05:  # 5% daily VaR
            analysis["key_concerns"].append("High Value at Risk (95% confidence)")
            analysis["risk_indicators"]["var"] = "high"
        elif var_95 < -0.02:
            analysis["risk_indicators"]["var"] = "medium"
        else:
            analysis["risk_indicators"]["var"] = "low"

        # Analyze Sharpe ratio
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        if sharpe < 0.5:
            analysis["key_concerns"].append("Poor risk-adjusted returns")
            analysis["risk_indicators"]["sharpe"] = "low"
        elif sharpe > 1.0:
            analysis["risk_indicators"]["sharpe"] = "good"
        else:
            analysis["risk_indicators"]["sharpe"] = "medium"

        # Analyze max drawdown
        max_drawdown = risk_metrics.get("max_drawdown", 0)
        if max_drawdown < -0.2:  # 20% drawdown
            analysis["key_concerns"].append("Significant historical drawdowns")
            analysis["risk_indicators"]["drawdown"] = "high"
        elif max_drawdown < -0.1:
            analysis["risk_indicators"]["drawdown"] = "medium"
        else:
            analysis["risk_indicators"]["drawdown"] = "low"

        # Analyze concentration
        concentration = risk_metrics.get("concentration_ratio", 0)
        if concentration > 0.6:  # 60% in single asset
            analysis["key_concerns"].append("High portfolio concentration")
            analysis["risk_indicators"]["concentration"] = "high"

        # Determine overall risk level
        high_risk_count = sum(1 for indicator in analysis["risk_indicators"].values() if indicator == "high")
        if high_risk_count >= 2 or len(analysis["key_concerns"]) >= 3:
            analysis["overall_risk_level"] = "high"
        elif high_risk_count == 0 and len(analysis["key_concerns"]) == 0:
            analysis["overall_risk_level"] = "low"

        return analysis

    async def _generate_ai_forecast_narrative(
        self,
        symbol: str,
        forecast_summary: Dict,
        historical_volatility: float
    ) -> str:
        """Generate AI-powered forecast narrative."""
        try:
            prompt = f"""
            As a senior financial analyst, provide a clear, actionable explanation of the following forecast for {symbol}:

            Current Price: ${forecast_summary.get('current_price', 0):.2f}
            Predicted Price (30 days): ${forecast_summary.get('predicted_final_price', 0):.2f}
            Expected Return: {forecast_summary.get('expected_return', 0)*100:.1f}%
            Trend Direction: {forecast_summary.get('trend_direction', 'neutral')}
            Historical Volatility: {historical_volatility*100:.1f}%
            Models Used: {', '.join(forecast_summary.get('models_used', []))}

            Provide a 2-3 paragraph explanation that:
            1. Summarizes what the forecast indicates in plain language
            2. Explains the key factors driving the prediction
            3. Discusses the confidence level and uncertainty
            4. Offers practical implications for investors

            Use clear, professional language suitable for treasury teams and fund managers.
            Avoid jargon and focus on actionable insights.
            """

            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"AI narrative generation failed: {e}")
            return self._generate_rule_based_forecast_narrative(symbol, forecast_summary, historical_volatility)

    def _generate_rule_based_forecast_narrative(
        self,
        symbol: str,
        forecast_summary: Dict,
        historical_volatility: float
    ) -> str:
        """Generate rule-based forecast narrative when AI is unavailable."""
        current_price = forecast_summary.get("current_price", 0)
        predicted_price = forecast_summary.get("predicted_final_price", 0)
        expected_return = forecast_summary.get("expected_return", 0)
        trend = forecast_summary.get("trend_direction", "neutral")

        narrative = f"Based on our analysis of {symbol}, "

        if trend == "bullish":
            narrative += f"the forecast indicates positive momentum with an expected price increase to ${predicted_price:.2f} "
            narrative += f"over the next 30 days, representing a potential {expected_return*100:.1f}% gain. "
        elif trend == "bearish":
            narrative += f"the forecast suggests downward pressure with an expected decline to ${predicted_price:.2f} "
            narrative += f"over the next 30 days, representing a potential {abs(expected_return)*100:.1f}% loss. "
        else:
            narrative += f"the forecast indicates relatively stable price action around ${predicted_price:.2f}. "

        # Add volatility context
        if historical_volatility > 0.3:
            narrative += f"Given the high historical volatility of {historical_volatility*100:.1f}%, "
            narrative += "there is significant uncertainty around this forecast, and actual prices may deviate substantially. "
        elif historical_volatility < 0.1:
            narrative += f"The relatively low historical volatility of {historical_volatility*100:.1f}% "
            narrative += "suggests this forecast has higher reliability, though market conditions can change rapidly. "

        # Add confidence assessment
        models_count = len(forecast_summary.get("models_used", []))
        if models_count > 1:
            narrative += f"This forecast combines insights from {models_count} different models, "
            narrative += "providing a more robust prediction than single-model approaches."
        else:
            narrative += "This forecast is based on a single model and should be considered alongside other market indicators."

        return narrative

    async def _generate_ai_risk_narrative(
        self,
        portfolio_weights: Dict[str, float],
        risk_analysis: Dict,
        market_context: Dict = None
    ) -> str:
        """Generate AI-powered risk analysis narrative."""
        try:
            portfolio_desc = ", ".join([f"{symbol}: {weight*100:.1f}%" for symbol, weight in portfolio_weights.items()])

            prompt = f"""
            As a risk management expert, provide a clear assessment of this portfolio's risk profile:

            Portfolio Allocation: {portfolio_desc}
            Overall Risk Level: {risk_analysis.get('overall_risk_level', 'medium')}
            Key Concerns: {', '.join(risk_analysis.get('key_concerns', []))}
            Risk Indicators: {json.dumps(risk_analysis.get('risk_indicators', {}), indent=2)}

            Provide a 2-3 paragraph risk assessment that:
            1. Summarizes the portfolio's current risk profile in plain language
            2. Explains the most significant risk factors and their implications
            3. Offers specific, actionable risk management recommendations
            4. Discusses appropriate risk tolerance levels for this portfolio

            Focus on practical treasury management insights.
            Use clear language suitable for finance teams making allocation decisions.
            """

            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"AI risk narrative generation failed: {e}")
            return self._generate_rule_based_risk_narrative(portfolio_weights, risk_analysis)

    def _generate_rule_based_risk_narrative(
        self,
        portfolio_weights: Dict[str, float],
        risk_analysis: Dict
    ) -> str:
        """Generate rule-based risk narrative when AI is unavailable."""
        risk_level = risk_analysis.get("overall_risk_level", "medium")
        key_concerns = risk_analysis.get("key_concerns", [])

        narrative = f"Your portfolio currently exhibits {risk_level} risk characteristics. "

        # Address key concerns
        if key_concerns:
            narrative += f"Primary concerns include: {', '.join(key_concerns[:3])}. "

            if "High volatility detected" in key_concerns:
                narrative += "The elevated volatility suggests significant price swings are likely, "
                narrative += "which could result in substantial gains or losses over short periods. "

            if "High portfolio concentration" in key_concerns:
                max_weight = max(portfolio_weights.values())
                narrative += f"With {max_weight*100:.1f}% allocated to a single asset, "
                narrative += "the portfolio lacks diversification and is vulnerable to asset-specific risks. "

            if "Poor risk-adjusted returns" in key_concerns:
                narrative += "The current risk-return profile suggests you may be taking on more risk "
                narrative += "than necessary for the expected returns. "

        else:
            narrative += "The portfolio shows balanced risk characteristics with no major red flags identified. "

        # Add general risk management advice
        if risk_level == "high":
            narrative += "Consider reducing position sizes, increasing diversification, or adding hedge positions "
            narrative += "to bring risk levels within acceptable parameters."
        elif risk_level == "low":
            narrative += "The conservative risk profile may be appropriate for capital preservation strategies, "
            narrative += "though it may limit potential returns in favorable market conditions."

        return narrative

    async def _generate_ai_hedge_narrative(
        self,
        allocation_changes: Dict,
        rationale_data: Dict
    ) -> str:
        """Generate AI-powered hedge suggestion narrative."""
        try:
            changes_desc = self._format_allocation_changes(allocation_changes)

            prompt = f"""
            As a portfolio manager, explain this rebalancing recommendation in clear terms:

            Suggested Changes: {changes_desc}
            Expected Risk Reduction: {rationale_data.get('expected_risk_reduction', 0)*100:.1f}%
            Implementation Cost: {rationale_data.get('implementation_cost', 0)*100:.2f}%
            Confidence Score: {rationale_data.get('confidence_score', 0)*100:.1f}%

            Provide a clear 2-3 paragraph explanation that:
            1. Explains why these changes are recommended
            2. Describes the expected benefits and trade-offs
            3. Provides guidance on implementation timing and approach
            4. Addresses potential risks of making these changes

            Use language appropriate for treasury teams making allocation decisions.
            Focus on practical implementation considerations.
            """

            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"AI hedge narrative generation failed: {e}")
            return self._generate_rule_based_hedge_narrative(allocation_changes, rationale_data)

    def _generate_rule_based_hedge_narrative(
        self,
        allocation_changes: Dict,
        rationale_data: Dict
    ) -> str:
        """Generate rule-based hedge narrative when AI is unavailable."""
        expected_reduction = rationale_data.get("expected_risk_reduction", 0)
        implementation_cost = rationale_data.get("implementation_cost", 0)

        narrative = "Based on current risk analysis, we recommend the following portfolio adjustments: "

        # Describe major changes
        major_changes = [(asset, change) for asset, change in allocation_changes.items()
                        if abs(change) > 0.05]  # 5% threshold

        if major_changes:
            for asset, change in major_changes[:3]:  # Limit to top 3 changes
                if change > 0:
                    narrative += f"Increase {asset} allocation by {change*100:.1f}%. "
                else:
                    narrative += f"Reduce {asset} allocation by {abs(change)*100:.1f}%. "

        # Explain expected benefits
        if expected_reduction > 0.1:
            narrative += f"These changes are expected to reduce portfolio risk by {expected_reduction*100:.1f}%, "
            narrative += "improving the risk-return profile while maintaining growth potential. "
        elif expected_reduction > 0:
            narrative += f"A modest risk reduction of {expected_reduction*100:.1f}% is expected, "
            narrative += "with minimal impact on return potential. "

        # Address implementation costs
        if implementation_cost > 0.005:  # 0.5%
            narrative += f"Implementation costs are estimated at {implementation_cost*100:.2f}% of portfolio value. "
            narrative += "Consider executing these changes gradually to minimize market impact."
        else:
            narrative += "Implementation costs are minimal, allowing for immediate execution if desired."

        return narrative

    def _extract_forecast_insights(self, forecast_summary: Dict) -> List[str]:
        """Extract key insights from forecast summary."""
        insights = []

        expected_return = forecast_summary.get("expected_return", 0)
        if abs(expected_return) > 0.1:  # 10% move
            direction = "upward" if expected_return > 0 else "downward"
            insights.append(f"Significant {direction} price movement expected ({expected_return*100:.1f}%)")

        price_range = forecast_summary.get("price_range", 0)
        if price_range > 0.2:  # 20% range
            insights.append(f"High price volatility anticipated (±{price_range*50:.1f}%)")

        models_used = forecast_summary.get("models_used", [])
        if len(models_used) > 1:
            insights.append(f"Consensus across {len(models_used)} forecasting models")

        return insights

    def _assess_forecast_risk_level(self, forecast_summary: Dict) -> str:
        """Assess risk level of forecast."""
        price_range = forecast_summary.get("price_range", 0)
        expected_return = abs(forecast_summary.get("expected_return", 0))

        if price_range > 0.3 or expected_return > 0.2:
            return "high"
        elif price_range > 0.1 or expected_return > 0.05:
            return "medium"
        else:
            return "low"

    def _assess_forecast_confidence(self, forecast_data: Dict) -> str:
        """Assess confidence level of forecast."""
        models_count = len(forecast_data.get("forecasts", {}))
        data_points = forecast_data.get("data_points_used", 0)

        if models_count >= 2 and data_points >= 90:
            return "high"
        elif models_count >= 1 and data_points >= 30:
            return "medium"
        else:
            return "low"

    def _calculate_allocation_changes(
        self,
        current: Dict[str, float],
        suggested: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate allocation changes between current and suggested portfolios."""
        all_symbols = set(current.keys()) | set(suggested.keys())
        changes = {}

        for symbol in all_symbols:
            current_weight = current.get(symbol, 0)
            suggested_weight = suggested.get(symbol, 0)
            changes[symbol] = suggested_weight - current_weight

        return changes

    def _generate_risk_recommendations(self, risk_analysis: Dict) -> List[str]:
        """Generate specific risk management recommendations."""
        recommendations = []
        risk_level = risk_analysis.get("overall_risk_level", "medium")
        key_concerns = risk_analysis.get("key_concerns", [])

        if "High volatility detected" in key_concerns:
            recommendations.append("Consider adding stable assets (bonds, stablecoins) to reduce volatility")

        if "High portfolio concentration" in key_concerns:
            recommendations.append("Diversify holdings across more assets and asset classes")

        if "Poor risk-adjusted returns" in key_concerns:
            recommendations.append("Review asset selection and consider higher-quality investments")

        if "High Value at Risk" in key_concerns:
            recommendations.append("Implement position sizing limits and stop-loss strategies")

        if risk_level == "high" and not recommendations:
            recommendations.append("Reduce overall position sizes and increase cash reserves")

        return recommendations

    def _calculate_expected_impact(self, rationale_data: Dict) -> Dict:
        """Calculate expected impact of hedge suggestions."""
        return {
            "risk_reduction": rationale_data.get("expected_risk_reduction", 0),
            "return_impact": rationale_data.get("expected_return_impact", 0),
            "implementation_cost": rationale_data.get("implementation_cost", 0),
            "timeframe": "immediate"
        }

    def _generate_implementation_steps(self, allocation_changes: Dict) -> List[str]:
        """Generate step-by-step implementation guidance."""
        steps = []

        # Sort changes by magnitude
        sorted_changes = sorted(
            allocation_changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for i, (asset, change) in enumerate(sorted_changes[:5]):  # Top 5 changes
            if abs(change) > 0.01:  # 1% threshold
                action = "Increase" if change > 0 else "Reduce"
                steps.append(f"{i+1}. {action} {asset} allocation by {abs(change)*100:.1f}%")

        if not steps:
            steps.append("1. No significant allocation changes required")

        steps.append(f"{len(steps)+1}. Monitor portfolio performance and rebalance as needed")

        return steps

    def _format_allocation_changes(self, allocation_changes: Dict) -> str:
        """Format allocation changes for display."""
        changes = []
        for asset, change in allocation_changes.items():
            if abs(change) > 0.01:  # 1% threshold
                direction = "+" if change > 0 else ""
                changes.append(f"{asset}: {direction}{change*100:.1f}%")

        return ", ".join(changes[:5])  # Limit to top 5

    def _generate_fallback_explanation(self, explanation_type: str, subject: str, error: str) -> Dict:
        """Generate fallback explanation when other methods fail."""
        return {
            "explanation_type": explanation_type,
            "subject": subject,
            "generated_at": datetime.now().isoformat(),
            "narrative": f"Analysis for {subject} completed with limited explanatory capability. "
                        f"Technical details are available in the raw results. "
                        f"Please review the quantitative metrics for detailed insights.",
            "error": error,
            "status": "fallback_mode"
        }


# Global explainability engine instance
explainability_engine = FinancialExplainabilityEngine()