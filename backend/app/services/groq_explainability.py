"""
Groq-powered AI explanations for Treasury Risk Dashboard.
Replaces OpenAI with Groq for faster, more cost-effective AI responses.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from groq import Groq
from app.utils.config import settings

logger = logging.getLogger(__name__)


class GroqExplainabilityEngine:
    """
    Enhanced explainability engine using Groq for treasury analytics.
    Provides fast, contextual explanations for forecasts, risk metrics, and hedge suggestions.
    """

    def __init__(self):
        if not settings.groq_api_key:
            logger.warning("Groq API key not found. AI explanations will use fallback responses.")
            self.client = None
        else:
            self.client = Groq(api_key=settings.groq_api_key)

        # Treasury-focused system prompts
        self.system_prompts = {
            "forecast": """You are a treasury management AI advisor specializing in cryptocurrency and traditional asset forecasting.
            Provide clear, actionable explanations for financial forecasts that DAO treasury managers can understand.
            Focus on risk implications, market context, and practical decision-making insights.
            Always include confidence levels and key risks.""",

            "risk": """You are a risk management specialist for DAO treasuries.
            Explain complex risk metrics (VaR, volatility, Sharpe ratios) in plain language.
            Connect quantitative measures to real-world treasury decisions and portfolio implications.
            Highlight both opportunities and threats.""",

            "hedge": """You are a portfolio optimization expert for DAOs and institutional treasuries.
            Explain hedging strategies and rebalancing recommendations with clear rationale.
            Focus on why changes are suggested, expected outcomes, and implementation considerations.
            Always discuss trade-offs between risk reduction and potential returns."""
        }

    async def explain_forecast_results(
        self,
        symbol: str,
        forecast_data: Dict,
        current_price: float,
        historical_volatility: float,
        context: str = "forecast"
    ) -> Dict[str, Any]:
        """Generate AI explanation for forecast results."""

        if not self.client:
            return self._get_fallback_forecast_explanation(symbol, forecast_data, current_price)

        try:
            # Extract key forecast metrics
            forecasts = forecast_data.get("forecasts", {})
            horizon = forecast_data.get("forecast_horizon", 30)
            data_points = forecast_data.get("data_points_used", 0)

            # Get best performing model
            best_model = self._get_best_model(forecasts)
            forecast_values = forecasts.get(best_model, {}).get("forecast", [])

            if not forecast_values:
                return {"error": "No forecast data available for explanation"}

            final_price = forecast_values[-1]
            price_change = ((final_price - current_price) / current_price) * 100

            # Build context-rich prompt
            prompt = f"""
            Analyze this {horizon}-day forecast for {symbol.upper()}:

            Current Price: ${current_price:,.2f}
            Predicted Final Price: ${final_price:,.2f}
            Expected Change: {price_change:+.1f}%
            Historical Volatility: {historical_volatility*100:.1f}%
            Model Used: {best_model.upper()}
            Data Points: {data_points} days

            Key forecast metrics:
            {self._format_forecast_metrics(forecasts[best_model])}

            Provide a treasury-focused analysis including:
            1. Market outlook interpretation
            2. Risk assessment for treasury holdings
            3. Recommended actions for DAO treasury managers
            4. Key risks and confidence level

            Keep it concise but actionable (max 200 words).
            """

            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",  # Groq's fastest large model
                messages=[
                    {"role": "system", "content": self.system_prompts[context]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            narrative = response.choices[0].message.content

            # Extract insights and structure response
            insights = self._extract_key_insights(narrative)

            return {
                "narrative": narrative,
                "key_insights": insights,
                "confidence_score": self._calculate_confidence_score(forecasts[best_model]),
                "risk_level": self._assess_risk_level(historical_volatility, abs(price_change)),
                "model_used": best_model,
                "recommendation": self._generate_recommendation(price_change, historical_volatility)
            }

        except Exception as e:
            logger.error(f"Groq explanation failed: {e}")
            return self._get_fallback_forecast_explanation(symbol, forecast_data, current_price)

    async def explain_risk_metrics(
        self,
        portfolio_weights: Dict[str, float],
        risk_metrics: Dict,
        context: str = "risk"
    ) -> Dict[str, Any]:
        """Generate AI explanation for portfolio risk analysis."""

        if not self.client:
            return self._get_fallback_risk_explanation(risk_metrics)

        try:
            var_95 = risk_metrics.get("var_95", {}).get("historical", 0)
            sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
            volatility = risk_metrics.get("volatility_30d", 0)
            max_drawdown = risk_metrics.get("max_drawdown", 0)
            concentration = risk_metrics.get("concentration_ratio", 0)

            # Build portfolio context
            top_holdings = sorted(portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:3]

            prompt = f"""
            Analyze this DAO treasury portfolio risk profile:

            Portfolio Composition:
            {self._format_portfolio_holdings(portfolio_weights)}

            Risk Metrics:
            - VaR (95%): {abs(var_95)*100:.1f}% (worst 1-day loss)
            - Sharpe Ratio: {sharpe_ratio:.2f}
            - 30-Day Volatility: {volatility*100:.1f}%
            - Max Drawdown: {abs(max_drawdown)*100:.1f}%
            - Concentration Risk: {concentration*100:.1f}% (largest position)

            Provide treasury management insights including:
            1. Overall risk assessment and portfolio health
            2. Specific risks to highlight for DAO governance
            3. Diversification recommendations
            4. Risk-adjusted performance evaluation

            Focus on actionable insights for treasury decisions (max 200 words).
            """

            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompts[context]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            narrative = response.choices[0].message.content
            insights = self._extract_key_insights(narrative)

            return {
                "narrative": narrative,
                "key_insights": insights,
                "risk_rating": self._get_risk_rating(var_95, volatility, concentration),
                "primary_concerns": self._identify_primary_risks(risk_metrics),
                "recommendations": self._generate_risk_recommendations(risk_metrics, portfolio_weights)
            }

        except Exception as e:
            logger.error(f"Risk explanation failed: {e}")
            return self._get_fallback_risk_explanation(risk_metrics)

    async def explain_hedge_suggestion(
        self,
        current_allocation: Dict[str, float],
        suggested_allocation: Dict[str, float],
        expected_risk_reduction: float,
        rationale: str,
        context: str = "hedge"
    ) -> Dict[str, Any]:
        """Generate AI explanation for hedge suggestions."""

        if not self.client:
            return self._get_fallback_hedge_explanation(current_allocation, suggested_allocation)

        try:
            # Calculate rebalancing changes
            changes = {}
            for asset in set(list(current_allocation.keys()) + list(suggested_allocation.keys())):
                current = current_allocation.get(asset, 0)
                suggested = suggested_allocation.get(asset, 0)
                changes[asset] = suggested - current

            significant_changes = {k: v for k, v in changes.items() if abs(v) > 0.01}  # >1% change

            prompt = f"""
            Analyze this DAO treasury rebalancing recommendation:

            Current Portfolio:
            {self._format_allocation_changes(current_allocation, suggested_allocation)}

            Expected Risk Reduction: {expected_risk_reduction*100:.1f}%

            Rationale: {rationale}

            Significant Changes:
            {self._format_significant_changes(significant_changes)}

            Provide strategic guidance including:
            1. Why this rebalancing makes sense for DAOs
            2. Expected impact on portfolio stability and returns
            3. Implementation timing and considerations
            4. Potential risks of making these changes

            Focus on practical execution advice (max 200 words).
            """

            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompts[context]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            narrative = response.choices[0].message.content
            insights = self._extract_key_insights(narrative)

            return {
                "narrative": narrative,
                "key_insights": insights,
                "implementation_priority": self._assess_implementation_priority(expected_risk_reduction),
                "execution_steps": self._generate_execution_steps(significant_changes),
                "risk_considerations": self._identify_rebalancing_risks(changes)
            }

        except Exception as e:
            logger.error(f"Hedge explanation failed: {e}")
            return self._get_fallback_hedge_explanation(current_allocation, suggested_allocation)

    # Helper methods
    def _get_best_model(self, forecasts: Dict) -> str:
        """Select the best performing model based on available metrics."""
        if "ensemble" in forecasts:
            return "ensemble"
        elif "prophet" in forecasts:
            return "prophet"
        elif "arima" in forecasts:
            return "arima"
        else:
            return list(forecasts.keys())[0] if forecasts else "unknown"

    def _format_forecast_metrics(self, forecast_data: Dict) -> str:
        """Format forecast metrics for prompt."""
        metrics = forecast_data.get("metrics", {})
        return f"MAE: {metrics.get('mae', 0):.2f}, RMSE: {metrics.get('rmse', 0):.2f}"

    def _extract_key_insights(self, narrative: str) -> List[str]:
        """Extract bullet points from narrative."""
        lines = narrative.split('\n')
        insights = []
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                insights.append(line.lstrip('•-*1234. '))
        return insights[:5]  # Max 5 insights

    def _calculate_confidence_score(self, forecast_data: Dict) -> float:
        """Calculate confidence score based on model metrics."""
        metrics = forecast_data.get("metrics", {})
        mae = metrics.get("mae", 100)
        # Lower MAE = higher confidence (simplified)
        return max(0.1, min(0.95, 1.0 - (mae / 1000)))

    def _assess_risk_level(self, volatility: float, price_change_abs: float) -> str:
        """Assess risk level based on volatility and expected change."""
        if volatility > 0.5 or price_change_abs > 20:
            return "high"
        elif volatility > 0.3 or price_change_abs > 10:
            return "medium"
        else:
            return "low"

    def _generate_recommendation(self, price_change: float, volatility: float) -> str:
        """Generate simple recommendation based on forecast."""
        if price_change > 10 and volatility < 0.3:
            return "Consider maintaining or increasing position"
        elif price_change < -10 or volatility > 0.5:
            return "Consider reducing exposure or hedging"
        else:
            return "Monitor closely and maintain current allocation"

    def _format_portfolio_holdings(self, weights: Dict[str, float]) -> str:
        """Format portfolio holdings for prompt."""
        sorted_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"  {asset}: {weight*100:.1f}%" for asset, weight in sorted_holdings[:5]])

    def _get_risk_rating(self, var_95: float, volatility: float, concentration: float) -> str:
        """Generate overall risk rating."""
        risk_score = abs(var_95) * 2 + volatility + concentration
        if risk_score > 0.6:
            return "High Risk"
        elif risk_score > 0.3:
            return "Medium Risk"
        else:
            return "Low Risk"

    def _identify_primary_risks(self, risk_metrics: Dict) -> List[str]:
        """Identify the primary risk factors."""
        risks = []
        concentration = risk_metrics.get("concentration_ratio", 0)
        volatility = risk_metrics.get("volatility_30d", 0)
        max_drawdown = abs(risk_metrics.get("max_drawdown", 0))

        if concentration > 0.4:
            risks.append("High concentration risk")
        if volatility > 0.4:
            risks.append("High volatility exposure")
        if max_drawdown > 0.2:
            risks.append("Significant drawdown potential")

        return risks[:3]

    def _generate_risk_recommendations(self, risk_metrics: Dict, portfolio_weights: Dict) -> List[str]:
        """Generate specific risk management recommendations."""
        recommendations = []
        concentration = risk_metrics.get("concentration_ratio", 0)

        if concentration > 0.4:
            recommendations.append("Diversify by reducing largest position")
        if risk_metrics.get("volatility_30d", 0) > 0.4:
            recommendations.append("Add stable assets to reduce volatility")
        if risk_metrics.get("sharpe_ratio", 0) < 0.5:
            recommendations.append("Optimize for better risk-adjusted returns")

        return recommendations

    def _format_allocation_changes(self, current: Dict, suggested: Dict) -> str:
        """Format allocation changes for prompt."""
        all_assets = set(list(current.keys()) + list(suggested.keys()))
        changes = []
        for asset in sorted(all_assets):
            curr = current.get(asset, 0) * 100
            sugg = suggested.get(asset, 0) * 100
            changes.append(f"  {asset}: {curr:.1f}% → {sugg:.1f}%")
        return "\n".join(changes)

    def _format_significant_changes(self, changes: Dict) -> str:
        """Format significant changes for prompt."""
        if not changes:
            return "  No significant changes"

        formatted = []
        for asset, change in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "increase" if change > 0 else "decrease"
            formatted.append(f"  {asset}: {direction} by {abs(change)*100:.1f}%")
        return "\n".join(formatted[:3])

    def _assess_implementation_priority(self, risk_reduction: float) -> str:
        """Assess how urgent the implementation is."""
        if risk_reduction > 0.2:
            return "High Priority"
        elif risk_reduction > 0.1:
            return "Medium Priority"
        else:
            return "Low Priority"

    def _generate_execution_steps(self, changes: Dict) -> List[str]:
        """Generate practical execution steps."""
        steps = ["Review current market conditions"]

        # Sort by magnitude of change
        sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)

        for asset, change in sorted_changes[:2]:
            if change > 0:
                steps.append(f"Gradually accumulate {asset}")
            else:
                steps.append(f"Reduce {asset} position over time")

        steps.append("Monitor portfolio impact after changes")
        return steps

    def _identify_rebalancing_risks(self, changes: Dict) -> List[str]:
        """Identify risks in the rebalancing process."""
        risks = []

        large_changes = [asset for asset, change in changes.items() if abs(change) > 0.15]
        if large_changes:
            risks.append("Market impact from large position changes")

        if len([c for c in changes.values() if c != 0]) > 5:
            risks.append("Execution complexity with multiple assets")

        risks.append("Timing risk in volatile markets")
        return risks[:3]

    # Fallback methods when Groq API is unavailable
    def _get_fallback_forecast_explanation(self, symbol: str, forecast_data: Dict, current_price: float) -> Dict:
        """Fallback explanation when Groq API is unavailable."""
        return {
            "narrative": f"30-day forecast generated for {symbol.upper()} using advanced ML models. "
                        f"Current price: ${current_price:,.2f}. Review forecast chart for detailed predictions.",
            "key_insights": [
                "Forecast generated using ensemble of statistical models",
                "Consider market volatility in decision making",
                "Monitor forecast accuracy over time"
            ],
            "confidence_score": 0.7,
            "risk_level": "medium",
            "recommendation": "Review forecast data carefully before making treasury decisions"
        }

    def _get_fallback_risk_explanation(self, risk_metrics: Dict) -> Dict:
        """Fallback risk explanation."""
        var_95 = abs(risk_metrics.get("var_95", {}).get("historical", 0))
        return {
            "narrative": f"Portfolio risk analysis complete. Value at Risk (95%): {var_95*100:.1f}%. "
                        "Review detailed metrics for comprehensive risk assessment.",
            "key_insights": [
                "VaR represents worst-case daily loss with 95% confidence",
                "Monitor concentration risk across assets",
                "Regular rebalancing may improve risk-adjusted returns"
            ],
            "risk_rating": "Medium Risk",
            "primary_concerns": ["Market volatility", "Concentration risk"],
            "recommendations": ["Diversify holdings", "Monitor correlations"]
        }

    def _get_fallback_hedge_explanation(self, current: Dict, suggested: Dict) -> Dict:
        """Fallback hedge explanation."""
        return {
            "narrative": "Portfolio rebalancing recommendation generated based on risk optimization analysis. "
                        "Review suggested allocation changes to reduce portfolio risk.",
            "key_insights": [
                "Rebalancing aims to optimize risk-return profile",
                "Consider transaction costs in implementation",
                "Gradual implementation may reduce market impact"
            ],
            "implementation_priority": "Medium Priority",
            "execution_steps": ["Review recommendations", "Plan gradual implementation", "Monitor results"],
            "risk_considerations": ["Market timing", "Transaction costs", "Liquidity requirements"]
        }


# Global instance
groq_explainability_engine = GroqExplainabilityEngine()