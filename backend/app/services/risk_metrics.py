"""Real portfolio risk metrics calculation using actual market data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import empyrical

from app.db.connection import db_manager
from app.utils.config import settings

logger = logging.getLogger(__name__)


class PortfolioRiskCalculator:
    """Calculate comprehensive portfolio risk metrics using real market data."""

    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate (US Treasury)

    async def calculate_portfolio_metrics(
        self,
        portfolio_weights: Dict[str, float],
        lookback_days: int = 90,
        confidence_levels: List[float] = None
    ) -> Dict:
        """Calculate comprehensive portfolio risk metrics."""
        if confidence_levels is None:
            confidence_levels = settings.var_confidence_levels

        try:
            # Validate portfolio weights
            if not self._validate_portfolio_weights(portfolio_weights):
                raise ValueError("Portfolio weights must sum to 1.0")

            # Get historical price data for all assets
            price_data = {}
            for symbol, weight in portfolio_weights.items():
                if weight > 0:  # Only fetch data for assets with positive weights
                    history = await db_manager.get_price_history(symbol, lookback_days)
                    if history:
                        df = pd.DataFrame(history)
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.sort_values('time').set_index('time')
                        price_data[symbol] = df['price']

            if not price_data:
                raise ValueError("No price data available for portfolio assets")

            # Combine price data and calculate returns
            combined_prices = pd.DataFrame(price_data)
            combined_prices = combined_prices.dropna()

            if len(combined_prices) < 30:
                raise ValueError("Insufficient price data for reliable risk calculations")

            # Calculate daily returns
            returns = combined_prices.pct_change().dropna()

            # Calculate portfolio returns
            weights_series = pd.Series(portfolio_weights)
            portfolio_returns = (returns * weights_series).sum(axis=1)

            # Calculate individual metrics
            metrics = {}

            # Basic risk metrics
            metrics.update(self._calculate_basic_metrics(portfolio_returns))

            # Value at Risk (VaR)
            metrics.update(self._calculate_var_metrics(portfolio_returns, confidence_levels))

            # Expected Shortfall (Conditional VaR)
            metrics.update(self._calculate_expected_shortfall(portfolio_returns, confidence_levels))

            # Advanced portfolio metrics
            metrics.update(self._calculate_portfolio_metrics(returns, weights_series))

            # Risk-adjusted performance
            metrics.update(self._calculate_performance_metrics(portfolio_returns))

            # Correlation and diversification metrics
            metrics.update(self._calculate_diversification_metrics(returns, weights_series))

            # Maximum Drawdown analysis
            metrics.update(self._calculate_drawdown_metrics(portfolio_returns))

            # Beta calculation (if benchmark data available)
            beta_metrics = await self._calculate_beta_metrics(portfolio_returns, lookback_days)
            metrics.update(beta_metrics)

            # Store metrics in database
            await self._store_risk_metrics(metrics, portfolio_weights)

            return {
                "portfolio_id": "default",
                "calculation_date": datetime.now().isoformat(),
                "lookback_days": lookback_days,
                "portfolio_weights": portfolio_weights,
                "metrics": metrics,
                "data_quality": {
                    "total_observations": len(portfolio_returns),
                    "assets_with_data": len(price_data),
                    "date_range": {
                        "start": combined_prices.index.min().isoformat(),
                        "end": combined_prices.index.max().isoformat()
                    }
                }
            }

        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            raise

    def _validate_portfolio_weights(self, weights: Dict[str, float]) -> bool:
        """Validate that portfolio weights are valid."""
        total_weight = sum(weights.values())
        return abs(total_weight - 1.0) < 0.01  # Allow small rounding errors

    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict:
        """Calculate basic risk and return metrics."""
        annualized_return = empyrical.annual_return(returns)
        annualized_volatility = empyrical.annual_volatility(returns)

        return {
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "daily_volatility": float(returns.std()),
            "mean_return": float(returns.mean()),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
            "downside_deviation": float(empyrical.downside_risk(returns))
        }

    def _calculate_var_metrics(self, returns: pd.Series, confidence_levels: List[float]) -> Dict:
        """Calculate Value at Risk at different confidence levels."""
        var_metrics = {}

        for confidence in confidence_levels:
            # Historical VaR
            historical_var = np.percentile(returns, (1 - confidence) * 100)

            # Parametric VaR (assuming normal distribution)
            parametric_var = stats.norm.ppf(1 - confidence, returns.mean(), returns.std())

            # Modified VaR (Cornish-Fisher expansion for non-normal distributions)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            z_score = stats.norm.ppf(1 - confidence)

            # Cornish-Fisher adjustment
            modified_z = (z_score +
                         (z_score**2 - 1) * skew / 6 +
                         (z_score**3 - 3*z_score) * kurt / 24 -
                         (2*z_score**3 - 5*z_score) * skew**2 / 36)

            modified_var = returns.mean() + modified_z * returns.std()

            var_metrics[f"var_{int(confidence*100)}"] = {
                "historical": float(historical_var),
                "parametric": float(parametric_var),
                "modified": float(modified_var)
            }

        return var_metrics

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_levels: List[float]) -> Dict:
        """Calculate Expected Shortfall (Conditional VaR)."""
        es_metrics = {}

        for confidence in confidence_levels:
            # Historical Expected Shortfall
            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            tail_returns = returns[returns <= var_threshold]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_threshold

            es_metrics[f"expected_shortfall_{int(confidence*100)}"] = float(expected_shortfall)

        return es_metrics

    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: pd.Series) -> Dict:
        """Calculate portfolio-specific metrics."""
        # Portfolio variance decomposition
        cov_matrix = returns.cov() * 252  # Annualized covariance matrix
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

        # Individual asset contributions to portfolio risk
        marginal_contributions = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_contributions / portfolio_variance

        # Effective number of assets (diversification ratio)
        individual_volatilities = np.sqrt(np.diag(cov_matrix))
        portfolio_volatility = np.sqrt(portfolio_variance)
        diversification_ratio = np.dot(weights, individual_volatilities) / portfolio_volatility

        return {
            "portfolio_variance": float(portfolio_variance),
            "portfolio_volatility": float(portfolio_volatility),
            "diversification_ratio": float(diversification_ratio),
            "effective_assets": float(1 / np.sum(weights**2)),  # Inverse Herfindahl index
            "risk_contributions": {symbol: float(contrib) for symbol, contrib in
                                 zip(weights.index, risk_contributions)}
        }

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        sharpe_ratio = empyrical.sharpe_ratio(returns, risk_free=self.risk_free_rate)
        sortino_ratio = empyrical.sortino_ratio(returns, required_return=self.risk_free_rate)
        calmar_ratio = empyrical.calmar_ratio(returns)

        # Information ratio (vs risk-free rate)
        excess_returns = returns - self.risk_free_rate / 252
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Omega ratio
        omega_ratio = self._calculate_omega_ratio(returns, self.risk_free_rate / 252)

        return {
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "information_ratio": float(information_ratio),
            "omega_ratio": float(omega_ratio)
        }

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()
        return gains / losses if losses > 0 else np.inf

    def _calculate_diversification_metrics(self, returns: pd.DataFrame, weights: pd.Series) -> Dict:
        """Calculate correlation and diversification metrics."""
        correlation_matrix = returns.corr()

        # Average correlation
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        avg_correlation = correlation_matrix.values[mask].mean()

        # Weighted average correlation
        weighted_correlations = []
        for i, asset1 in enumerate(correlation_matrix.index):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid double counting
                    weight_product = weights[asset1] * weights[asset2]
                    correlation = correlation_matrix.iloc[i, j]
                    weighted_correlations.append(weight_product * correlation)

        weighted_avg_correlation = sum(weighted_correlations) / sum(
            weights[i] * weights[j] for i in range(len(weights)) for j in range(i+1, len(weights))
        ) if len(weights) > 1 else 0

        # Concentration metrics
        herfindahl_index = sum(w**2 for w in weights.values)
        concentration_ratio = max(weights.values)  # Largest single weight

        return {
            "average_correlation": float(avg_correlation),
            "weighted_avg_correlation": float(weighted_avg_correlation),
            "herfindahl_index": float(herfindahl_index),
            "concentration_ratio": float(concentration_ratio),
            "correlation_matrix": correlation_matrix.to_dict()
        }

    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate drawdown analysis metrics."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Drawdown duration analysis
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None

        for i, in_drawdown in enumerate(is_drawdown):
            if in_drawdown and start_idx is None:
                start_idx = i
            elif not in_drawdown and start_idx is not None:
                drawdown_periods.append(i - start_idx)
                start_idx = None

        # Handle case where drawdown period extends to end
        if start_idx is not None:
            drawdown_periods.append(len(is_drawdown) - start_idx)

        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

        return {
            "max_drawdown": float(max_drawdown),
            "average_drawdown": float(avg_drawdown),
            "average_drawdown_duration": float(avg_drawdown_duration),
            "max_drawdown_duration": float(max_drawdown_duration),
            "current_drawdown": float(drawdown.iloc[-1])
        }

    async def _calculate_beta_metrics(self, portfolio_returns: pd.Series, lookback_days: int) -> Dict:
        """Calculate beta against market benchmark (SPY or BTC)."""
        try:
            # Try to get SPY data as market benchmark
            benchmark_data = await db_manager.get_price_history("SPY", lookback_days)

            if not benchmark_data:
                # Fallback to Bitcoin as crypto benchmark
                benchmark_data = await db_manager.get_price_history("bitcoin", lookback_days)

            if not benchmark_data:
                return {"beta": None, "alpha": None, "r_squared": None}

            # Calculate benchmark returns
            benchmark_df = pd.DataFrame(benchmark_data)
            benchmark_df['time'] = pd.to_datetime(benchmark_df['time'])
            benchmark_df = benchmark_df.sort_values('time').set_index('time')
            benchmark_returns = benchmark_df['price'].pct_change().dropna()

            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 20:  # Need minimum data for reliable beta
                return {"beta": None, "alpha": None, "r_squared": None}

            aligned_portfolio = portfolio_returns.loc[common_dates]
            aligned_benchmark = benchmark_returns.loc[common_dates]

            # Calculate beta using linear regression
            covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Calculate alpha
            portfolio_mean = aligned_portfolio.mean() * 252  # Annualized
            benchmark_mean = aligned_benchmark.mean() * 252  # Annualized
            alpha = portfolio_mean - beta * benchmark_mean

            # Calculate R-squared
            correlation = np.corrcoef(aligned_portfolio, aligned_benchmark)[0, 1]
            r_squared = correlation**2

            return {
                "beta": float(beta),
                "alpha": float(alpha),
                "r_squared": float(r_squared),
                "correlation_with_market": float(correlation)
            }

        except Exception as e:
            logger.warning(f"Beta calculation failed: {e}")
            return {"beta": None, "alpha": None, "r_squared": None}

    async def _store_risk_metrics(self, metrics: Dict, portfolio_weights: Dict[str, float]):
        """Store calculated risk metrics in database."""
        try:
            # Extract key metrics for database storage
            db_metrics = {
                "var_95": metrics.get("var_95", {}).get("historical"),
                "var_99": metrics.get("var_99", {}).get("historical"),
                "expected_shortfall": metrics.get("expected_shortfall_95"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "volatility_30d": metrics.get("annualized_volatility"),
                "max_drawdown": metrics.get("max_drawdown"),
                "beta": metrics.get("beta")
            }

            query = """
            INSERT INTO risk_metrics (time, portfolio_id, var_95, var_99, expected_shortfall,
                                    sharpe_ratio, volatility_30d, max_drawdown, beta)
            VALUES (NOW(), 'default', $1, $2, $3, $4, $5, $6, $7)
            """

            await db_manager.execute_raw_query(query, db_metrics)

        except Exception as e:
            logger.error(f"Failed to store risk metrics: {e}")

    async def calculate_optimal_portfolio(
        self,
        symbols: List[str],
        lookback_days: int = 90,
        target_return: Optional[float] = None
    ) -> Dict:
        """Calculate optimal portfolio weights using Modern Portfolio Theory."""
        try:
            # Get historical data for all symbols
            price_data = {}
            for symbol in symbols:
                history = await db_manager.get_price_history(symbol, lookback_days)
                if history:
                    df = pd.DataFrame(history)
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.sort_values('time').set_index('time')
                    price_data[symbol] = df['price']

            if len(price_data) < 2:
                raise ValueError("Need at least 2 assets for portfolio optimization")

            # Calculate returns and covariance matrix
            combined_prices = pd.DataFrame(price_data).dropna()
            returns = combined_prices.pct_change().dropna()

            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252  # Annualized

            n_assets = len(symbols)

            # Optimization constraints
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            bounds = tuple((0, settings.max_single_asset_weight) for _ in range(n_assets))

            # Objective function (minimize negative Sharpe ratio)
            def negative_sharpe(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_vol

            # Optimize for maximum Sharpe ratio
            initial_guess = np.array([1/n_assets] * n_assets)
            result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = {symbol: float(weight) for symbol, weight in
                                 zip(symbols, result.x)}

                # Calculate portfolio metrics for optimal weights
                portfolio_return = np.sum(mean_returns * result.x)
                portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

                return {
                    "optimal_weights": optimal_weights,
                    "expected_return": float(portfolio_return),
                    "expected_volatility": float(portfolio_vol),
                    "sharpe_ratio": float(sharpe_ratio),
                    "optimization_success": True
                }
            else:
                return {"optimization_success": False, "error": result.message}

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {"optimization_success": False, "error": str(e)}


# Global risk calculator instance
risk_calculator = PortfolioRiskCalculator()