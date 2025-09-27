"""Comprehensive backtesting framework for forecast and hedge validation."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.db.connection import db_manager
from app.services.forecasting import ARIMAForecaster, ProphetForecaster, GARCHVolatilityModel
from app.services.risk_metrics import risk_calculator
from app.utils.config import settings

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results."""

    def __init__(self, model_name: str, symbol: str):
        self.model_name = model_name
        self.symbol = symbol
        self.predictions = []
        self.actuals = []
        self.prediction_dates = []
        self.metrics = {}
        self.rolling_metrics = {}

    def add_prediction(self, date: datetime, predicted: float, actual: float):
        """Add a prediction result."""
        self.prediction_dates.append(date)
        self.predictions.append(predicted)
        self.actuals.append(actual)

    def calculate_metrics(self):
        """Calculate performance metrics."""
        if len(self.predictions) == 0:
            return

        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)

        # Basic metrics
        self.metrics = {
            'mae': float(mean_absolute_error(actuals, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
            'mape': float(np.mean(np.abs((actuals - predictions) / actuals)) * 100),
            'directional_accuracy': float(np.mean(np.sign(predictions[1:] - predictions[:-1]) ==
                                                  np.sign(actuals[1:] - actuals[:-1]))),
            'correlation': float(np.corrcoef(predictions, actuals)[0, 1]) if len(predictions) > 1 else 0,
            'total_predictions': len(predictions)
        }

        # Calculate rolling metrics (if enough data)
        if len(predictions) >= 10:
            window = min(10, len(predictions) // 2)
            rolling_mae = []
            rolling_mape = []

            for i in range(window, len(predictions)):
                window_predictions = predictions[i-window:i]
                window_actuals = actuals[i-window:i]

                mae = mean_absolute_error(window_actuals, window_predictions)
                mape = np.mean(np.abs((window_actuals - window_predictions) / window_actuals)) * 100

                rolling_mae.append(mae)
                rolling_mape.append(mape)

            self.rolling_metrics = {
                'rolling_mae': rolling_mae,
                'rolling_mape': rolling_mape,
                'mae_trend': 'improving' if rolling_mae[-1] < rolling_mae[0] else 'degrading',
                'mape_trend': 'improving' if rolling_mape[-1] < rolling_mape[0] else 'degrading'
            }


class ForecastBacktester:
    """Backtest forecasting models using historical data."""

    def __init__(self):
        self.arima_forecaster = ARIMAForecaster()
        self.prophet_forecaster = ProphetForecaster()
        self.garch_model = GARCHVolatilityModel()

    async def run_forecast_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        forecast_horizon: int = 7,
        refit_frequency: int = 30,
        models: List[str] = None
    ) -> Dict[str, BacktestResult]:
        """Run walk-forward backtest for forecasting models."""

        if models is None:
            models = ['arima', 'prophet']

        try:
            # Get historical data
            total_days = (end_date - start_date).days + 60  # Extra buffer for initial training
            historical_data = await db_manager.get_price_history(symbol, total_days)

            if len(historical_data) < 90:
                raise ValueError(f"Insufficient data for backtesting {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').set_index('time')

            # Filter to backtest period
            mask = (df.index >= start_date) & (df.index <= end_date)
            backtest_data = df[mask]

            if len(backtest_data) < forecast_horizon:
                raise ValueError(f"Backtest period too short for {forecast_horizon}-day forecasts")

            results = {}

            # Run backtest for each model
            for model_name in models:
                logger.info(f"Running {model_name} backtest for {symbol}")
                result = BacktestResult(model_name, symbol)

                # Walk-forward validation
                current_date = start_date
                refit_counter = 0

                while current_date + timedelta(days=forecast_horizon) <= end_date:
                    try:
                        # Get training data up to current date
                        training_end = current_date
                        training_start = training_end - timedelta(days=90)  # 3 months training

                        training_mask = (df.index >= training_start) & (df.index < training_end)
                        training_data = df[training_mask]['price']

                        if len(training_data) < 30:
                            current_date += timedelta(days=1)
                            continue

                        # Refit model if needed
                        if refit_counter % refit_frequency == 0:
                            if model_name == 'arima':
                                fitted_model = await self._fit_arima_model(training_data)
                            elif model_name == 'prophet':
                                fitted_model = await self._fit_prophet_model(training_data)

                        # Generate forecast
                        if model_name == 'arima' and fitted_model:
                            forecast = await self._generate_arima_forecast(fitted_model, forecast_horizon)
                        elif model_name == 'prophet' and fitted_model:
                            forecast = await self._generate_prophet_forecast(fitted_model, forecast_horizon, current_date)
                        else:
                            current_date += timedelta(days=1)
                            continue

                        # Get actual value
                        target_date = current_date + timedelta(days=forecast_horizon)
                        actual_mask = (df.index >= target_date) & (df.index < target_date + timedelta(days=1))

                        if actual_mask.sum() > 0:
                            actual_value = df[actual_mask]['price'].iloc[0]
                            predicted_value = forecast[-1] if isinstance(forecast, list) else forecast

                            result.add_prediction(target_date, predicted_value, actual_value)

                        refit_counter += 1
                        current_date += timedelta(days=1)

                    except Exception as e:
                        logger.warning(f"Forecast failed for {current_date}: {e}")
                        current_date += timedelta(days=1)
                        continue

                # Calculate metrics
                result.calculate_metrics()
                results[model_name] = result

                logger.info(f"Completed {model_name} backtest: {result.metrics.get('total_predictions', 0)} predictions")

            return results

        except Exception as e:
            logger.error(f"Forecast backtest failed for {symbol}: {e}")
            raise

    async def _fit_arima_model(self, training_data: pd.Series):
        """Fit ARIMA model on training data."""
        try:
            order = self.arima_forecaster.find_optimal_order(training_data)
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(training_data, order=order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}")
            return None

    async def _fit_prophet_model(self, training_data: pd.Series):
        """Fit Prophet model on training data."""
        try:
            from prophet import Prophet

            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': training_data.index,
                'y': training_data.values
            })

            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                seasonality_mode='multiplicative',
                interval_width=0.95
            )

            model.fit(prophet_data)
            return model

        except Exception as e:
            logger.warning(f"Prophet fitting failed: {e}")
            return None

    async def _generate_arima_forecast(self, fitted_model, horizon: int):
        """Generate ARIMA forecast."""
        try:
            forecast_result = fitted_model.forecast(steps=horizon)
            return forecast_result.tolist() if hasattr(forecast_result, 'tolist') else [forecast_result]
        except Exception as e:
            logger.warning(f"ARIMA forecast generation failed: {e}")
            return None

    async def _generate_prophet_forecast(self, fitted_model, horizon: int, start_date: datetime):
        """Generate Prophet forecast."""
        try:
            future = fitted_model.make_future_dataframe(periods=horizon, freq='D')
            future = future[future['ds'] > start_date]  # Only future dates

            forecast = fitted_model.predict(future)
            return forecast['yhat'].tolist()

        except Exception as e:
            logger.warning(f"Prophet forecast generation failed: {e}")
            return None


class RiskBacktester:
    """Backtest risk metrics and hedge suggestions."""

    async def backtest_risk_metrics(
        self,
        portfolio_symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        rebalance_frequency: int = 30
    ) -> Dict:
        """Backtest portfolio risk metrics over time."""

        try:
            # Get historical data for all symbols
            total_days = (end_date - start_date).days + 90
            price_data = {}

            for symbol in portfolio_symbols:
                history = await db_manager.get_price_history(symbol, total_days)
                if history:
                    df = pd.DataFrame(history)
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.sort_values('time').set_index('time')
                    price_data[symbol] = df['price']

            if not price_data:
                raise ValueError("No price data available for portfolio symbols")

            # Combine price data
            combined_prices = pd.DataFrame(price_data).dropna()
            returns = combined_prices.pct_change().dropna()

            # Calculate equal weights for simplicity
            weights = {symbol: 1.0/len(portfolio_symbols) for symbol in portfolio_symbols}

            # Run rolling risk analysis
            risk_results = []
            current_date = start_date

            while current_date <= end_date:
                try:
                    # Get lookback window
                    lookback_start = current_date - timedelta(days=90)

                    window_mask = (returns.index >= lookback_start) & (returns.index <= current_date)
                    window_returns = returns[window_mask]

                    if len(window_returns) >= 30:
                        # Calculate portfolio returns
                        portfolio_returns = (window_returns * pd.Series(weights)).sum(axis=1)

                        # Calculate risk metrics
                        metrics = self._calculate_risk_snapshot(portfolio_returns, current_date)
                        risk_results.append(metrics)

                    current_date += timedelta(days=rebalance_frequency)

                except Exception as e:
                    logger.warning(f"Risk calculation failed for {current_date}: {e}")
                    current_date += timedelta(days=rebalance_frequency)
                    continue

            return {
                'portfolio_symbols': portfolio_symbols,
                'backtest_period': f"{start_date.date()} to {end_date.date()}",
                'risk_snapshots': risk_results,
                'total_snapshots': len(risk_results)
            }

        except Exception as e:
            logger.error(f"Risk backtest failed: {e}")
            raise

    def _calculate_risk_snapshot(self, portfolio_returns: pd.Series, date: datetime) -> Dict:
        """Calculate risk metrics snapshot."""

        # Basic metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        mean_return = portfolio_returns.mean() * 252

        # VaR
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)

        # Sharpe ratio
        risk_free_rate = 0.02
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'date': date.isoformat(),
            'volatility': float(volatility),
            'var_95': float(var_95),
            'var_99': float(var_99),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'mean_return': float(mean_return)
        }


class PerformanceAnalyzer:
    """Analyze backtest results and generate insights."""

    def analyze_forecast_performance(self, results: Dict[str, BacktestResult]) -> Dict:
        """Analyze forecast backtest results."""

        analysis = {
            'model_comparison': {},
            'best_model': None,
            'overall_insights': []
        }

        best_mae = float('inf')
        best_model = None

        for model_name, result in results.items():
            metrics = result.metrics

            analysis['model_comparison'][model_name] = {
                'mae': metrics.get('mae', 0),
                'rmse': metrics.get('rmse', 0),
                'mape': metrics.get('mape', 0),
                'directional_accuracy': metrics.get('directional_accuracy', 0),
                'correlation': metrics.get('correlation', 0),
                'total_predictions': metrics.get('total_predictions', 0),
                'performance_grade': self._grade_performance(metrics)
            }

            # Track best model
            if metrics.get('mae', float('inf')) < best_mae:
                best_mae = metrics.get('mae', float('inf'))
                best_model = model_name

        analysis['best_model'] = best_model

        # Generate insights
        if best_model:
            best_metrics = results[best_model].metrics

            if best_metrics.get('mape', 100) < 10:
                analysis['overall_insights'].append("Forecast accuracy is excellent (MAPE < 10%)")
            elif best_metrics.get('mape', 100) < 20:
                analysis['overall_insights'].append("Forecast accuracy is good (MAPE < 20%)")
            else:
                analysis['overall_insights'].append("Forecast accuracy needs improvement (MAPE > 20%)")

            if best_metrics.get('directional_accuracy', 0) > 0.6:
                analysis['overall_insights'].append("Strong directional prediction capability")
            else:
                analysis['overall_insights'].append("Directional prediction needs improvement")

        return analysis

    def _grade_performance(self, metrics: Dict) -> str:
        """Grade model performance."""
        mape = metrics.get('mape', 100)
        directional_accuracy = metrics.get('directional_accuracy', 0)

        score = 0

        # MAPE scoring
        if mape < 5:
            score += 40
        elif mape < 10:
            score += 30
        elif mape < 20:
            score += 20
        elif mape < 30:
            score += 10

        # Directional accuracy scoring
        if directional_accuracy > 0.7:
            score += 30
        elif directional_accuracy > 0.6:
            score += 20
        elif directional_accuracy > 0.5:
            score += 10

        # Correlation scoring
        correlation = metrics.get('correlation', 0)
        if correlation > 0.8:
            score += 30
        elif correlation > 0.6:
            score += 20
        elif correlation > 0.4:
            score += 10

        if score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'


# Global backtest instances
forecast_backtester = ForecastBacktester()
risk_backtester = RiskBacktester()
performance_analyzer = PerformanceAnalyzer()