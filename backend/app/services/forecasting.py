"""Advanced forecasting service using real statistical models and market data."""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.db.connection import db_manager
from app.utils.config import settings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class GARCHVolatilityModel:
    """GARCH model for volatility forecasting using real returns data."""

    def __init__(self):
        try:
            from arch import arch_model
            self.arch_model = arch_model
        except ImportError:
            logger.warning("ARCH library not available, using simplified volatility model")
            self.arch_model = None

    def fit_and_forecast(self, returns: pd.Series, horizon: int = 30) -> Dict:
        """Fit GARCH model and forecast volatility."""
        try:
            if self.arch_model is None:
                # Fallback to rolling volatility
                return self._rolling_volatility_forecast(returns, horizon)

            # Remove any NaN or infinite values
            clean_returns = returns.dropna()
            clean_returns = clean_returns[np.isfinite(clean_returns)]

            if len(clean_returns) < 50:
                return self._rolling_volatility_forecast(returns, horizon)

            # Fit GARCH(1,1) model
            model = self.arch_model(clean_returns * 100, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')

            # Forecast volatility
            forecasts = fitted_model.forecast(horizon=horizon)

            volatility_forecast = np.sqrt(forecasts.variance.values[-1, :]) / 100

            return {
                "volatility_forecast": volatility_forecast.tolist(),
                "model_params": {
                    "omega": float(fitted_model.params['omega']),
                    "alpha": float(fitted_model.params['alpha[1]']),
                    "beta": float(fitted_model.params['beta[1]'])
                },
                "log_likelihood": float(fitted_model.loglikelihood),
                "aic": float(fitted_model.aic)
            }

        except Exception as e:
            logger.warning(f"GARCH model failed: {e}, using rolling volatility")
            return self._rolling_volatility_forecast(returns, horizon)

    def _rolling_volatility_forecast(self, returns: pd.Series, horizon: int) -> Dict:
        """Fallback volatility forecast using rolling windows."""
        # Calculate rolling volatility with different windows
        vol_short = returns.rolling(window=10).std()
        vol_medium = returns.rolling(window=30).std()
        vol_long = returns.rolling(window=90).std()

        # Use EWMA for recent volatility weighting
        ewma_vol = returns.ewm(span=30).std()

        # Combine forecasts (ensemble approach)
        current_vol = ewma_vol.iloc[-1]
        trend_vol = (vol_short.iloc[-1] - vol_long.iloc[-1]) / vol_long.iloc[-1]

        # Simple forecast with mean reversion
        forecast_vols = []
        for i in range(horizon):
            # Gradual mean reversion to long-term volatility
            reversion_factor = 0.95 ** i
            forecast_vol = current_vol * reversion_factor + vol_long.iloc[-1] * (1 - reversion_factor)
            forecast_vols.append(forecast_vol)

        return {
            "volatility_forecast": forecast_vols,
            "model_params": {"type": "rolling_ewma"},
            "current_volatility": float(current_vol)
        }


class LSTMPriceModel(nn.Module):
    """LSTM neural network for price forecasting."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2):
        super(LSTMPriceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ARIMAForecaster:
    """ARIMA model implementation for time series forecasting."""

    def __init__(self):
        self.model = None
        self.fitted_model = None

    def find_optimal_order(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using AIC criterion."""
        best_aic = float('inf')
        best_order = (1, 1, 1)

        # Test stationarity and determine d
        d = 0
        test_data = data.copy()

        for i in range(max_d + 1):
            adf_result = adfuller(test_data.dropna())
            if adf_result[1] <= 0.05:  # p-value threshold for stationarity
                d = i
                break
            test_data = test_data.diff()

        # Grid search for p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue

        logger.info(f"Optimal ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        return best_order

    def fit_and_forecast(self, data: pd.Series, horizon: int = 30, confidence_levels: List[float] = None) -> Dict:
        """Fit ARIMA model and generate forecasts with confidence intervals."""
        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95]

        try:
            # Find optimal order
            order = self.find_optimal_order(data)

            # Fit model
            self.model = ARIMA(data, order=order)
            self.fitted_model = self.model.fit()

            # Generate forecasts
            forecast_result = self.fitted_model.forecast(steps=horizon)
            forecast_values = forecast_result

            # Get confidence intervals
            conf_intervals = {}
            for level in confidence_levels:
                get_forecast = self.fitted_model.get_forecast(steps=horizon, alpha=(1 - level))
                conf_int = get_forecast.conf_int()
                conf_intervals[f"{int(level*100)}%"] = {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                }

            # Calculate model metrics
            residuals = self.fitted_model.resid
            metrics = {
                "aic": float(self.fitted_model.aic),
                "bic": float(self.fitted_model.bic),
                "mae": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "ljung_box_p": float(self.fitted_model.diagnostic_summary().iloc[2, 3])  # Ljung-Box p-value
            }

            return {
                "forecast": forecast_values.tolist(),
                "confidence_intervals": conf_intervals,
                "model_order": order,
                "metrics": metrics,
                "model_summary": str(self.fitted_model.summary())
            }

        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            raise


class ProphetForecaster:
    """Facebook Prophet implementation for trend and seasonality analysis."""

    def __init__(self):
        self.model = None

    def fit_and_forecast(self, data: pd.DataFrame, horizon: int = 30, confidence_interval: float = 0.95) -> Dict:
        """Fit Prophet model and generate forecasts."""
        try:
            # Prepare data in Prophet format
            prophet_data = data.copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])

            # Initialize and fit model
            self.model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='multiplicative',
                interval_width=confidence_interval
            )

            # Add custom seasonalities for financial markets
            self.model.add_seasonality(name='weekly', period=7, fourier_order=3)
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            self.model.fit(prophet_data)

            # Create future dataframe
            future = self.model.make_future_dataframe(periods=horizon, freq='D')

            # Generate forecast
            forecast = self.model.predict(future)

            # Extract forecast values and confidence intervals
            forecast_values = forecast['yhat'][-horizon:].tolist()
            lower_bound = forecast['yhat_lower'][-horizon:].tolist()
            upper_bound = forecast['yhat_upper'][-horizon:].tolist()

            # Calculate components
            trend = forecast['trend'][-horizon:].tolist()
            weekly_seasonality = forecast.get('weekly', [0] * horizon)
            monthly_seasonality = forecast.get('monthly', [0] * horizon)

            # Model metrics
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'][:len(y_true)].values

            metrics = {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            }

            return {
                "forecast": forecast_values,
                "confidence_intervals": {
                    f"{int(confidence_interval*100)}%": {
                        "lower": lower_bound,
                        "upper": upper_bound
                    }
                },
                "components": {
                    "trend": trend,
                    "weekly_seasonality": weekly_seasonality,
                    "monthly_seasonality": monthly_seasonality
                },
                "metrics": metrics,
                "changepoints": self.model.changepoints.dt.strftime('%Y-%m-%d').tolist()
            }

        except Exception as e:
            logger.error(f"Prophet forecasting failed: {e}")
            raise


class ForecastingService:
    """Main forecasting service coordinating multiple models."""

    def __init__(self):
        self.arima_forecaster = ARIMAForecaster()
        self.prophet_forecaster = ProphetForecaster()
        self.garch_model = GARCHVolatilityModel()

    async def generate_price_forecast(self, symbol: str, horizon: int = 30, models: List[str] = None) -> Dict:
        """Generate comprehensive price forecasts using multiple models."""
        if models is None:
            models = ["arima", "prophet"]

        try:
            # Fetch historical data
            historical_data = await db_manager.get_price_history(symbol, days=max(180, horizon * 3))

            if len(historical_data) < settings.min_historical_days:
                raise ValueError(f"Insufficient data for {symbol}: need at least {settings.min_historical_days} days")

            # Convert to pandas DataFrame
            df = pd.DataFrame(historical_data)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')

            price_series = df.set_index('time')['price']

            forecasts = {}

            # ARIMA forecast
            if "arima" in models:
                try:
                    arima_result = self.arima_forecaster.fit_and_forecast(
                        price_series, horizon, settings.confidence_intervals
                    )
                    forecasts["arima"] = arima_result
                except Exception as e:
                    logger.warning(f"ARIMA forecast failed for {symbol}: {e}")

            # Prophet forecast
            if "prophet" in models:
                try:
                    prophet_data = df[['time', 'price']].copy()
                    prophet_result = self.prophet_forecaster.fit_and_forecast(prophet_data, horizon)
                    forecasts["prophet"] = prophet_result
                except Exception as e:
                    logger.warning(f"Prophet forecast failed for {symbol}: {e}")

            # Volatility forecast
            returns = price_series.pct_change().dropna()
            volatility_result = self.garch_model.fit_and_forecast(returns, horizon)
            forecasts["volatility"] = volatility_result

            # Ensemble forecast (combine models)
            if len(forecasts) > 1:
                ensemble_forecast = self._create_ensemble_forecast(forecasts, horizon)
                forecasts["ensemble"] = ensemble_forecast

            # Store forecast in database
            await self._store_forecast_results(symbol, forecasts, horizon)

            return {
                "symbol": symbol,
                "forecast_horizon": horizon,
                "generated_at": datetime.now().isoformat(),
                "forecasts": forecasts,
                "data_points_used": len(price_series),
                "last_price": float(price_series.iloc[-1])
            }

        except Exception as e:
            logger.error(f"Forecast generation failed for {symbol}: {e}")
            raise

    def _create_ensemble_forecast(self, forecasts: Dict, horizon: int) -> Dict:
        """Create ensemble forecast by combining multiple models."""
        available_models = [k for k in forecasts.keys() if k != "volatility"]

        if not available_models:
            return {}

        # Weighted average of forecasts (equal weights for simplicity)
        ensemble_values = np.zeros(horizon)
        total_weight = 0

        for model_name in available_models:
            if "forecast" in forecasts[model_name]:
                model_forecast = np.array(forecasts[model_name]["forecast"])
                weight = 1.0  # Equal weights
                ensemble_values += weight * model_forecast
                total_weight += weight

        if total_weight > 0:
            ensemble_values /= total_weight

        # Calculate ensemble confidence intervals (conservative approach)
        confidence_intervals = {}
        for level in settings.confidence_intervals:
            level_key = f"{int(level*100)}%"
            lower_bounds = []
            upper_bounds = []

            for model_name in available_models:
                if level_key in forecasts[model_name].get("confidence_intervals", {}):
                    lower_bounds.append(forecasts[model_name]["confidence_intervals"][level_key]["lower"])
                    upper_bounds.append(forecasts[model_name]["confidence_intervals"][level_key]["upper"])

            if lower_bounds and upper_bounds:
                # Use widest bounds (most conservative)
                confidence_intervals[level_key] = {
                    "lower": np.min(lower_bounds, axis=0).tolist(),
                    "upper": np.max(upper_bounds, axis=0).tolist()
                }

        return {
            "forecast": ensemble_values.tolist(),
            "confidence_intervals": confidence_intervals,
            "models_used": available_models,
            "method": "weighted_average"
        }

    async def _store_forecast_results(self, symbol: str, forecasts: Dict, horizon: int):
        """Store forecast results in database."""
        try:
            for model_name, forecast_data in forecasts.items():
                # Create forecast dates
                start_date = datetime.now().date()
                forecast_dates = [(start_date + timedelta(days=i)).isoformat() for i in range(1, horizon + 1)]

                forecast_entry = {
                    "symbol": symbol,
                    "forecast_type": "price",
                    "forecast_horizon": horizon,
                    "model_name": model_name,
                    "forecast_data": {
                        "dates": forecast_dates,
                        "values": forecast_data.get("forecast", []),
                        "confidence_intervals": forecast_data.get("confidence_intervals", {})
                    },
                    "model_metrics": forecast_data.get("metrics", {})
                }

                # Calculate accuracy score if available
                if "metrics" in forecast_data and "mae" in forecast_data["metrics"]:
                    # Simple accuracy score based on MAE (lower is better)
                    mae = forecast_data["metrics"]["mae"]
                    last_price = await self._get_last_price(symbol)
                    if last_price:
                        accuracy_score = max(0, 1 - (mae / last_price))
                        forecast_entry["accuracy_score"] = accuracy_score

                # Insert into database (using raw query for JSONB)
                query = """
                INSERT INTO forecasts (symbol, forecast_type, forecast_horizon, model_name,
                                     forecast_data, model_metrics, accuracy_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """
                await db_manager.execute_raw_query(query, {
                    "symbol": forecast_entry["symbol"],
                    "forecast_type": forecast_entry["forecast_type"],
                    "forecast_horizon": forecast_entry["forecast_horizon"],
                    "model_name": forecast_entry["model_name"],
                    "forecast_data": forecast_entry["forecast_data"],
                    "model_metrics": forecast_entry["model_metrics"],
                    "accuracy_score": forecast_entry.get("accuracy_score")
                })

        except Exception as e:
            logger.error(f"Failed to store forecast results: {e}")

    async def _get_last_price(self, symbol: str) -> Optional[float]:
        """Get the last known price for a symbol."""
        try:
            latest_prices = await db_manager.get_latest_prices([symbol])
            if latest_prices:
                return latest_prices[0]["price"]
        except Exception as e:
            logger.error(f"Failed to get last price for {symbol}: {e}")
        return None

    async def get_forecast_accuracy(self, symbol: str, days_back: int = 30) -> Dict:
        """Calculate forecast accuracy by comparing past predictions with actual prices."""
        try:
            # Get historical forecasts
            query = """
            SELECT * FROM forecasts
            WHERE symbol = $1
            AND created_at >= NOW() - INTERVAL '%s days'
            ORDER BY created_at DESC
            """ % days_back

            historical_forecasts = await db_manager.execute_raw_query(query, {"symbol": symbol})

            if not historical_forecasts:
                return {"error": "No historical forecasts found"}

            accuracy_results = {}

            for forecast in historical_forecasts:
                forecast_date = forecast["created_at"]
                forecast_data = forecast["forecast_data"]
                model_name = forecast["model_name"]

                # Get actual prices for comparison
                actual_prices = await db_manager.get_price_history(
                    symbol,
                    days=(datetime.now() - forecast_date).days + forecast["forecast_horizon"]
                )

                if actual_prices:
                    # Calculate accuracy metrics
                    predicted = forecast_data["values"]
                    actual = [p["price"] for p in actual_prices[-len(predicted):]]

                    if len(actual) == len(predicted):
                        mae = mean_absolute_error(actual, predicted)
                        rmse = np.sqrt(mean_squared_error(actual, predicted))
                        mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100

                        accuracy_results[f"{model_name}_{forecast_date.strftime('%Y%m%d')}"] = {
                            "mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                            "forecast_date": forecast_date.isoformat(),
                            "data_points": len(actual)
                        }

            return accuracy_results

        except Exception as e:
            logger.error(f"Failed to calculate forecast accuracy: {e}")
            return {"error": str(e)}


# Global forecasting service instance
forecasting_service = ForecastingService()