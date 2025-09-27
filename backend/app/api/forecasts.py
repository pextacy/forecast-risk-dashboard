"""API endpoints for forecasting functionality."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.forecasting import forecasting_service
from app.services.explainability import explainability_engine
from app.services.ingestion import ingestion_service
from app.db.connection import db_manager
from app.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class ForecastRequest(BaseModel):
    symbol: str = Field(..., description="Asset symbol to forecast")
    horizon_days: int = Field(30, ge=1, le=90, description="Forecast horizon in days")
    models: Optional[List[str]] = Field(None, description="Models to use: arima, prophet, ensemble")
    include_explanation: bool = Field(True, description="Include AI-generated explanation")


class ForecastResponse(BaseModel):
    symbol: str
    forecast_horizon: int
    generated_at: str
    forecasts: Dict
    data_points_used: int
    last_price: float
    explanation: Optional[Dict] = None


class BatchForecastRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of asset symbols")
    horizon_days: int = Field(30, ge=1, le=90)
    models: Optional[List[str]] = None
    include_explanation: bool = Field(True)


class AccuracyRequest(BaseModel):
    symbol: str
    lookback_days: int = Field(30, ge=7, le=180, description="Days to look back for accuracy analysis")


class AccuracyResponse(BaseModel):
    symbol: str
    accuracy_results: Dict
    summary_metrics: Dict


class SupportedAssetsResponse(BaseModel):
    crypto_assets: List[str]
    stock_assets: List[str]
    total_count: int


@router.get("/supported-assets", response_model=SupportedAssetsResponse)
async def get_supported_assets():
    """Get list of supported assets for forecasting."""
    return SupportedAssetsResponse(
        crypto_assets=settings.supported_crypto_symbols,
        stock_assets=settings.supported_stock_symbols,
        total_count=len(settings.supported_crypto_symbols) + len(settings.supported_stock_symbols)
    )


@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate forecast for a single asset."""
    try:
        # Validate symbol
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        if request.symbol not in all_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported symbol: {request.symbol}. Use /supported-assets to see available symbols."
            )

        # Ensure we have recent data
        try:
            await _ensure_data_availability(request.symbol)
        except Exception as e:
            logger.warning(f"Data availability check failed for {request.symbol}: {e}")

        # Generate forecast
        forecast_result = await forecasting_service.generate_price_forecast(
            symbol=request.symbol,
            horizon=request.horizon_days,
            models=request.models
        )

        response = ForecastResponse(**forecast_result)

        # Add explanation if requested
        if request.include_explanation:
            try:
                current_price = forecast_result["last_price"]
                historical_volatility = await _get_historical_volatility(request.symbol)

                explanation = await explainability_engine.explain_forecast_results(
                    symbol=request.symbol,
                    forecast_data=forecast_result,
                    current_price=current_price,
                    historical_volatility=historical_volatility or 0.2
                )
                response.explanation = explanation

            except Exception as e:
                logger.warning(f"Explanation generation failed for {request.symbol}: {e}")
                response.explanation = {"error": "Explanation unavailable", "details": str(e)}

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast generation failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@router.post("/batch", response_model=List[ForecastResponse])
async def generate_batch_forecasts(request: BatchForecastRequest):
    """Generate forecasts for multiple assets."""
    try:
        # Validate symbols
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        invalid_symbols = [s for s in request.symbols if s not in all_symbols]
        if invalid_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported symbols: {invalid_symbols}"
            )

        # Limit batch size
        if len(request.symbols) > 10:
            raise HTTPException(
                status_code=400,
                detail="Batch size limited to 10 symbols"
            )

        results = []
        for symbol in request.symbols:
            try:
                # Create individual request
                individual_request = ForecastRequest(
                    symbol=symbol,
                    horizon_days=request.horizon_days,
                    models=request.models,
                    include_explanation=request.include_explanation
                )

                # Generate forecast
                forecast_response = await generate_forecast(individual_request)
                results.append(forecast_response)

            except Exception as e:
                logger.warning(f"Batch forecast failed for {symbol}: {e}")
                # Add error result
                error_response = ForecastResponse(
                    symbol=symbol,
                    forecast_horizon=request.horizon_days,
                    generated_at=datetime.now().isoformat(),
                    forecasts={"error": str(e)},
                    data_points_used=0,
                    last_price=0.0
                )
                results.append(error_response)

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch forecast failed: {str(e)}")


@router.get("/accuracy/{symbol}", response_model=AccuracyResponse)
async def get_forecast_accuracy(symbol: str, lookback_days: int = Query(30, ge=7, le=180)):
    """Get forecast accuracy analysis for a symbol."""
    try:
        # Validate symbol
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        if symbol not in all_symbols:
            raise HTTPException(status_code=400, detail=f"Unsupported symbol: {symbol}")

        # Get accuracy results
        accuracy_results = await forecasting_service.get_forecast_accuracy(symbol, lookback_days)

        if "error" in accuracy_results:
            raise HTTPException(status_code=404, detail=accuracy_results["error"])

        # Calculate summary metrics
        summary_metrics = _calculate_accuracy_summary(accuracy_results)

        return AccuracyResponse(
            symbol=symbol,
            accuracy_results=accuracy_results,
            summary_metrics=summary_metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Accuracy analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_forecast_history(
    symbol: str,
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0)
):
    """Get historical forecasts for a symbol."""
    try:
        query = """
        SELECT id, created_at, forecast_type, forecast_horizon, model_name,
               forecast_data, model_metrics, accuracy_score
        FROM forecasts
        WHERE symbol = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        """

        forecasts = await db_manager.execute_raw_query(query, {
            "symbol": symbol,
            "limit": limit,
            "offset": offset
        })

        return {
            "symbol": symbol,
            "forecasts": forecasts,
            "total_returned": len(forecasts)
        }

    except Exception as e:
        logger.error(f"Forecast history query failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_forecasts(limit: int = Query(20, ge=1, le=100)):
    """Get latest forecasts across all symbols."""
    try:
        query = """
        SELECT DISTINCT ON (symbol, model_name) symbol, created_at, forecast_type,
               forecast_horizon, model_name, forecast_data, accuracy_score
        FROM forecasts
        WHERE created_at >= NOW() - INTERVAL '7 days'
        ORDER BY symbol, model_name, created_at DESC
        LIMIT $1
        """

        latest_forecasts = await db_manager.execute_raw_query(query, {"limit": limit})

        return {
            "latest_forecasts": latest_forecasts,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Latest forecasts query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh-data/{symbol}")
async def refresh_symbol_data(symbol: str, background_tasks: BackgroundTasks):
    """Refresh historical data for a specific symbol."""
    try:
        # Validate symbol
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        if symbol not in all_symbols:
            raise HTTPException(status_code=400, detail=f"Unsupported symbol: {symbol}")

        async def refresh_data():
            try:
                records_count = await ingestion_service.ingest_historical_data(symbol, days=90)
                logger.info(f"Refreshed {records_count} records for {symbol}")
            except Exception as e:
                logger.error(f"Data refresh failed for {symbol}: {e}")

        background_tasks.add_task(refresh_data)

        return {
            "message": f"Data refresh started for {symbol}",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data refresh trigger failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility/{symbol}")
async def get_volatility_analysis(
    symbol: str,
    window_days: int = Query(30, ge=7, le=180)
):
    """Get volatility analysis for a symbol."""
    try:
        # Calculate current volatility
        current_volatility = await ingestion_service.calculate_real_volatility(symbol, window_days)

        if current_volatility is None:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data for volatility calculation for {symbol}"
            )

        # Get price history for volatility chart
        price_history = await db_manager.get_price_history(symbol, window_days)

        # Calculate rolling volatility
        if len(price_history) >= 10:
            import pandas as pd
            import numpy as np

            df = pd.DataFrame(price_history)
            df['returns'] = df['price'].pct_change()

            # Calculate different volatility windows
            volatility_data = {
                "current_volatility": current_volatility,
                "volatility_7d": float(df['returns'].tail(7).std() * np.sqrt(252)),
                "volatility_30d": float(df['returns'].tail(30).std() * np.sqrt(252)),
                "volatility_90d": float(df['returns'].std() * np.sqrt(252)),
                "price_data": [
                    {"time": p["time"].isoformat(), "price": p["price"], "volume": p.get("volume")}
                    for p in price_history
                ]
            }

            return {
                "symbol": symbol,
                "analysis_window": window_days,
                "volatility_analysis": volatility_data,
                "generated_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Insufficient data for detailed volatility analysis"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Volatility analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation")
async def get_correlation_analysis(
    symbols: List[str] = Query(..., description="List of symbols to analyze"),
    days: int = Query(30, ge=7, le=180)
):
    """Get correlation analysis between multiple symbols."""
    try:
        # Validate symbols
        all_symbols = settings.supported_crypto_symbols + settings.supported_stock_symbols
        invalid_symbols = [s for s in symbols if s not in all_symbols]
        if invalid_symbols:
            raise HTTPException(status_code=400, detail=f"Unsupported symbols: {invalid_symbols}")

        if len(symbols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for correlation analysis")

        if len(symbols) > 15:
            raise HTTPException(status_code=400, detail="Maximum 15 symbols allowed")

        # Calculate correlation matrix
        correlation_matrix = await ingestion_service.get_correlation_matrix(symbols, days)

        if correlation_matrix is None:
            raise HTTPException(
                status_code=404,
                detail="Insufficient data for correlation analysis"
            )

        return {
            "symbols": symbols,
            "analysis_period": days,
            "correlation_matrix": correlation_matrix.to_dict(),
            "average_correlation": float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def _ensure_data_availability(symbol: str, min_days: int = 30):
    """Ensure sufficient data is available for forecasting."""
    price_history = await db_manager.get_price_history(symbol, min_days)

    if len(price_history) < min_days:
        # Try to fetch more data
        await ingestion_service.ingest_historical_data(symbol, days=90)

        # Check again
        price_history = await db_manager.get_price_history(symbol, min_days)
        if len(price_history) < min_days:
            raise ValueError(f"Insufficient data for {symbol}: only {len(price_history)} days available")


async def _get_historical_volatility(symbol: str, days: int = 30) -> Optional[float]:
    """Get historical volatility for a symbol."""
    try:
        return await ingestion_service.calculate_real_volatility(symbol, days)
    except Exception as e:
        logger.warning(f"Historical volatility calculation failed for {symbol}: {e}")
        return None


def _calculate_accuracy_summary(accuracy_results: Dict) -> Dict:
    """Calculate summary metrics from accuracy results."""
    if not accuracy_results:
        return {}

    # Extract MAE and RMSE values
    mae_values = []
    rmse_values = []
    mape_values = []

    for result in accuracy_results.values():
        if isinstance(result, dict):
            if "mae" in result:
                mae_values.append(result["mae"])
            if "rmse" in result:
                rmse_values.append(result["rmse"])
            if "mape" in result:
                mape_values.append(result["mape"])

    summary = {}
    if mae_values:
        summary["average_mae"] = sum(mae_values) / len(mae_values)
        summary["best_mae"] = min(mae_values)
        summary["worst_mae"] = max(mae_values)

    if rmse_values:
        summary["average_rmse"] = sum(rmse_values) / len(rmse_values)

    if mape_values:
        summary["average_mape"] = sum(mape_values) / len(mape_values)

    summary["total_forecasts_analyzed"] = len(accuracy_results)

    return summary