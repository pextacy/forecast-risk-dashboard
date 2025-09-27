"""FastAPI main application for the Treasury Risk Dashboard."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
import uvicorn

from app.api import forecasts, balances, hedge
from app.db.connection import db_manager
from app.services.ingestion import ingestion_service
from app.utils.config import settings, validate_environment

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('treasury_dashboard_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('treasury_dashboard_request_duration_seconds', 'Request duration')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Treasury Risk Dashboard API")

    # Validate environment configuration
    try:
        validate_environment()
        logger.info("Environment validation successful")
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        if settings.environment == "production":
            raise

    # Initialize database connection
    try:
        # Test database connection
        await db_manager.execute_raw_query("SELECT 1")
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        if settings.environment == "production":
            raise

    # Start background data ingestion
    if settings.environment in ["production", "development"]:
        try:
            # Initial data load
            crypto_count, stock_count = await ingestion_service.ingest_current_market_data()
            logger.info(f"Initial data load: {crypto_count} crypto + {stock_count} stock prices")
        except Exception as e:
            logger.warning(f"Initial data load failed: {e}")

    yield

    logger.info("Shutting down Treasury Risk Dashboard API")


# Create FastAPI application
app = FastAPI(
    title="Treasury Risk Dashboard API",
    description="AI-powered treasury management with forecasting and risk analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Middleware for metrics collection
@app.middleware("http")
async def add_metrics_middleware(request, call_next):
    """Add Prometheus metrics collection."""
    method = request.method
    endpoint = request.url.path

    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()

    with REQUEST_DURATION.time():
        response = await call_next(request)

    return response


# Health check models
class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str


class SystemStatus(BaseModel):
    database: str
    services: dict
    data_freshness: dict


# Include API routers
app.include_router(forecasts.router, prefix="/api/v1/forecasts", tags=["Forecasts"])
app.include_router(balances.router, prefix="/api/v1/balances", tags=["Portfolio"])
app.include_router(hedge.router, prefix="/api/v1/hedge", tags=["Risk Management"])


@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with basic health information."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        environment=settings.environment
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        environment=settings.environment
    )


@app.get("/status", response_model=SystemStatus)
async def system_status():
    """Comprehensive system status including database and services."""
    try:
        # Check database connection
        await db_manager.execute_raw_query("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"

    # Check service status
    services = {
        "ingestion": "healthy",  # Could add actual service checks
        "forecasting": "healthy",
        "risk_calculation": "healthy",
        "explainability": "healthy" if settings.openai_api_key else "limited"
    }

    # Check data freshness
    try:
        latest_prices = await db_manager.get_latest_prices()
        if latest_prices:
            latest_time = max(price["time"] for price in latest_prices)
            hours_ago = (datetime.now() - latest_time).total_seconds() / 3600
            data_freshness = {
                "last_update": latest_time.isoformat(),
                "hours_ago": round(hours_ago, 1),
                "status": "fresh" if hours_ago < 6 else "stale"
            }
        else:
            data_freshness = {"status": "no_data"}
    except Exception as e:
        logger.error(f"Data freshness check failed: {e}")
        data_freshness = {"status": "error", "error": str(e)}

    return SystemStatus(
        database=db_status,
        services=services,
        data_freshness=data_freshness
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.post("/admin/ingest-data")
async def trigger_data_ingestion(background_tasks: BackgroundTasks):
    """Manually trigger data ingestion (admin endpoint)."""
    async def ingest_data():
        try:
            crypto_count, stock_count = await ingestion_service.ingest_current_market_data()
            logger.info(f"Manual data ingestion: {crypto_count} crypto + {stock_count} stock prices")
        except Exception as e:
            logger.error(f"Manual data ingestion failed: {e}")

    background_tasks.add_task(ingest_data)
    return {"message": "Data ingestion started", "timestamp": datetime.now().isoformat()}


@app.get("/admin/database-stats")
async def get_database_stats():
    """Get database statistics (admin endpoint)."""
    try:
        # Get table row counts
        stats_query = """
        SELECT
            'asset_prices' as table_name,
            COUNT(*) as row_count,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(time) as earliest_date,
            MAX(time) as latest_date
        FROM asset_prices
        UNION ALL
        SELECT
            'forecasts' as table_name,
            COUNT(*) as row_count,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(created_at) as earliest_date,
            MAX(created_at) as latest_date
        FROM forecasts
        UNION ALL
        SELECT
            'risk_metrics' as table_name,
            COUNT(*) as row_count,
            COUNT(DISTINCT portfolio_id) as unique_symbols,
            MIN(time) as earliest_date,
            MAX(time) as latest_date
        FROM risk_metrics
        """

        stats = await db_manager.execute_raw_query(stats_query)
        return {"database_statistics": stats}

    except Exception as e:
        logger.error(f"Database stats query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.warning(f"ValueError in request {request.url}: {exc}")
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception in request {request.url}: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )