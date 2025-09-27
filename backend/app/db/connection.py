"""Database connection and session management for TimescaleDB."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://treasury_user:treasury_pass_2024@localhost:5432/treasury_dashboard")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()
metadata = MetaData()


class DatabaseManager:
    """Manages database connections and provides utility methods."""

    def __init__(self):
        self.engine = engine
        self.async_engine = async_engine

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_session(self):
        """Get a synchronous database session."""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def execute_raw_query(self, query: str, params: dict = None):
        """Execute raw SQL query with asyncpg for TimescaleDB optimizations."""
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            if params:
                result = await conn.fetch(query, *params.values())
            else:
                result = await conn.fetch(query)
            return [dict(row) for row in result]
        finally:
            await conn.close()

    async def bulk_insert_prices(self, price_data: list[dict]):
        """Optimized bulk insert for price data using COPY."""
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            # Prepare data for COPY
            records = [
                (row['time'], row['symbol'], row['price'], row.get('volume'),
                 row.get('market_cap'), row.get('source', 'api'))
                for row in price_data
            ]

            await conn.copy_records_to_table(
                'asset_prices',
                records=records,
                columns=['time', 'symbol', 'price', 'volume', 'market_cap', 'source']
            )
        finally:
            await conn.close()

    async def get_latest_prices(self, symbols: list[str] = None):
        """Get latest prices for specified symbols or all symbols."""
        query = """
        SELECT DISTINCT ON (symbol) symbol, time, price, volume, market_cap
        FROM asset_prices
        WHERE ($1::text[] IS NULL OR symbol = ANY($1))
        ORDER BY symbol, time DESC
        """
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            result = await conn.fetch(query, symbols)
            return [dict(row) for row in result]
        finally:
            await conn.close()

    async def get_price_history(self, symbol: str, days: int = 30):
        """Get price history for a symbol over specified days."""
        query = """
        SELECT time, price, volume
        FROM asset_prices
        WHERE symbol = $1
        AND time >= NOW() - INTERVAL '%s days'
        ORDER BY time ASC
        """ % days

        conn = await asyncpg.connect(DATABASE_URL)
        try:
            result = await conn.fetch(query, symbol)
            return [dict(row) for row in result]
        finally:
            await conn.close()


# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
async def get_async_db():
    """FastAPI dependency for async database sessions."""
    async with db_manager.get_async_session() as session:
        yield session

def get_sync_db():
    """FastAPI dependency for sync database sessions."""
    yield from db_manager.get_sync_session()