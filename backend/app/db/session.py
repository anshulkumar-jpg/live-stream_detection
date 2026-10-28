from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings


def make_engine(s: Settings) -> AsyncEngine:
    if s.database_url.startswith("sqlite"):
        kw: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in s.database_url:
            kw["poolclass"] = StaticPool
        return create_async_engine(s.database_url, **kw)
    return create_async_engine(s.database_url, pool_size=s.db_pool_size, max_overflow=s.db_pool_size,
                               pool_pre_ping=True, pool_recycle=1800)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
