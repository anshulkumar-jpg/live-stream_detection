"""FastAPI control plane entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import cameras, events, health, streams, viewer, workers
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.messaging.kafka_producer import EventBus
from app.services.container import Container

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, bus: EventBus | None = None, create_schema: bool = False) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level)
    container = Container(s, bus, create_schema)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await container.start()
        yield
        await container.stop()

    app = FastAPI(title="VMS Control Plane", version="1.0.0", lifespan=lifespan,
                  docs_url="/docs", openapi_url="/openapi.json")
    app.state.container = container
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"],
                       allow_credentials=False)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.url.path not in ("/health", "/metrics") and not container.limiter.allow(
                request.client.host if request.client else "unknown"):
            return JSONResponse({"error": {"code": 429, "message": "rate limit exceeded"}}, status_code=429)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_exc(_: Request, exc: HTTPException):
        msg = exc.detail if isinstance(exc.detail, str) else "error"
        body = {"error": {"code": exc.status_code, "message": msg}}
        if not isinstance(exc.detail, str):
            body["error"]["details"] = exc.detail
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def val_exc(_: Request, exc: RequestValidationError):
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]  # no input echo (could hold secrets)
        return JSONResponse({"error": {"code": 422, "message": "validation error", "details": details}}, status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse({"error": {"code": 500, "message": "internal error"}}, status_code=500)

    for r in (health.router, cameras.router, workers.router, events.router, streams.router, viewer.router):
        app.include_router(r)
    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.main:app_factory
    return create_app()
