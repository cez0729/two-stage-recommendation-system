"""FastAPI surface for recommendations, metrics, and feedback events."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from recsys.serving.cache import RedisCache
from recsys.serving.events import SQLiteFeedbackStore
from recsys.serving.pipeline import ServingPipeline

LOGGER = logging.getLogger("recsys.serving")


class FeedbackEvent(BaseModel):
    """Validated impression/click/purchase event payload."""

    event_id: str | None = None
    user_id: str
    item_id: str
    event_type: Literal["impression", "click", "purchase"]
    timestamp: datetime
    request_id: str
    model_version: str
    rank_position: int = Field(ge=1)
    simulated: bool = True


def _prometheus(metrics: dict[str, Any], model_version: str) -> str:
    values = metrics["latency_ms"]
    if values:
        ordered = sorted(values)
        p50 = ordered[int(0.50 * (len(ordered) - 1))] / 1000.0
        p95 = ordered[int(0.95 * (len(ordered) - 1))] / 1000.0
    else:
        p50 = p95 = 0.0
    escaped_version = model_version.replace("\\", "\\\\").replace('"', '\\"')
    lines = [
        "# HELP recsys_request_count Total recommendation requests.",
        "# TYPE recsys_request_count counter",
        f"recsys_request_count {metrics['request_count']}",
        "# HELP recsys_error_count Total recommendation errors.",
        "# TYPE recsys_error_count counter",
        f"recsys_error_count {metrics['error_count']}",
        "# TYPE recsys_cache_hit_count counter",
        f"recsys_cache_hit_count {metrics['cache_hit']}",
        "# TYPE recsys_cache_miss_count counter",
        f"recsys_cache_miss_count {metrics['cache_miss']}",
        "# TYPE recsys_fallback_count counter",
        f"recsys_fallback_count {metrics['fallback_count']}",
        "# TYPE recsys_recommendation_count counter",
        f"recsys_recommendation_count {metrics['recommendation_count']}",
        "# TYPE recsys_request_latency_seconds gauge",
        f'recsys_request_latency_seconds{{quantile="0.5"}} {p50:.6f}',
        f'recsys_request_latency_seconds{{quantile="0.95"}} {p95:.6f}',
        "# TYPE recsys_model_version_info gauge",
        f'recsys_model_version_info{{version="{escaped_version}"}} 1',
    ]
    return "\n".join(lines) + "\n"


def create_app(
    config_path: str | None = None,
    *,
    pipeline: ServingPipeline | None = None,
    cache: RedisCache | None = None,
    feedback_store: SQLiteFeedbackStore | None = None,
) -> FastAPI:
    """Create an app; expensive model loading happens exactly once per app."""
    config_path = config_path or os.getenv("RECSYS_CONFIG", "configs/serving.yaml")
    feedback_path = os.getenv("FEEDBACK_DB_PATH", "data/feedback/events.db")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.pipeline is None:
            application.state.pipeline = ServingPipeline.from_yaml(config_path)
        if application.state.cache is None:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                application.state.cache = RedisCache(redis_url, ttl_seconds=300)
        if application.state.feedback_store is None:
            application.state.feedback_store = SQLiteFeedbackStore(feedback_path)
        yield

    app = FastAPI(title="Two-stage Recommendation Demo", version="v1", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.cache = cache
    app.state.feedback_store = feedback_store
    app.state.metrics = {
        "request_count": 0,
        "error_count": 0,
        "cache_hit": 0,
        "cache_miss": 0,
        "fallback_count": 0,
        "recommendation_count": 0,
        "latency_ms": [],
    }

    @app.get("/health")
    def health() -> dict[str, Any]:
        active_pipeline = app.state.pipeline
        active_cache = app.state.cache
        return {
            "status": "ok",
            "version": getattr(active_pipeline, "model_version", "v1"),
            "model_loaded": active_pipeline is not None,
            "catalog_size": active_pipeline.catalog_size if active_pipeline else 0,
            "redis_available": active_cache.ping() if active_cache else False,
            "feedback_store": app.state.feedback_store is not None,
        }

    @app.get("/recommend/{user_id}")
    def recommend(
        user_id: str,
        k: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        started = perf_counter()
        app.state.metrics["request_count"] += 1
        active_pipeline = app.state.pipeline
        if active_pipeline is None:
            raise HTTPException(status_code=503, detail="model_not_loaded")
        model_version = str(getattr(active_pipeline, "model_version", "v1"))
        request_id = str(uuid4())
        key = f"rec:{model_version}:{user_id}:{k}"
        active_cache = app.state.cache
        if active_cache:
            cached = active_cache.get(key)
            if cached is not None:
                app.state.metrics["cache_hit"] += 1
                app.state.metrics["fallback_count"] += int(cached.get("fallback", False))
                app.state.metrics["recommendation_count"] += len(
                    cached.get("recommendations", [])
                )
                app.state.metrics["latency_ms"].append((perf_counter() - started) * 1000)
                LOGGER.info(json.dumps({"event": "recommend", "request_id": request_id,
                                        "user_id": user_id, "k": k, "cache_hit": True}))
                return cached
            app.state.metrics["cache_miss"] += 1
        try:
            recommendations = [
                item.__dict__ for item in active_pipeline.recommend(user_id, k)
            ]
            known_histories = getattr(active_pipeline, "histories", None)
            is_fallback = known_histories is not None and str(user_id) not in known_histories
            app.state.metrics["fallback_count"] += int(is_fallback)
            app.state.metrics["recommendation_count"] += len(recommendations)
            result: dict[str, Any] = {
                "user_id": user_id,
                "k": k,
                "recommendations": recommendations,
            }
            # Keep stub-based tests backward compatible; real serving always has metadata.
            if hasattr(active_pipeline, "model_version"):
                result.update({
                    "request_id": request_id,
                    "model_version": model_version,
                    "fallback": is_fallback,
                })
            if active_cache:
                active_cache.set(key, result)
            return result
        except Exception as exc:
            app.state.metrics["error_count"] += 1
            raise HTTPException(status_code=500, detail="recommendation_failed") from exc
        finally:
            elapsed = (perf_counter() - started) * 1000
            app.state.metrics["latency_ms"].append(elapsed)
            LOGGER.info(json.dumps({"event": "recommend", "request_id": request_id,
                                    "user_id": user_id, "k": k, "cache_hit": False,
                                    "latency_ms": elapsed}))

    @app.post("/events")
    def events(event: FeedbackEvent) -> dict[str, Any]:
        event_id = event.event_id or str(uuid4())
        payload = event.model_dump()
        payload["event_id"] = event_id
        payload["timestamp"] = event.timestamp.isoformat()
        try:
            inserted = app.state.feedback_store.write(payload)
        except Exception:
            LOGGER.exception("feedback_write_failed")
            return {"accepted": False, "event_id": event_id, "error": "feedback_write_failed"}
        return {"accepted": True, "inserted": inserted, "event_id": event_id}

    @app.get("/metrics")
    def metrics(format: str | None = Query(default=None)) -> Any:
        active_pipeline = app.state.pipeline
        model_version = str(getattr(active_pipeline, "model_version", "v1"))
        current = app.state.metrics
        values = current["latency_ms"]
        if values:
            values_sorted = sorted(values)
            p50 = values_sorted[int(0.50 * (len(values_sorted) - 1))]
            p95 = values_sorted[int(0.95 * (len(values_sorted) - 1))]
        else:
            p50 = p95 = 0.0
        if format == "prometheus":
            return PlainTextResponse(_prometheus(current, model_version))
        return {
            "request_count": current["request_count"],
            "error_count": current["error_count"],
            "cache_hit": current["cache_hit"],
            "cache_miss": current["cache_miss"],
            "fallback_count": current["fallback_count"],
            "recommendation_count": current["recommendation_count"],
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "model_version": model_version,
        }

    return app


app = create_app()
