"""
MLflow Metrics API — lightweight FastAPI service exposing MLflow run data
as a clean time-series endpoint for Prism and other consumers.

Usage:
    python mlflow_metrics_api.py
    # or:
    uvicorn mlflow_metrics_api:app --host 0.0.0.0 --port 5002

Swagger UI: http://<host>:5002/docs

Authentication (same pattern as UST Pulse):
    Header:        x-api-key: <METRICS_API_KEY>
    Query param:   ?api-key=<METRICS_API_KEY>

Environment variables:
    MLFLOW_TRACKING_URI   — default: http://localhost:5001
    MLFLOW_EXPERIMENT     — default: sql-genai-agent
    METRICS_API_KEY       — default: changeme  (set this in .env!)
    METRICS_API_PORT      — default: 5002
"""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

TRACKING_URI    = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "sql-genai-agent")
API_KEY         = os.environ.get("METRICS_API_KEY", "changeme")

AVAILABLE_METRICS = [
    "total_latency_seconds",
    "total_cost_usd",
    "llm_cost_usd",
    "athena_cost_usd",
    "total_input_tokens",
    "total_output_tokens",
    "llm_call_count",
]

UNIT_MAP = {
    "total_latency_seconds": "seconds",
    "total_cost_usd":        "USD",
    "llm_cost_usd":          "USD",
    "athena_cost_usd":       "USD",
    "total_input_tokens":    "tokens",
    "total_output_tokens":   "tokens",
    "llm_call_count":        "count",
}

# ── Auth (same pattern as UST Pulse) ──────────────────────────────────────────

_header_scheme = APIKeyHeader(name="x-api-key", auto_error=False)
_query_scheme  = APIKeyQuery(name="api-key",    auto_error=False)


async def auth_api_key(
    key_header: str = Security(_header_scheme),
    key_query:  str = Security(_query_scheme),
) -> str:
    api_key = key_header or key_query
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via 'x-api-key' header or 'api-key' query parameter.",
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key


# ── Pydantic models ────────────────────────────────────────────────────────────

class DataPoint(BaseModel):
    timestamp: str          # ISO 8601 UTC
    value: float
    run_id: str
    question: Optional[str] = None

class MetricResponse(BaseModel):
    metric: str
    unit: str
    timeframe: dict
    total_points: int
    data: List[DataPoint]

class AvailableMetricsResponse(BaseModel):
    metrics: List[str]
    experiment: str
    tracking_uri: str

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UST Pulse — MLflow Metrics API",
    description=(
        "Query MLflow run metrics as a time-series.\n\n"
        "**Authentication**: pass your API key via the `x-api-key` header "
        "or the `api-key` query parameter.\n\n"
        "**Timeframe**: use `hours` for a rolling window (e.g. `hours=24`) "
        "or pass explicit `start` / `end` as ISO 8601 UTC strings "
        "(e.g. `2026-04-21T00:00:00Z`)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(TRACKING_URI)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check (no auth required)")
def health():
    """Liveness probe — returns 200 if the service is up."""
    return {"status": "ok", "tracking_uri": TRACKING_URI}


@app.get(
    "/metrics/available",
    response_model=AvailableMetricsResponse,
    summary="List available metrics",
)
async def list_metrics(_: str = Security(auth_api_key)):
    """Returns all metric names that can be passed to GET /metrics."""
    return AvailableMetricsResponse(
        metrics=AVAILABLE_METRICS,
        experiment=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
    )


@app.get(
    "/metrics",
    response_model=MetricResponse,
    summary="Get a metric time-series",
)
async def get_metric(
    metric: str,
    hours: Optional[float] = 24,
    start: Optional[str] = None,
    end: Optional[str] = None,
    _: str = Security(auth_api_key),
):
    """
    Returns `[{timestamp, value, run_id, question}]` for every user query
    logged in the given timeframe that has the requested metric recorded.

    Each data point corresponds to one query run.

    **metric** — metric name (see /metrics/available)\n
    **hours** — rolling window in hours, default 24 (ignored when start+end are set)\n
    **start** — ISO 8601 UTC start, e.g. `2026-04-21T00:00:00Z`\n
    **end** — ISO 8601 UTC end (defaults to now)
    """
    if metric not in AVAILABLE_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric '{metric}'. Call /metrics/available for valid names.",
        )

    now = datetime.now(timezone.utc)

    if start:
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid start: {start!r}")
    else:
        start_dt = now - timedelta(hours=hours)

    if end:
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid end: {end!r}")
    else:
        end_dt = now

    cutoff_ms = int(start_dt.timestamp() * 1000)
    end_ms    = int(end_dt.timestamp() * 1000)

    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=(
                f"tags.run_type = 'query' "
                f"and start_time >= {cutoff_ms} "
                f"and start_time <= {end_ms}"
            ),
            order_by=["start_time ASC"],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLflow unreachable: {e}")

    col = f"metrics.{metric}"
    data_points: List[DataPoint] = []

    if not runs.empty and col in runs.columns:
        for _, row in runs.iterrows():
            raw_value = row.get(col)
            if pd.isna(raw_value):
                continue
            try:
                ts = pd.Timestamp(row["start_time"]).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                ts = "unknown"
            data_points.append(DataPoint(
                timestamp=ts,
                value=round(float(raw_value), 6),
                run_id=(row.get("run_id") or "")[:8],
                question=((row.get("params.question") or "")[:80]) or None,
            ))

    return MetricResponse(
        metric=metric,
        unit=UNIT_MAP.get(metric, ""),
        timeframe={
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        },
        total_points=len(data_points),
        data=data_points,
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("METRICS_API_PORT", 5002))
    print(f"Starting MLflow Metrics API on port {port}")
    print(f"MLflow URI  : {TRACKING_URI}")
    print(f"Swagger UI  : http://0.0.0.0:{port}/docs")
    uvicorn.run("mlflow_metrics_api:app", host="0.0.0.0", port=port, reload=False)
