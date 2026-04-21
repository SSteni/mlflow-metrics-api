"""
MLflow Metrics API — lightweight FastAPI service exposing MLflow run data
as a Prism-compatible time-series endpoint.

Usage:
    python mlflow_metrics_api.py
    # or:
    uvicorn mlflow_metrics_api:app --host 0.0.0.0 --port 5002

Swagger UI: http://<host>:5002/docs

Authentication (Prism api_key scheme):
    Header:  X-API-Key: <METRICS_API_KEY>

Prism temporal call:
    GET /metrics?from=2026-04-21T11:00:00Z&to=2026-04-21T12:00:00Z

Manual / rolling window call:
    GET /metrics?hours=24

Response format (Prism temporal):
    {
      "datapoints": [
        {
          "timestamp":               "2026-04-21T11:05:00Z",
          "request_id":              "23940d4b",
          "question":                "How many customers churned...",
          "total_latency_seconds":   4.89,
          "total_cost_usd":          0.000312,
          "llm_cost_usd":            0.000300,
          "athena_cost_usd":         0.000012,
          "total_input_tokens":      1500,
          "total_output_tokens":     320,
          "llm_call_count":          2
        },
        ...
      ]
    }
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

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

METRIC_COLUMNS = [
    "total_latency_seconds",
    "total_cost_usd",
    "llm_cost_usd",
    "athena_cost_usd",
    "total_input_tokens",
    "total_output_tokens",
    "llm_call_count",
]

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
    timestamp:               str
    request_id:              str
    question:                Optional[str] = None
    total_latency_seconds:   Optional[float] = None
    total_cost_usd:          Optional[float] = None
    llm_cost_usd:            Optional[float] = None
    athena_cost_usd:         Optional[float] = None
    total_input_tokens:      Optional[float] = None
    total_output_tokens:     Optional[float] = None
    llm_call_count:          Optional[float] = None

class MetricsResponse(BaseModel):
    fetched_at: str
    source: str
    timeframe: Dict[str, Any]
    total_points: int
    datapoints: List[DataPoint]

class AvailableMetricsResponse(BaseModel):
    metrics: List[str]
    experiment: str
    tracking_uri: str

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UST Pulse — MLflow Metrics API",
    description=(
        "Exposes MLflow run metrics as Prism-compatible time-series datapoints.\n\n"
        "**Authentication**: pass your API key via the `x-api-key` header "
        "or the `api-key` query parameter.\n\n"
        "**Prism temporal call**: `GET /metrics?from=<ISO>&to=<ISO>`\n\n"
        "**Rolling window**: `GET /metrics?hours=24`\n\n"
        "Each datapoint contains all available metrics for that query run."
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
    """Returns all metric names included in each datapoint."""
    return AvailableMetricsResponse(
        metrics=METRIC_COLUMNS,
        experiment=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get metrics time-series (Prism-compatible)",
)
async def get_metrics(
    # Prism sends these
    from_time: Optional[str] = None,
    to: Optional[str] = None,
    resolution: Optional[str] = None,
    # Fallback for manual / rolling window calls
    hours: Optional[float] = 24,
    _: str = Security(auth_api_key),
):
    """
    Returns all metrics for every query run in the given timeframe.

    **Prism temporal call** — pass `from` and `to` as ISO 8601 UTC:\n
    `GET /metrics?from=2026-04-21T11:00:00Z&to=2026-04-21T12:00:00Z`

    **Manual rolling window** — pass `hours`:\n
    `GET /metrics?hours=24`

    Each object in `datapoints` contains `timestamp`, `request_id`, `question`,
    and all numeric metrics for that run. Prism computes aggregates (`avg`, `max`,
    `min`, `count`) internally — no summary needed in the response.
    """
    now = datetime.now(timezone.utc)

    if from_time:
        try:
            start_dt = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid from: {from_time!r}")
    else:
        start_dt = now - timedelta(hours=hours or 24)

    if to:
        try:
            end_dt = datetime.fromisoformat(to.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid to: {to!r}")
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

    datapoints: List[DataPoint] = []

    if not runs.empty:
        for _, row in runs.iterrows():
            try:
                ts = pd.Timestamp(row["start_time"]).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                ts = "unknown"

            def _val(col):
                v = row.get(f"metrics.{col}")
                return round(float(v), 6) if v is not None and not pd.isna(v) else None

            datapoints.append(DataPoint(
                timestamp=ts,
                request_id=(row.get("run_id") or "")[:8],
                question=((row.get("params.question") or "")[:120]) or None,
                total_latency_seconds=_val("total_latency_seconds"),
                total_cost_usd=_val("total_cost_usd"),
                llm_cost_usd=_val("llm_cost_usd"),
                athena_cost_usd=_val("athena_cost_usd"),
                total_input_tokens=_val("total_input_tokens"),
                total_output_tokens=_val("total_output_tokens"),
                llm_call_count=_val("llm_call_count"),
            ))

    return MetricsResponse(
        fetched_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        source="UST Pulse — SQL GenAI Agent",
        timeframe={
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        },
        total_points=len(datapoints),
        datapoints=datapoints,
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("METRICS_API_PORT", 5002))
    print(f"Starting MLflow Metrics API on port {port}")
    print(f"MLflow URI  : {TRACKING_URI}")
    print(f"Swagger UI  : http://0.0.0.0:{port}/docs")
    uvicorn.run("mlflow_metrics_api:app", host="0.0.0.0", port=port, reload=False)
