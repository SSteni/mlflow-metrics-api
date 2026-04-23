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

Example calls:
    GET /metrics?metric=duration_seconds&hours=24
    GET /metrics?metric=total_cost_usd&from=2026-04-21T11:00:00Z&to=2026-04-21T12:00:00Z
"""

import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

TRACKING_URI    = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "sql-genai-agent")
API_KEY         = os.environ.get("METRICS_API_KEY", "changeme")

# Maps API metric names → MLflow metric column names
METRIC_MAP = {
    "duration_seconds":  "total_latency_seconds",
    "total_cost_usd":    "total_cost_usd",
    "llm_cost_usd":      "llm_cost_usd",
    "athena_cost_usd":   "athena_cost_usd",
    "total_input_tokens": "total_input_tokens",
    "total_output_tokens": "total_output_tokens",
    "llm_call_count":    "llm_call_count",
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
    timestamp:  str
    request_id: str
    value:      Optional[float] = None

class MetricsResponse(BaseModel):
    metric:       str
    fetched_at:   str
    source:       str
    timeframe:    Dict[str, Any]
    total_points: int
    datapoints:   List[DataPoint]

class AvailableMetricsResponse(BaseModel):
    metrics:       List[str]
    experiment:    str
    tracking_uri:  str

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UST Pulse — MLflow Metrics API",
    description=(
        "Exposes MLflow run metrics.\n\n"
        "**Authentication**: pass your API key via the `x-api-key` header "
        "or the `api-key` query parameter.\n\n"
        "**One metric at a time** - pass the `metric` name you want.\n\n"
        "**Prism temporal call**: `GET /metrics?metric=duration_seconds&from=<ISO>&to=<ISO>`\n\n"
        "**Rolling window**: `GET /metrics?metric=duration_seconds&hours=24`\n\n"
        "Each datapoint contains `timestamp`, `request_id`, and `value` of the requested metric."
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
    """Liveness probe - returns 200 if the service is up."""
    return {"status": "ok", "tracking_uri": TRACKING_URI}


@app.get(
    "/metrics/available",
    response_model=AvailableMetricsResponse,
    summary="List available metrics",
)
async def list_metrics(_: str = Security(auth_api_key)):
    """Returns all metric names that can be passed to GET /metrics."""
    return AvailableMetricsResponse(
        metrics=list(METRIC_MAP.keys()),
        experiment=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get a metric time-series (Prism-compatible)",
)
async def get_metrics(
    metric: str = Query(
        ...,
        description="Metric name - see /metrics/available for valid names",
        example="duration_seconds",
    ),
    from_: Optional[str] = Query(
        None,
        alias="from",
        description="Window start - ISO 8601 UTC, e.g. 2026-04-21T00:00:00Z",
    ),
    to: Optional[str] = Query(
        None,
        description="Window end - ISO 8601 UTC (defaults to now)",
    ),
    resolution: Optional[str] = Query(
        None,
        description="Optional resolution hint (e.g. 5m) - passed by Prism, not used internally",
    ),
    hours: Optional[float] = Query(
        24,
        description="Rolling window in hours - used when from/to are not provided",
    ),
    _: str = Security(auth_api_key),
):
    """
    Returns `[{timestamp, request_id, value}]` for the requested metric
    across all query runs in the given timeframe.

    Each datapoint = one query run.\n
    **metric** — metric name (see /metrics/available)\n
    **from** — ISO 8601 UTC start (optional)\n
    **to** — ISO 8601 UTC end (optional, defaults to now)\n
    **hours** — rolling window fallback, default 24
    """
    if metric not in METRIC_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric '{metric}'. Call /metrics/available for valid names.",
        )

    now = datetime.now(timezone.utc)

    if from_:
        try:
            start_dt = datetime.fromisoformat(from_.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid from: {from_!r}")
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

    mlflow_col = f"metrics.{METRIC_MAP[metric]}"
    datapoints: List[DataPoint] = []

    if not runs.empty and mlflow_col in runs.columns:
        for _, row in runs.iterrows():
            raw_value = row.get(mlflow_col)
            if raw_value is None or pd.isna(raw_value):
                continue
            try:
                ts = pd.Timestamp(row["start_time"]).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                ts = "unknown"
            datapoints.append(DataPoint(
                timestamp=ts,
                request_id=(row.get("run_id") or "")[:8],
                value=round(float(raw_value), 6),
            ))

    return MetricsResponse(
        metric=metric,
        fetched_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        source="UST Pulse - SQL GenAI Agent",
        timeframe={
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        },
        total_points=len(datapoints),
        datapoints=datapoints,
    )


# ── Synthetic data + GovAI metrics (/metricsv2) ───────────────────────────────

random.seed(42)  # reproducible

def _generate_synthetic_dataset(n: int = 2000):
    """Generate n synthetic query records with realistic distributions."""
    now = datetime.now(timezone.utc)
    records = []
    for i in range(n):
        # Spread records over last 30 days
        ts = now - timedelta(seconds=random.randint(0, 30 * 24 * 3600))
        # Realistic ranges based on real Pulse data
        duration    = max(1.0, random.gauss(30, 15))          # seconds
        llm_cost    = max(0.0001, random.gauss(0.12, 0.06))   # USD
        athena_cost = max(0.0, random.gauss(1.2, 0.8))        # USD
        in_tokens   = int(max(500, random.gauss(15000, 8000)))
        out_tokens  = int(max(100, random.gauss(2000, 1000)))
        llm_calls   = max(1, int(random.gauss(4, 2)))
        records.append({
            "timestamp":          ts,
            "request_id":         f"syn{i:05d}",
            "duration_seconds":   round(duration, 3),
            "total_cost_usd":     round(llm_cost + athena_cost, 6),
            "llm_cost_usd":       round(llm_cost, 6),
            "athena_cost_usd":    round(athena_cost, 6),
            "total_input_tokens":  in_tokens,
            "total_output_tokens": out_tokens,
            "llm_call_count":     llm_calls,
        })
    records.sort(key=lambda r: r["timestamp"])
    return records

# Generate once at startup
_SYNTHETIC = _generate_synthetic_dataset(2000)

GOVAI_METRICS = {
    "sql_sanity_score": {
        "description": "SQL execution cost as a fraction of total cost. Low value = well-optimised query spending most budget on data retrieval (good). High value = excess LLM retries relative to data work (investigate).",
        "derivation":  "llm_cost_usd / total_cost_usd",
        "unit":        "ratio (0–1)",
    },
    "answer_groundedness_score": {
        "description": "Proportion of cost driven by actual data retrieval (Athena). High value = response grounded in real queried data. Low value = mostly LLM reasoning with little data backing.",
        "derivation":  "athena_cost_usd / total_cost_usd",
        "unit":        "ratio (0–1)",
    },
    "audit_trail_completeness": {
        "description": "1 if the query involved multi-step reasoning (>3 LLM calls), producing a richer, auditable trace. 0 if a simple single-pass query. Average across window = proportion of fully-traceable queries.",
        "derivation":  "1 if llm_call_count > 3 else 0",
        "unit":        "binary (0 or 1)",
    },
    "response_accuracy_score": {
        "description": "How much of the 90-second timeout budget was consumed. Low value = fast, confident response. Value approaching 1 = near-timeout risk. Above 1 = timeout breach. Future: replace with user feedback rating (0–5).",
        "derivation":  "duration_seconds / 90",
        "unit":        "ratio (0–1+)",
    },
}

RAW_METRICS_V2 = list(METRIC_MAP.keys())
ALL_METRICS_V2 = RAW_METRICS_V2 + list(GOVAI_METRICS.keys())


def _derive_govai(record: dict, metric: str) -> float:
    if metric == "sql_sanity_score":
        total = record["total_cost_usd"]
        return round(record["llm_cost_usd"] / total, 4) if total > 0 else 0.0
    if metric == "answer_groundedness_score":
        total = record["total_cost_usd"]
        return round(record["athena_cost_usd"] / total, 4) if total > 0 else 0.0
    if metric == "audit_trail_completeness":
        return 1.0 if record["llm_call_count"] > 3 else 0.0
    if metric == "response_accuracy_score":
        return round(record["duration_seconds"] / 90, 4)
    return 0.0


class MetricsV2Response(BaseModel):
    metric:      str
    is_derived:  bool
    description: Optional[str] = None
    derivation:  Optional[str] = None
    unit:        str
    fetched_at:  str
    source:      str
    timeframe:   Dict[str, Any]
    total_points: int
    datapoints:  List[DataPoint]


@app.get(
    "/metricsv2",
    response_model=MetricsV2Response,
    summary="Get a Governance Metric (GovAI-ready)",
)
async def get_metrics_v2(
    metric: str = Query(
        ...,
        description="Raw KPI or derived GovAI metric — see /metricsv2/available",
        example="sql_sanity_score",
    ),
    from_: Optional[str] = Query(None, alias="from", description="ISO 8601 UTC start"),
    to:    Optional[str] = Query(None, description="ISO 8601 UTC end (defaults to now)"),
    hours: Optional[float] = Query(24, description="Rolling window in hours (fallback)"),
    _: str = Security(auth_api_key),
):
    """
    Returns data for any KPI or Governance AI (GovAI) metric.

    **Raw KPIs** — same names as `/metrics` (served from synthetic data instead of MLflow):\n
    `duration_seconds`, `total_cost_usd`, `llm_cost_usd`, `athena_cost_usd`,
    `total_input_tokens`, `total_output_tokens`, `llm_call_count`

    ---

    **Derived GovAI metrics**:

    - **`sql_sanity_score`** *(ratio 0–1)*\n
      SQL execution cost as a fraction of total cost. Low value = well-optimised query spending
      most of its budget on data retrieval (good). High value = excess LLM retries relative to
      data work (investigate).

    - **`answer_groundedness_score`** *(ratio 0–1)*\n
      Proportion of cost driven by actual data retrieval (Athena). High value = response is
      grounded in real queried data. Low value = mostly LLM reasoning with little data backing.
      

    - **`audit_trail_completeness`** *(binary 0 or 1)*\n
      1 if the query involved multi-step reasoning (>3 LLM calls), producing a richer, auditable
      trace. 0 if a simple single-pass query. Average across a window = proportion of
      fully-traceable queries. 

    - **`response_accuracy_score`** *(ratio 0–1+)*\n
      How much of the 90-second timeout budget was consumed. Low value = fast, confident response.
      Value approaching 1 = near-timeout risk. Above 1 = timeout breach. Future: replace with
      user feedback rating (0–5).
    """
    if metric not in ALL_METRICS_V2:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric '{metric}'. Call /metricsv2/available for valid names.",
        )

    now = datetime.now(timezone.utc)

    if from_:
        try:
            start_dt = datetime.fromisoformat(from_.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid from: {from_!r}")
    else:
        start_dt = now - timedelta(hours=hours or 24)

    end_dt = now
    if to:
        try:
            end_dt = datetime.fromisoformat(to.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid to: {to!r}")

    is_derived = metric in GOVAI_METRICS
    govai_info = GOVAI_METRICS.get(metric, {})
    unit       = govai_info.get("unit", "")

    datapoints: List[DataPoint] = []
    for rec in _SYNTHETIC:
        ts = rec["timestamp"]
        if ts < start_dt or ts > end_dt:
            continue
        if is_derived:
            value = _derive_govai(rec, metric)
        else:
            value = rec.get(metric)
            if value is None:
                continue
            value = round(float(value), 6)
        datapoints.append(DataPoint(
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            request_id=rec["request_id"],
            value=value,
        ))

    return MetricsV2Response(
        metric=metric,
        is_derived=is_derived,
        description=govai_info.get("description"),
        derivation=govai_info.get("derivation"),
        unit=unit,
        fetched_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        source="UST Pulse — Synthetic Demo Data",
        timeframe={
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": round((end_dt - start_dt).total_seconds() / 3600, 2),
        },
        total_points=len(datapoints),
        datapoints=datapoints,
    )


@app.get(
    "/metricsv2/available",
    summary="List available metrics for /metricsv2",
)
async def list_metrics_v2(_: str = Security(auth_api_key)):
    """Returns raw KPIs and derived GovAI metrics available in /metricsv2."""
    return {
        "raw_metrics": RAW_METRICS_V2,
        "govai_derived_metrics": {
            name: {
                "description": info["description"],
                "derivation":  info["derivation"],
                "unit":        info["unit"],
            }
            for name, info in GOVAI_METRICS.items()
        },
        "source": "synthetic demo dataset (2000 records, 30-day window)",
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("METRICS_API_PORT", 5002))
    print(f"Starting MLflow Metrics API on port {port}")
    print(f"MLflow URI  : {TRACKING_URI}")
    print(f"Swagger UI  : http://0.0.0.0:{port}/docs")
    uvicorn.run("mlflow_metrics_api:app", host="0.0.0.0", port=port, reload=False)
