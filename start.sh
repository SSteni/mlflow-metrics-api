#!/bin/bash
# Start the MLflow Metrics API
# Run once on EC2 after cloning and installing deps

set -e

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PORT=${METRICS_API_PORT:-5002}

echo "Starting MLflow Metrics API on port $PORT..."
source venv/bin/activate
uvicorn mlflow_metrics_api:app --host 0.0.0.0 --port "$PORT"
