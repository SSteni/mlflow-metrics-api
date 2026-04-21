#!/bin/bash
# One-time setup on the MLflow EC2
# Run: bash setup.sh

set -e

echo "Setting up MLflow Metrics API..."

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Copy env file
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "Created .env — edit it and set METRICS_API_KEY before starting."
fi

echo ""
echo "Setup complete. Run:  bash start.sh"
