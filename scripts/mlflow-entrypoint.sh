#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Start MLflow server in the background
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://postgres:postgres@postgres:5432/mlflow

# Execute the command passed to docker
exec "$@" 