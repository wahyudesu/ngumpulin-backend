#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Ensure MLflow artifacts directory exists
mkdir -p /mlflow/artifacts

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until nc -z postgres 5432; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 1
done

# Execute the command passed to docker
exec "$@"