#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Wait for postgres
echo "Waiting for postgres..."
while ! nc -z postgres 5432; do
  sleep 0.1
done
echo "PostgreSQL started"

# Initialize the database
echo "Initializing Airflow database..."
airflow db migrate

# Create Airflow user if it doesn't exist
echo "Creating Airflow user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin \
    || true

# Execute the command passed to docker
exec "$@" 