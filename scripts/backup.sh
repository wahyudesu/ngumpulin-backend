#!/bin/bash

# Set backup directory
BACKUP_DIR="./scripts/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_PATH"

# Backup PostgreSQL databases
echo "Backing up PostgreSQL databases..."
docker exec postgres pg_dump -U airflow airflow > "$BACKUP_PATH/airflow_db.sql"
docker exec postgres pg_dump -U airflow mlflow > "$BACKUP_PATH/mlflow_db.sql"

# Backup MLflow artifacts
echo "Backing up MLflow artifacts..."
docker cp mlflow:/mlflow/artifacts "$BACKUP_PATH/mlflow_artifacts"

# Backup Airflow DAGs and logs
echo "Backing up Airflow files..."
cp -r ./airflow/dags "$BACKUP_PATH/airflow_dags"
cp -r ./airflow/logs "$BACKUP_PATH/airflow_logs"

# Backup FastAPI app
echo "Backing up FastAPI app..."
cp -r ./app "$BACKUP_PATH/app"

# Backup configuration files
echo "Backing up configuration files..."
cp docker-compose.yaml "$BACKUP_PATH/"
cp .env "$BACKUP_PATH/"

# Create backup archive
echo "Creating backup archive..."
tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "backup_$TIMESTAMP"

# Remove temporary backup directory
rm -rf "$BACKUP_PATH"

# Keep only last 5 backups
ls -t "$BACKUP_DIR"/backup_*.tar.gz | tail -n +6 | xargs -r rm

echo "Backup completed: $BACKUP_PATH.tar.gz" 