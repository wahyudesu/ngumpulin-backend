#!/bin/bash

set -e
set -u

function create_user_and_database() {
    local database=$1
    local user=$2
    local password=$3
    echo "Creating user '$user' and database '$database'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE USER $user WITH PASSWORD '$password';
        CREATE DATABASE $database;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
EOSQL
}

# Create Airflow database and user
create_user_and_database "airflow" "airflow" "airflow"

# Create MLflow database and user
create_user_and_database "mlflow" "mlflow" "mlflow"

# Create FastAPI database and user
create_user_and_database "fastapi" "fastapi" "fastapi" 