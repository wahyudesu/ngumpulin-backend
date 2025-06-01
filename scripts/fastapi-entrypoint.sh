#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Function to check if a service is ready
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    
    echo "Waiting for $service to be ready..."
    for i in {1..30}; do
        if [ "$service" = "Postgres" ]; then
            if pg_isready -h "$host" -p "$port" -U postgres; then
                echo "$service is ready!"
                return 0
            fi
        else
            if curl -s "http://$host:$port" > /dev/null; then
                echo "$service is ready!"
                return 0
            fi
        fi
        echo "Waiting for $service... attempt $i"
        sleep 2
    done
    echo "$service is not ready after 60 seconds"
    return 1
}

# Function to check DNS resolution with multiple DNS servers
check_dns() {
    local host="$1"
    local dns_servers=("8.8.8.8" "8.8.4.4" "1.1.1.1")
    
    echo "Checking DNS resolution for $host..."
    for dns in "${dns_servers[@]}"; do
        echo "Trying DNS server: $dns"
        if nslookup "$host" "$dns" > /dev/null 2>&1; then
            echo "DNS resolution successful for $host using $dns"
            return 0
        fi
    done
    
    echo "DNS resolution failed for $host with all DNS servers"
    return 1
}

# Check DNS resolution for required hosts
echo "Checking DNS resolution for required hosts..."
for host in "cdn-lfs-us-1.hf.co" "mlflow" "postgres"; do
    if ! check_dns "$host"; then
        echo "Warning: DNS resolution failed for $host, but continuing..."
    fi
done

# Wait for MLflow
wait_for_service "mlflow" "5000" "MLflow"

# Wait for Postgres
wait_for_service "postgres" "5432" "Postgres"

# Create necessary directories
mkdir -p /app/models
mkdir -p /app/logs

# Execute the command passed to docker
echo "Starting FastAPI application..."
exec "$@" 