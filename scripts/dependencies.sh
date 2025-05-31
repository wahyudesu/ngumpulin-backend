#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Generating lock files for all services...${NC}\n"

# Create necessary directories
echo -e "${GREEN}Creating service directories...${NC}"
mkdir -p app mlflow airflow

# Function to generate lock file
generate_lock() {
    local service=$1
    local service_dir=$2
    
    echo -e "${GREEN}Generating lock file for $service...${NC}"
    cd "$service_dir" || exit
    
    # Install uv if not installed
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        pip install uv
    fi
    
    # Generate lock file
    echo "Generating lock file..."
    uv lock
    
    # Return to root directory
    cd ..
    
    echo -e "${GREEN}$service lock file generated!${NC}\n"
}

# Generate lock files for each service
generate_lock "FastAPI" "app"
generate_lock "MLflow" "mlflow"
generate_lock "Airflow" "airflow"

echo -e "${BLUE}All lock files have been generated successfully!${NC}"
echo -e "You can now run: ${GREEN}docker compose build${NC}" 