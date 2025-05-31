#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Docker services...${NC}\n"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is installed
if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command_exists docker compose; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Stop and remove existing containers
echo -e "${GREEN}Cleaning up existing containers...${NC}"
docker compose down -v --remove-orphans

# Build images
echo -e "${GREEN}Building Docker images...${NC}"
docker compose build

# Start services
echo -e "${GREEN}Starting services...${NC}"
docker compose up -d

# Check if services are running
echo -e "${GREEN}Checking service status...${NC}"
docker compose ps

echo -e "\n${BLUE}Services are now running!${NC}"
echo -e "FastAPI: ${GREEN}http://localhost:8000${NC}"
echo -e "MLflow: ${GREEN}http://localhost:5000${NC}"
echo -e "Airflow: ${GREEN}http://localhost:8080${NC}"
echo -e "Adminer: ${GREEN}http://localhost:8088${NC}"
echo -e "Prometheus: ${GREEN}http://localhost:9090${NC}"
echo -e "Grafana: ${GREEN}http://localhost:3030${NC}" 