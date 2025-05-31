#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Cleaning up environment...${NC}\n"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is installed
if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Stop and remove containers
echo -e "${GREEN}Stopping and removing containers...${NC}"
docker compose down -v

# Remove all unused images
echo -e "${GREEN}Removing unused images...${NC}"
docker image prune -f

# Remove all unused volumes
echo -e "${GREEN}Removing unused volumes...${NC}"
docker volume prune -f

# Remove all unused networks
echo -e "${GREEN}Removing unused networks...${NC}"
docker network prune -f

# Clean up virtual environments and service directories
echo -e "${GREEN}Cleaning up service directories...${NC}"
for service in app mlflow airflow; do
    if [ -d "$service" ]; then
        echo "Cleaning up $service..."
        rm -rf "$service/.venv"
        rm -f "$service/uv.lock"
    fi
done

echo -e "\n${BLUE}Environment has been cleaned up!${NC}"
echo -e "To set up again, run: ${GREEN}./scripts/setup_dependencies.sh${NC}" 