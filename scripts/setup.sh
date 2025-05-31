#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting complete setup...${NC}\n"

# Make scripts executable
chmod +x scripts/*.sh

# Run setup
echo -e "${GREEN}Setting up dependencies...${NC}"
./scripts/dependencies.sh

# Run Docker
echo -e "${GREEN}Starting Docker services...${NC}"
./scripts/docker.sh

echo -e "\n${BLUE}Setup completed!${NC}"
echo -e "To backup, run: ${GREEN}./scripts/backup.sh${NC}"
echo -e "To cleanup, run: ${GREEN}./scripts/cleanup.sh${NC}"