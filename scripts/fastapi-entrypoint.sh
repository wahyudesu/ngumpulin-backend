#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Execute the command passed to docker
exec "$@" 