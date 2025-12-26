#!/bin/bash

# Script to setup and run the Federated Summarization Platform
# Usage: ./scripts/run_training.sh

set -e

echo "=========================================="
echo "Federated Summarization Training Launcher"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}! Creating .env from .env.example${NC}"
    cp .env.example .env
fi

echo -e "${GREEN}✓ Environment file ready${NC}"

# Build images
echo ""
echo "Building Docker images..."
docker-compose build

echo -e "${GREEN}✓ Images built${NC}"

# Start infrastructure
echo ""
echo "Starting infrastructure (Zookeeper, Kafka)..."
docker-compose up -d zookeeper kafka

echo "Waiting for Kafka to be ready (15s)..."
sleep 15

echo -e "${GREEN}✓ Infrastructure ready${NC}"

# Run producer
echo ""
echo "Starting data ingestion..."
docker-compose up producer

echo -e "${GREEN}✓ Data ingested${NC}"

# Start Flower server
echo ""
echo "Starting Flower server..."
docker-compose up -d flower-server

echo "Waiting for server to initialize (10s)..."
sleep 10

echo -e "${GREEN}✓ Flower server ready${NC}"

# Start clients
echo ""
echo "Starting Flower clients..."
docker-compose up -d health-client finance-client legal-client

echo -e "${GREEN}✓ Clients started${NC}"

# Start dashboard
echo ""
echo "Starting dashboard..."
docker-compose up -d dashboard

echo -e "${GREEN}✓ Dashboard ready${NC}"

# Show status
echo ""
echo "=========================================="
echo "System Status"
echo "=========================================="
docker-compose ps

echo ""
echo "=========================================="
echo "Access Points"
echo "=========================================="
echo "Dashboard: http://localhost:8501"
echo "Flower Server: localhost:8080"
echo "Kafka: localhost:9092"

echo ""
echo "=========================================="
echo "Monitoring"
echo "=========================================="
echo "View logs: docker-compose logs -f"
echo "View specific service: docker-compose logs -f flower-server"
echo "Stop all: docker-compose stop"

echo ""
echo -e "${GREEN}✓ Training started successfully!${NC}"
