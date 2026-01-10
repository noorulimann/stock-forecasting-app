#!/bin/bash

# ============================================================================
# Stock Forecasting App - Quick Start Script
# ============================================================================

set -e

echo "🚀 Stock Forecasting Application - Docker Setup"
echo "=============================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed."
    exit 1
fi

echo "✅ Docker installed"
echo "✅ Docker Compose installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env created (you may want to customize it)"
else
    echo "✅ .env file already exists"
fi
echo ""

# Create required directories
echo "📁 Creating required directories..."
mkdir -p saved_models mlruns logs nginx/ssl
echo "✅ Directories created"
echo ""

# Ask user what to do
echo "Choose an option:"
echo "1) Start all services (Production)"
echo "2) Start all services (Development with hot reload)"
echo "3) Build only (no start)"
echo "4) Stop all services"
echo "5) Stop and remove all (including volumes)"
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting all services in production mode..."
        docker-compose up -d
        ;;
    2)
        echo ""
        echo "🚀 Starting all services in development mode..."
        docker-compose -f docker-compose.dev.yml up -d
        ;;
    3)
        echo ""
        echo "🔨 Building images..."
        docker-compose build
        ;;
    4)
        echo ""
        echo "⏹️  Stopping services..."
        docker-compose stop
        docker-compose -f docker-compose.dev.yml stop 2>/dev/null || true
        ;;
    5)
        echo ""
        echo "⚠️  This will remove all containers and volumes!"
        read -p "Are you sure? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            docker-compose down -v
            docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
            echo "✅ All services and volumes removed"
        else
            echo "❌ Cancelled"
        fi
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo ""
echo "📊 Service Status:"
docker-compose ps || docker-compose -f docker-compose.dev.yml ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Access your services:"
echo "   - FastAPI App:      http://localhost:8000"
echo "   - API Docs:         http://localhost:8000/docs"
echo "   - MLflow UI:        http://localhost:5000"
echo "   - MongoDB:          localhost:27017"
echo ""
echo "📝 Useful commands:"
echo "   View logs:          docker-compose logs -f"
echo "   Stop services:      docker-compose stop"
echo "   Restart:            docker-compose restart"
echo ""
echo "📖 For more info, see docs/DEPLOYMENT.md"
