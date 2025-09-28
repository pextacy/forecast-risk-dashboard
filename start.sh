#!/bin/bash

# Treasury Risk Dashboard - Complete Setup Script
# For DEGA Hackathon - AI for DAO Treasury Management

set -e

echo "🏦 Starting Treasury Risk Dashboard Setup..."
echo "==============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your API keys if needed."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data/timescaledb

# Start the services
echo "🚀 Starting services with Docker Compose..."
cd infra
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check if TimescaleDB is ready
echo "🗄️  Checking database connection..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec treasury_timescaledb pg_isready -U treasury_user -d treasury_dashboard > /dev/null 2>&1; then
        echo "✅ Database is ready!"
        break
    fi

    attempt=$((attempt + 1))
    echo "Waiting for database... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Database failed to start within expected time"
    exit 1
fi

# Check if backend is ready
echo "🔧 Checking backend API..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend API is ready!"
        break
    fi

    attempt=$((attempt + 1))
    echo "Waiting for backend API... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Backend API failed to start within expected time"
    exit 1
fi

# Check if frontend is ready
echo "🌐 Checking frontend..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is ready!"
        break
    fi

    attempt=$((attempt + 1))
    echo "Waiting for frontend... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Frontend failed to start within expected time"
    exit 1
fi

# Initialize with some sample data
echo "📊 Initializing sample data..."
curl -s -X POST http://localhost:8000/admin/ingest-data > /dev/null || echo "⚠️  Sample data initialization may have failed"

echo ""
echo "🎉 Treasury Risk Dashboard is ready!"
echo "==============================================="
echo "📊 Dashboard:    http://localhost:3000"
echo "🔧 Backend API:  http://localhost:8000"
echo "📚 API Docs:     http://localhost:8000/docs"
echo "🗄️  Database:    localhost:5432 (treasury_user/treasury_pass_2024)"
echo ""
echo "🚀 Quick Start Guide:"
echo "1. Visit http://localhost:3000 to access the dashboard"
echo "2. Navigate to 'Forecasts' tab to generate AI predictions"
echo "3. Use 'Portfolio' tab to manage your asset allocations"
echo "4. Check 'Risk Analysis' for portfolio risk metrics"
echo "5. Generate 'Hedge Suggestions' for optimization"
echo ""
echo "🔑 Features Available:"
echo "• Real-time price forecasting (ARIMA, Prophet, LSTM)"
echo "• AI-powered explanations (requires OpenAI API key)"
echo "• Portfolio risk analysis and optimization"
echo "• Hedge suggestion engine with simulation"
echo "• TimescaleDB for efficient time-series storage"
echo ""
echo "📝 To stop the services, run: docker-compose down"
echo "📋 To view logs, run: docker-compose logs -f"
echo ""
echo "💡 This is built for the DEGA Hackathon!"
echo "   Theme: AI for DAO Treasury Management on Midnight"
echo "==============================================="