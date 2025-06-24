#!/bin/bash

# AI Options Trading System - Development Startup Script
# This script sets up and starts the development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Main script
main() {
    echo "=============================================="
    echo "AI Options Trading System - Development Setup"
    echo "=============================================="
    echo
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
    echo
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from .env.example..."
        cp .env.example .env
        print_status "Please edit .env file with your configuration before continuing."
        read -p "Press Enter to continue after editing .env file..."
    fi
    
    # Create necessary directories
    print_status "Creating necessary directories..."
    mkdir -p logs/{gateway,cache,data-ingestion,analytics,nginx}
    mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
    mkdir -p nginx/{conf.d,ssl}
    mkdir -p notebooks
    mkdir -p services/{signals,portfolio}
    
    # Create placeholder files for services not yet implemented
    echo '<html><body><h1>AI-OTS Signals Service</h1><p>Coming Soon...</p></body></html>' > services/signals/placeholder.html
    echo '<html><body><h1>AI-OTS Portfolio Service</h1><p>Coming Soon...</p></body></html>' > services/portfolio/placeholder.html
    
    print_success "Directories created"
    echo
    
    # Stop any existing containers
    print_status "Stopping any existing containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Build and start core services
    print_status "Building and starting core services..."
    docker-compose up -d timescaledb redis
    
    # Wait for databases to be ready
    wait_for_service "TimescaleDB" "http://localhost:5432" || {
        print_error "TimescaleDB failed to start"
        docker-compose logs timescaledb
        exit 1
    }
    
    wait_for_service "Redis" "http://localhost:6379" || {
        print_error "Redis failed to start"
        docker-compose logs redis
        exit 1
    }
    
    # Start application services
    print_status "Starting application services..."
    docker-compose up -d cache-service data-ingestion analytics api-gateway
    
    # Wait for services to be ready
    wait_for_service "Cache Service" "http://localhost:8001/health"
    wait_for_service "Data Ingestion Service" "http://localhost:8002/health"
    wait_for_service "Analytics Service" "http://localhost:8003/health"
    wait_for_service "API Gateway" "http://localhost:8000/health"
    
    # Start placeholder services
    print_status "Starting placeholder services..."
    docker-compose up -d signals portfolio
    
    echo
    print_success "All services are running!"
    echo
    
    # Display service URLs
    echo "=============================================="
    echo "Service URLs:"
    echo "=============================================="
    echo "🌐 API Gateway:        http://localhost:8000"
    echo "💾 Cache Service:      http://localhost:8001"
    echo "📊 Data Ingestion:     http://localhost:8002"
    echo "📈 Analytics:          http://localhost:8003"
    echo "🎯 Signals:            http://localhost:8004"
    echo "💼 Portfolio:          http://localhost:8005"
    echo
    echo "🗄️  TimescaleDB:       localhost:5432"
    echo "🔴 Redis:              localhost:6379"
    echo
    
    # Check if user wants to start monitoring tools
    echo "=============================================="
    echo "Optional Tools:"
    echo "=============================================="
    read -p "Start monitoring tools (Prometheus, Grafana)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting monitoring tools..."
        docker-compose --profile monitoring up -d
        
        wait_for_service "Prometheus" "http://localhost:9090"
        wait_for_service "Grafana" "http://localhost:3000"
        
        echo "📊 Prometheus:         http://localhost:9090"
        echo "📈 Grafana:            http://localhost:3000 (admin/admin)"
        echo
    fi
    
    read -p "Start development tools (Redis Commander, pgAdmin, Jupyter)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting development tools..."
        docker-compose --profile tools --profile development up -d
        
        wait_for_service "Redis Commander" "http://localhost:8081"
        wait_for_service "pgAdmin" "http://localhost:8080"
        wait_for_service "Jupyter" "http://localhost:8888"
        
        echo "🔴 Redis Commander:    http://localhost:8081"
        echo "🐘 pgAdmin:            http://localhost:8080"
        echo "📓 Jupyter:            http://localhost:8888"
        echo
    fi
    
    # Health check summary
    echo "=============================================="
    echo "Health Check Summary:"
    echo "=============================================="
    
    services=(
        "API Gateway:http://localhost:8000/health"
        "Cache Service:http://localhost:8001/health"
        "Data Ingestion:http://localhost:8002/health"
        "Analytics:http://localhost:8003/health"
    )
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        url=$(echo $service | cut -d: -f2-)
        
        if curl -f -s "$url" >/dev/null 2>&1; then
            echo -e "✅ $name: ${GREEN}Healthy${NC}"
        else
            echo -e "❌ $name: ${RED}Unhealthy${NC}"
        fi
    done
    
    echo
    echo "=============================================="
    echo "Quick Start Commands:"
    echo "=============================================="
    echo "📋 View all services:     docker-compose ps"
    echo "📜 View logs:             docker-compose logs -f [service-name]"
    echo "🛑 Stop all services:     docker-compose down"
    echo "🔄 Restart service:       docker-compose restart [service-name]"
    echo "🧹 Clean up:              docker-compose down -v --remove-orphans"
    echo
    echo "🔍 Test API Gateway:      curl http://localhost:8000/health"
    echo "📊 Test Analytics:        curl http://localhost:8000/api/v1/analytics/indicators/AAPL"
    echo "💾 Test Cache:            curl http://localhost:8000/api/v1/cache/stock-prices/AAPL"
    echo
    
    print_success "Development environment is ready!"
    print_status "Check the logs if any service shows as unhealthy:"
    echo "   docker-compose logs [service-name]"
    echo
}

# Handle script interruption
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"

