#!/bin/bash

# AI Options Trading System - Enhanced Development Setup Script
# BDT-P1: Enhanced Local Development Environment with AI-specific optimizations
# This script sets up and starts the development environment with AI model development capabilities

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/setup"
AI_MODEL_DIR="$SCRIPT_DIR/ai-models"
DEV_TOOLS_DIR="$SCRIPT_DIR/dev-tools"

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

print_ai() {
    echo -e "${PURPLE}[AI-MODEL]${NC} $1"
}

print_dev() {
    echo -e "${CYAN}[DEV-TOOLS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready with enhanced monitoring
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=${3:-30}
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

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources for AI development..."
    
    # Check available memory (minimum 8GB recommended for AI models)
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$mem_gb" -lt 8 ]; then
        print_warning "Available memory: ${mem_gb}GB. Recommended: 8GB+ for AI model development"
    else
        print_success "Available memory: ${mem_gb}GB - Sufficient for AI development"
    fi
    
    # Check available disk space (minimum 20GB recommended)
    local disk_gb=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$disk_gb" -lt 20 ]; then
        print_warning "Available disk space: ${disk_gb}GB. Recommended: 20GB+ for AI models and data"
    else
        print_success "Available disk space: ${disk_gb}GB - Sufficient for AI development"
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    print_status "CPU cores available: $cpu_cores"
    if [ "$cpu_cores" -lt 4 ]; then
        print_warning "CPU cores: $cpu_cores. Recommended: 4+ for optimal performance"
    fi
}

# Function to setup AI model development environment
setup_ml_environment() {
    print_ai "Setting up AI model development environment..."
    
    # Create AI model directories
    mkdir -p "$AI_MODEL_DIR"/{models,data,training,validation,experiments}
    mkdir -p "$AI_MODEL_DIR/models"/{signal-generation,pattern-recognition,confidence-calibration}
    mkdir -p "$AI_MODEL_DIR/data"/{historical,real-time,backtesting,validation}
    mkdir -p "$AI_MODEL_DIR/training"/{logs,checkpoints,configs}
    mkdir -p "$AI_MODEL_DIR/validation"/{results,reports,metrics}
    mkdir -p "$AI_MODEL_DIR/experiments"/{notebooks,scripts,results}
    
    # Create AI model configuration files
    cat > "$AI_MODEL_DIR/config.yaml" << 'EOF'
# AI Model Development Configuration
ai_models:
  signal_generation:
    model_type: "ensemble"
    strategies: 10
    confidence_threshold: 0.6
    retraining_interval: "weekly"
    
  pattern_recognition:
    model_type: "cnn_lstm"
    accuracy_threshold: 0.8
    validation_split: 0.2
    
  confidence_calibration:
    model_type: "isotonic_regression"
    target_accuracy: 0.8
    calibration_method: "platt_scaling"

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping: true
  validation_patience: 10

data:
  historical_days: 365
  validation_split: 0.2
  test_split: 0.1
  features:
    - "price"
    - "volume"
    - "volatility"
    - "technical_indicators"
    - "options_flow"
EOF
    
    # Create AI model requirements file
    cat > "$AI_MODEL_DIR/requirements.txt" << 'EOF'
# AI/ML Dependencies for Signal Generation
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
catboost>=1.2.0
tensorflow>=2.13.0
torch>=2.0.0
optuna>=3.3.0
mlflow>=2.5.0
wandb>=0.15.0

# Technical Analysis
ta-lib>=0.4.0
yfinance>=0.2.0
pandas-ta>=0.3.0

# Signal Processing
scipy>=1.11.0
statsmodels>=0.14.0

# Visualization
plotly>=5.15.0
bokeh>=3.2.0

# Development Tools
jupyter>=1.0.0
ipykernel>=6.25.0
black>=23.7.0
flake8>=6.0.0
pytest>=7.4.0
EOF
    
    # Create signal generation debugging tools
    cat > "$AI_MODEL_DIR/debug_signals.py" << 'EOF'
#!/usr/bin/env python3
"""
AI-OTS Signal Generation Debugging Tool
Provides debugging capabilities for signal generation algorithms
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class SignalDebugger:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.signals_history = []
        
    def analyze_signal_quality(self, signals_data):
        """Analyze signal quality metrics"""
        df = pd.DataFrame(signals_data)
        
        print("=== Signal Quality Analysis ===")
        print(f"Total signals: {len(df)}")
        print(f"Average confidence: {df['confidence'].mean():.3f}")
        print(f"Confidence std: {df['confidence'].std():.3f}")
        print(f"High confidence signals (>0.8): {len(df[df['confidence'] > 0.8])}")
        
        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(df['confidence'], bins=20, alpha=0.7)
        plt.title('Signal Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.scatter(df['expected_return'], df['confidence'], alpha=0.6)
        plt.title('Expected Return vs Confidence')
        plt.xlabel('Expected Return')
        plt.ylabel('Confidence')
        
        plt.tight_layout()
        plt.savefig('signal_analysis.png')
        plt.show()
        
    def validate_pattern_recognition(self, patterns_data):
        """Validate pattern recognition accuracy"""
        df = pd.DataFrame(patterns_data)
        
        print("=== Pattern Recognition Validation ===")
        print(f"Total patterns detected: {len(df)}")
        
        pattern_counts = df['pattern_type'].value_counts()
        print("Pattern distribution:")
        for pattern, count in pattern_counts.items():
            print(f"  {pattern}: {count}")
            
        # Plot pattern accuracy
        if 'accuracy' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='pattern_type', y='accuracy')
            plt.title('Pattern Recognition Accuracy by Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('pattern_accuracy.png')
            plt.show()
    
    def test_confidence_calibration(self, predictions, actual_outcomes):
        """Test confidence calibration accuracy"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_outcomes, predictions, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="AI-OTS")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Confidence Calibration Curve")
        plt.legend()
        plt.savefig('calibration_curve.png')
        plt.show()
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        print(f"Calibration Error: {calibration_error:.4f}")
        
        return calibration_error

if __name__ == "__main__":
    debugger = SignalDebugger()
    print("AI-OTS Signal Debugging Tool initialized")
    print("Use debugger.analyze_signal_quality(data) to analyze signals")
    print("Use debugger.validate_pattern_recognition(data) to validate patterns")
    print("Use debugger.test_confidence_calibration(pred, actual) to test calibration")
EOF
    
    # Create confidence calibration environment
    cat > "$AI_MODEL_DIR/calibrate_confidence.py" << 'EOF'
#!/usr/bin/env python3
"""
AI-OTS Confidence Calibration Tool
Calibrates signal confidence scores for accurate probability estimation
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

class ConfidenceCalibrator:
    def __init__(self):
        self.calibrator = None
        self.method = None
        
    def fit_calibration(self, confidence_scores, actual_outcomes, method='isotonic'):
        """Fit confidence calibration model"""
        self.method = method
        
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            self.calibrator = CalibratedClassifierCV(
                LogisticRegression(), method='sigmoid', cv=3
            )
        
        # Reshape for sklearn
        X = confidence_scores.reshape(-1, 1) if confidence_scores.ndim == 1 else confidence_scores
        
        self.calibrator.fit(X, actual_outcomes)
        
        print(f"Confidence calibration fitted using {method} method")
        
    def calibrate_confidence(self, confidence_scores):
        """Apply calibration to confidence scores"""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit_calibration first.")
            
        X = confidence_scores.reshape(-1, 1) if confidence_scores.ndim == 1 else confidence_scores
        
        if self.method == 'isotonic':
            return self.calibrator.predict(X.ravel())
        else:
            return self.calibrator.predict_proba(X)[:, 1]
    
    def evaluate_calibration(self, confidence_scores, actual_outcomes):
        """Evaluate calibration performance"""
        calibrated_scores = self.calibrate_confidence(confidence_scores)
        
        # Calculate reliability diagram
        from sklearn.calibration import calibration_curve
        
        fraction_pos, mean_pred = calibration_curve(
            actual_outcomes, calibrated_scores, n_bins=10
        )
        
        # Plot reliability diagram
        plt.figure(figsize=(8, 6))
        plt.plot(mean_pred, fraction_pos, "s-", label="Calibrated")
        plt.plot([0, 1], [0, 1], "k:", label="Perfect")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True)
        plt.savefig('reliability_diagram.png')
        plt.show()
        
        # Calculate Brier score
        brier_score = np.mean((calibrated_scores - actual_outcomes) ** 2)
        print(f"Brier Score: {brier_score:.4f}")
        
        return brier_score
    
    def save_calibrator(self, filepath):
        """Save calibration model"""
        joblib.dump({
            'calibrator': self.calibrator,
            'method': self.method
        }, filepath)
        print(f"Calibrator saved to {filepath}")
    
    def load_calibrator(self, filepath):
        """Load calibration model"""
        data = joblib.load(filepath)
        self.calibrator = data['calibrator']
        self.method = data['method']
        print(f"Calibrator loaded from {filepath}")

if __name__ == "__main__":
    calibrator = ConfidenceCalibrator()
    print("AI-OTS Confidence Calibration Tool initialized")
    print("Use calibrator.fit_calibration(scores, outcomes) to train")
    print("Use calibrator.calibrate_confidence(scores) to calibrate")
EOF
    
    chmod +x "$AI_MODEL_DIR/debug_signals.py"
    chmod +x "$AI_MODEL_DIR/calibrate_confidence.py"
    
    print_ai "AI model development environment created successfully"
}

# Function to setup developer experience tools
setup_developer_tools() {
    print_dev "Setting up enhanced developer experience tools..."
    
    # Create dev tools directory
    mkdir -p "$DEV_TOOLS_DIR"/{scripts,configs,templates}
    
    # Create hot-reload configuration
    cat > "$DEV_TOOLS_DIR/hot-reload.sh" << 'EOF'
#!/bin/bash
# Hot-reload script for AI-OTS services

SERVICE_NAME=${1:-"all"}
WATCH_DIRS="./services"

print_status() {
    echo -e "\033[0;34m[HOT-RELOAD]\033[0m $1"
}

if command -v inotifywait >/dev/null 2>&1; then
    print_status "Starting hot-reload for $SERVICE_NAME..."
    
    while inotifywait -r -e modify,create,delete $WATCH_DIRS; do
        print_status "Changes detected, reloading $SERVICE_NAME..."
        
        if [ "$SERVICE_NAME" = "all" ]; then
            docker-compose restart api-gateway cache-service data-ingestion analytics
        else
            docker-compose restart "$SERVICE_NAME"
        fi
        
        sleep 2
    done
else
    echo "inotifywait not found. Install inotify-tools for hot-reload functionality."
    echo "Ubuntu/Debian: sudo apt-get install inotify-tools"
    echo "macOS: brew install fswatch"
fi
EOF
    
    # Create performance profiling script
    cat > "$DEV_TOOLS_DIR/profile-performance.sh" << 'EOF'
#!/bin/bash
# Performance profiling script for AI-OTS services

SERVICE_NAME=${1:-"signals"}
DURATION=${2:-60}

print_status() {
    echo -e "\033[0;34m[PROFILER]\033[0m $1"
}

print_status "Starting performance profiling for $SERVICE_NAME (${DURATION}s)..."

# Create profiling directory
mkdir -p ./logs/profiling

# Profile CPU usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" \
    --no-stream > "./logs/profiling/${SERVICE_NAME}_stats_$(date +%Y%m%d_%H%M%S).log" &

STATS_PID=$!

# Profile signal generation performance if signals service
if [ "$SERVICE_NAME" = "signals" ]; then
    print_status "Profiling signal generation performance..."
    
    for i in {1..10}; do
        start_time=$(date +%s%N)
        curl -s "http://localhost:8000/api/v1/signals/generate/AAPL" > /dev/null
        end_time=$(date +%s%N)
        
        duration=$((($end_time - $start_time) / 1000000))
        echo "Signal generation $i: ${duration}ms" >> "./logs/profiling/signal_latency_$(date +%Y%m%d_%H%M%S).log"
        
        sleep 1
    done
fi

sleep $DURATION
kill $STATS_PID

print_status "Profiling complete. Results in ./logs/profiling/"
EOF
    
    # Create service debugging script
    cat > "$DEV_TOOLS_DIR/debug-service.sh" << 'EOF'
#!/bin/bash
# Service debugging script for AI-OTS

SERVICE_NAME=${1:-"api-gateway"}

print_status() {
    echo -e "\033[0;34m[DEBUG]\033[0m $1"
}

print_status "Debugging service: $SERVICE_NAME"

echo "=== Service Status ==="
docker-compose ps $SERVICE_NAME

echo -e "\n=== Recent Logs ==="
docker-compose logs --tail=50 $SERVICE_NAME

echo -e "\n=== Health Check ==="
case $SERVICE_NAME in
    "api-gateway")
        curl -s http://localhost:8000/health | jq . || echo "Health check failed"
        ;;
    "cache-service")
        curl -s http://localhost:8001/health | jq . || echo "Health check failed"
        ;;
    "data-ingestion")
        curl -s http://localhost:8002/health | jq . || echo "Health check failed"
        ;;
    "analytics")
        curl -s http://localhost:8003/health | jq . || echo "Health check failed"
        ;;
    *)
        echo "No specific health check for $SERVICE_NAME"
        ;;
esac

echo -e "\n=== Resource Usage ==="
docker stats --no-stream $SERVICE_NAME

echo -e "\n=== Environment Variables ==="
docker inspect $SERVICE_NAME | jq '.[0].Config.Env' || echo "Could not retrieve environment"
EOF
    
    # Create local testing script
    cat > "$DEV_TOOLS_DIR/run-tests.sh" << 'EOF'
#!/bin/bash
# Enhanced testing script for AI-OTS

TEST_TYPE=${1:-"all"}
VERBOSE=${2:-false}

print_status() {
    echo -e "\033[0;34m[TEST]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# Create test results directory
mkdir -p ./logs/testing

case $TEST_TYPE in
    "unit")
        print_status "Running unit tests..."
        python -m pytest tests/ -v --tb=short --junitxml=./logs/testing/unit_results.xml
        ;;
    "integration")
        print_status "Running integration tests..."
        python -m pytest tests/test_comprehensive.py -v --tb=short --junitxml=./logs/testing/integration_results.xml
        ;;
    "performance")
        print_status "Running performance tests..."
        ./dev-tools/profile-performance.sh signals 30
        ;;
    "ai-models")
        print_status "Running AI model tests..."
        cd ai-models && python debug_signals.py && cd ..
        ;;
    "all")
        print_status "Running all tests..."
        $0 unit
        $0 integration
        $0 performance
        $0 ai-models
        ;;
    *)
        print_error "Unknown test type: $TEST_TYPE"
        echo "Available types: unit, integration, performance, ai-models, all"
        exit 1
        ;;
esac

print_success "Testing complete. Results in ./logs/testing/"
EOF
    
    chmod +x "$DEV_TOOLS_DIR"/*.sh
    
    print_dev "Developer experience tools created successfully"
}

# Function to create enhanced directories
create_enhanced_directories() {
    print_status "Creating enhanced directory structure..."
    
    # Original directories
    mkdir -p logs/{gateway,cache,data-ingestion,analytics,nginx,setup,profiling,testing}
    mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
    mkdir -p nginx/{conf.d,ssl}
    mkdir -p notebooks
    mkdir -p services/{signals,portfolio}
    
    # Enhanced directories for AI development
    mkdir -p "$AI_MODEL_DIR"
    mkdir -p "$DEV_TOOLS_DIR"
    mkdir -p data/{historical,real-time,backtesting,validation}
    mkdir -p experiments/{notebooks,scripts,results}
    mkdir -p docs/{api,development,deployment}
    
    # Create placeholder files for services not yet implemented
    echo '<html><body><h1>AI-OTS Signals Service</h1><p>Enhanced with AI Models</p></body></html>' > services/signals/placeholder.html
    echo '<html><body><h1>AI-OTS Portfolio Service</h1><p>Enhanced with Risk Management</p></body></html>' > services/portfolio/placeholder.html
    
    print_success "Enhanced directories created"
}

# Function to setup enhanced environment
setup_enhanced_environment() {
    print_status "Setting up enhanced environment configuration..."
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating enhanced .env from .env.example..."
        cp .env.example .env
        
        # Add AI-specific environment variables
        cat >> .env << 'EOF'

# AI Model Configuration
AI_MODEL_PATH=./ai-models
AI_MODEL_TRAINING_ENABLED=true
AI_MODEL_VALIDATION_ENABLED=true
SIGNAL_CONFIDENCE_THRESHOLD=0.6
PATTERN_RECOGNITION_THRESHOLD=0.8

# Development Tools
HOT_RELOAD_ENABLED=true
PERFORMANCE_MONITORING_ENABLED=true
DEBUG_MODE=true
LOG_LEVEL=DEBUG

# Enhanced Monitoring
PROMETHEUS_RETENTION=30d
GRAFANA_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
ALERT_WEBHOOK_URL=

# Testing Configuration
TEST_DATA_PATH=./data/validation
BACKTESTING_ENABLED=true
PAPER_TRADING_ENABLED=true
EOF
        
        print_status "Enhanced .env file created. Please review and edit as needed."
    else
        print_success ".env file already exists"
    fi
}

# Enhanced service health check
enhanced_health_check() {
    print_status "Performing enhanced health check..."
    
    # Core services health check
    services=(
        "API Gateway:http://localhost:8000/health"
        "Cache Service:http://localhost:8001/health"
        "Data Ingestion:http://localhost:8002/health"
        "Analytics:http://localhost:8003/health"
    )
    
    echo "=============================================="
    echo "Enhanced Health Check Summary:"
    echo "=============================================="
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        url=$(echo $service | cut -d: -f2-)
        
        if curl -f -s "$url" >/dev/null 2>&1; then
            echo -e "✅ $name: ${GREEN}Healthy${NC}"
        else
            echo -e "❌ $name: ${RED}Unhealthy${NC}"
        fi
    done
    
    # Check AI model environment
    if [ -d "$AI_MODEL_DIR" ]; then
        echo -e "✅ AI Model Environment: ${GREEN}Ready${NC}"
    else
        echo -e "❌ AI Model Environment: ${RED}Not Ready${NC}"
    fi
    
    # Check developer tools
    if [ -d "$DEV_TOOLS_DIR" ]; then
        echo -e "✅ Developer Tools: ${GREEN}Ready${NC}"
    else
        echo -e "❌ Developer Tools: ${RED}Not Ready${NC}"
    fi
    
    echo
}

# Main script
main() {
    echo "=============================================="
    echo "AI-OTS Enhanced Development Setup (BDT-P1)"
    echo "=============================================="
    echo
    
    # Check system resources
    check_system_resources
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
    
    if ! command_exists python3; then
        print_warning "Python 3 not found. AI model development features will be limited."
    fi
    
    if ! command_exists curl; then
        print_error "curl is required for health checks. Please install curl."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
    echo
    
    # Setup enhanced environment
    setup_enhanced_environment
    echo
    
    # Create enhanced directories
    create_enhanced_directories
    echo
    
    # Setup AI model development environment
    setup_ml_environment
    echo
    
    # Setup developer experience tools
    setup_developer_tools
    echo
    
    # Stop any existing containers
    print_status "Stopping any existing containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Build and start core services
    print_status "Building and starting core services..."
    docker-compose up -d timescaledb redis
    
    # Wait for databases to be ready
    wait_for_service "TimescaleDB" "http://localhost:5432" 60 || {
        print_error "TimescaleDB failed to start"
        docker-compose logs timescaledb
        exit 1
    }
    
    wait_for_service "Redis" "http://localhost:6379" 30 || {
        print_error "Redis failed to start"
        docker-compose logs redis
        exit 1
    }
    
    # Start application services
    print_status "Starting application services..."
    docker-compose up -d cache-service data-ingestion analytics api-gateway
    
    # Wait for services to be ready
    wait_for_service "Cache Service" "http://localhost:8001/health" 45
    wait_for_service "Data Ingestion Service" "http://localhost:8002/health" 45
    wait_for_service "Analytics Service" "http://localhost:8003/health" 45
    wait_for_service "API Gateway" "http://localhost:8000/health" 45
    
    # Start placeholder services
    print_status "Starting placeholder services..."
    docker-compose up -d signals portfolio
    
    echo
    print_success "All services are running!"
    echo
    
    # Enhanced health check
    enhanced_health_check
    
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
        
        wait_for_service "Prometheus" "http://localhost:9090" 30
        wait_for_service "Grafana" "http://localhost:3000" 30
        
        echo "📊 Prometheus:         http://localhost:9090"
        echo "📈 Grafana:            http://localhost:3000 (admin/admin)"
        echo
    fi
    
    read -p "Start development tools (Redis Commander, pgAdmin, Jupyter)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Starting development tools..."
        docker-compose --profile tools --profile development up -d
        
        wait_for_service "Redis Commander" "http://localhost:8081" 30
        wait_for_service "pgAdmin" "http://localhost:8080" 30
        wait_for_service "Jupyter" "http://localhost:8888" 30
        
        echo "🔴 Redis Commander:    http://localhost:8081"
        echo "🐘 pgAdmin:            http://localhost:8080"
        echo "📓 Jupyter:            http://localhost:8888"
        echo
    fi
    
    # Display enhanced commands
    echo "=============================================="
    echo "Enhanced Development Commands:"
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
    echo "=============================================="
    echo "AI Development Commands:"
    echo "=============================================="
    echo "🤖 Debug signals:         ./ai-models/debug_signals.py"
    echo "📊 Calibrate confidence:  ./ai-models/calibrate_confidence.py"
    echo "🔥 Hot reload:             ./dev-tools/hot-reload.sh [service]"
    echo "⚡ Profile performance:   ./dev-tools/profile-performance.sh [service]"
    echo "🐛 Debug service:         ./dev-tools/debug-service.sh [service]"
    echo "🧪 Run tests:             ./dev-tools/run-tests.sh [type]"
    echo
    echo "=============================================="
    echo "Quick Test Commands:"
    echo "=============================================="
    echo "🧪 Unit tests:            ./dev-tools/run-tests.sh unit"
    echo "🔗 Integration tests:     ./dev-tools/run-tests.sh integration"
    echo "⚡ Performance tests:     ./dev-tools/run-tests.sh performance"
    echo "🤖 AI model tests:        ./dev-tools/run-tests.sh ai-models"
    echo "🎯 All tests:             ./dev-tools/run-tests.sh all"
    echo
    
    print_success "Enhanced development environment is ready!"
    print_ai "AI model development environment configured"
    print_dev "Developer experience tools available"
    
    echo
    print_status "Next steps:"
    echo "1. Review and edit .env file if needed"
    echo "2. Install AI dependencies: pip install -r ai-models/requirements.txt"
    echo "3. Start developing with enhanced tools and AI capabilities"
    echo "4. Use ./dev-tools/run-tests.sh to validate your changes"
    echo
}

# Handle script interruption
trap 'print_error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"

