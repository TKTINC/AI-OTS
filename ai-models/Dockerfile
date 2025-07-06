# AI Model Training Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libta-lib-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ai-models/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional AI/ML packages
RUN pip install --no-cache-dir \
    tensorflow>=2.13.0 \
    torch>=2.0.0 \
    scikit-learn>=1.3.0 \
    xgboost>=1.7.0 \
    lightgbm>=4.0.0 \
    mlflow>=2.5.0 \
    optuna>=3.2.0 \
    shap>=0.42.0 \
    plotly>=5.15.0 \
    databento>=0.18.0 \
    psycopg2-binary>=2.9.0 \
    redis>=4.6.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0

# Copy AI model code
COPY ai-models/ /app/ai-models/
COPY services/ /app/services/

# Create necessary directories
RUN mkdir -p /app/data /app/cache /app/logs /var/log/ai-ots

# Set Python path
ENV PYTHONPATH=/app

# Expose port for model training API
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "ai-models.training_api:app", "--host", "0.0.0.0", "--port", "8010"]

