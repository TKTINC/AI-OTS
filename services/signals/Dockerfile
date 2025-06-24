# Signal Generation Service Dockerfile for AI Options Trading System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    redis-tools \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Set permissions
RUN chmod +x src/app.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash signals
RUN chown -R signals:signals /app
USER signals

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Expose port
EXPOSE 8004

# Default command
CMD ["python", "src/app.py"]

