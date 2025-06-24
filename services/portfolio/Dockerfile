# Portfolio Management Service Dockerfile for AI Options Trading System
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY database/ ./database/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash portfolio && \
    chown -R portfolio:portfolio /app

# Switch to non-root user
USER portfolio

# Set environment variables
ENV PYTHONPATH=/app/src
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the application
CMD ["python", "src/app.py"]

