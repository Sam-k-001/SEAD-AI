# ─── Dockerfile — SEAD-AI ────────────────────────────────────────────────────
# Multi-stage build for production deployment
# Usage:
#   docker build -t sead-ai .
#   docker run -p 5000:5000 sead-ai

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p datasets/raw datasets/processed models/saved database static templates

# Expose Flask port
EXPOSE 5000

# Environment variables for production
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

# Run the app
CMD ["python", "app.py"]
