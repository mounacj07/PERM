# PERM - Predictive Edge Resource Management
# Production Multi-Service Container Definition

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Ensure model directory exists
RUN mkdir -p models

EXPOSE 5000 5001 5002 5003 8000

CMD ["python", "run_all.py"]
