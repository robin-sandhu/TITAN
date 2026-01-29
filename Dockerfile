# TITAN - Docker Image
# A Standardized Framework for Clinical Prediction Model Development
# Developed by Robin Sandhu
#
# Build: docker build -t titan:latest .
# Run:   docker run -v $(pwd)/data:/data -v $(pwd)/output:/app/output titan:latest python TITAN.py /data/input.csv
#
# References:
# - Bouthillier X, et al. Accounting for variance in machine learning benchmarks.
#   MLSys 2021.

FROM python:3.11-slim-bookworm

LABEL maintainer="Robin Sandhu"
LABEL version="1.0.0"
LABEL description="TITAN: A Standardized Framework for Clinical Prediction Model Development"

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY TITAN.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash titan && \
    chown -R titan:titan /app
USER titan

# Create data mount point
RUN mkdir -p /app/data /app/output

# Health check - verify imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import TITAN; print('TITAN OK')" || exit 1

# Default command shows help
CMD ["python", "TITAN.py", "--help"]
