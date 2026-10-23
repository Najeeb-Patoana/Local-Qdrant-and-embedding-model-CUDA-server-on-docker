FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY embedding_app/requirements.txt .

# Install Python dependencies (without model download)
RUN pip install --no-cache-dir \
    --no-verify-dependencies \
    -r requirements.txt

# Copy the pre-downloaded model into the container
# This assumes you've manually downloaded the model to ./models/ on your host
COPY models/ /app/models/

# Verify model is present
RUN ls -lh /app/models/ && \
    if [ ! -d "/app/models/Qwen3-Embedding-4B" ]; then \
    echo "ERROR: Model folder not found! Make sure you've downloaded it to ./models/Qwen3-Embedding-4B"; \
    exit 1; \
    fi

# Copy application code
COPY embedding_app/ .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${EMBEDDING_PORT}/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${EMBEDDING_PORT}"]