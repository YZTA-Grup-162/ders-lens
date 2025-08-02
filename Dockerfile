# Multi-stage Dockerfile for AttentionPulse Backend
FROM python:3.11-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential g++ make libpq-dev libopencv-dev \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev \
      libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps once
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === development stage ===
FROM base AS development

# Install development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    debugpy \
    pytest-asyncio \
    pytest-cov

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/datasets /app/logs /app/uploads /app/model_cache

# Expose ports
EXPOSE 8000 5678

# Command for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM development AS jupyter

# Install additional Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    matplotlib \
    seaborn \
    plotly \
    ipywidgets

# Create Jupyter config
RUN jupyter lab --generate-config

# Expose Jupyter port
EXPOSE 8888

# Command for Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# === training stage ===
FROM base AS training

# Install training dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    optuna

# Install PyTorch with CUDA support (if available)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/datasets /app/logs /app/outputs /app/model_cache

# Command for training
CMD ["python", "train.py"]

# === production stage ===
FROM base AS production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy source code
COPY . .

# Create directories and set permissions
RUN mkdir -p /app/models /app/datasets /app/logs /app/uploads /app/model_cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
