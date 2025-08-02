# Multi-stage Dockerfile for DersLens
FROM python:3.11-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential g++ make libpq-dev libopencv-dev \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev \
      libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev ffmpeg \
      curl \
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
RUN mkdir -p /app/models /app/logs /app/uploads /app/model_cache

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
RUN mkdir -p /app/models /app/logs /app/outputs /app/model_cache

# Command for training
CMD ["python", "train.py"]

# === production stage ===
FROM base AS production

# Install nginx and supervisor for multi-service deployment
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy source code
COPY . .

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production
COPY frontend/ ./
RUN npm run build

# Setup nginx
WORKDIR /app
COPY frontend/nginx.conf /etc/nginx/sites-available/default
RUN rm -f /etc/nginx/sites-enabled/default \
    && ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

# Copy built frontend to nginx directory
RUN cp -r /app/frontend/dist/* /var/www/html/

# Create directories and set permissions
RUN mkdir -p /app/models /app/logs /app/uploads /app/model_cache \
    /var/log/supervisor \
    && chown -R appuser:appuser /app \
    && chown -R www-data:www-data /var/www/html

# Copy supervisor configuration
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:backend]
command=uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
directory=/app
user=appuser
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/backend.log
stderr_logfile=/var/log/supervisor/backend_error.log

[program:ai-service]
command=uvicorn app:app --host 0.0.0.0 --port 5000
directory=/app/ai-service
user=appuser
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/ai-service.log
stderr_logfile=/var/log/supervisor/ai-service_error.log

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/nginx.log
stderr_logfile=/var/log/supervisor/nginx_error.log
EOF

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Expose port
EXPOSE 80

# Command for production - use supervisor to manage all services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
