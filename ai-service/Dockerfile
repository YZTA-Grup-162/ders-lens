# Use Python 3.10 base image for better compatibility
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly with compatible versions
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies with compatible versions
RUN pip install --no-cache-dir \
    numpy==1.26.0 \
    scipy==1.11.4 \
    scikit-learn==1.6.1 \
    joblib==1.3.2 \
    torch==2.1.1 \
    torchvision==0.16.1 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.0.1 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    matplotlib==3.8.2 \
    onnx==1.15.0 \
    onnxruntime==1.16.3

# Copy application code
COPY app.py .
COPY model_architecture_fix.py .

# Create model directory structure
RUN mkdir -p /app/models_fer2013 && \
    mkdir -p /app/models_daisee && \
    mkdir -p /app/models_mendeley && \
    mkdir -p /app/models_comprehensive_fixed && \
    mkdir -p /app/models_comprehensive_real

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
