FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Environment — note: inline comments are NOT allowed inside multi-line ENV.
# Each variable must be on its own line within the block.
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------------
# System dependencies
# libglib2.0-0, libsm6, libxext6, libxrender-dev are required by OpenCV
# curl is required by entrypoint.sh to poll the Triton health endpoint
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY backend /app/backend
COPY models  /app/models

# ---------------------------------------------------------------------------
# Entrypoint — waits for Triton then launches uvicorn
# ---------------------------------------------------------------------------
COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
