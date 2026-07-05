# ──────────────────────────────────────────────────────────────────────────────
# GhostTunnel — Dockerized FastAPI Proxy Server
# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage build: compile native deps in a builder, copy into slim runtime.
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps for curl-cffi (needs libcurl + openssl headers) and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements-docker.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

# Runtime libs required by curl-cffi and opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 \
        libssl3 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY src/ ./src/

# Create data directories (for volume mounts)
RUN mkdir -p /app/data

# Expose the API port
EXPOSE 8100

# Unbuffered Python output for real-time container logs
ENV PYTHONUNBUFFERED=1

# Health check — hit the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

# Run the FastAPI server
CMD ["python", "-m", "src.server"]
