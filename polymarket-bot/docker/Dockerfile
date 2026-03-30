FROM python:3.11-slim

# System dependencies needed by py-clob-client (libssl for eth crypto)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY bot/ ./bot/
COPY scripts/ ./scripts/

# Create directories for logs and data
RUN mkdir -p logs data

# Default: paper trading
ENV MODE=paper \
    POLL_INTERVAL_SECONDS=60 \
    LOG_LEVEL=INFO \
    LOG_FILE=/app/logs/bot.log

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
