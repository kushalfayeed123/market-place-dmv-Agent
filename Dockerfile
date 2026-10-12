FROM python:3.12-slim

# Install uv (https://docs.astral.sh/uv/guides/docker/getting-started/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app 

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install system dependencies needed for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first — leverage Docker layer cache
COPY pyproject.toml ./

# Install dependencies (no lock file present, resolve on-the-fly)
RUN uv sync --no-dev --no-install-project

# Copy the application code
COPY . .

# Expose the port the app listens on
EXPOSE 8001

# create_app is an ASGI factory — use --factory so uvicorn calls it
CMD ["sh", "-c", "uv run uvicorn gateway.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8001}"]
