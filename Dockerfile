# Stage 1: Build production React frontend
FROM node:24-slim AS frontend-builder
WORKDIR /build

COPY app/frontend/package*.json ./
RUN npm ci

COPY app/frontend ./
RUN npm run build

# Stage 2: Python backend service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[test]"

# Copy built frontend bundle from Stage 1 into the static serving path
COPY --from=frontend-builder /build/dist ./app/frontend/dist

EXPOSE 8000

# Default container entrypoint: API server
CMD ["uvicorn", "sign_language.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
