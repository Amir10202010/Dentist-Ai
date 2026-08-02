# syntax=docker/dockerfile:1.7
#
# Multi-stage build. The frontend is compiled in a Node stage and only its
# output crosses into the runtime image, so no JavaScript toolchain ships to
# production.
#
#   docker build -t dentist-ai .
#   docker build --build-arg INSTALL_ML=1 -t dentist-ai:ml .   # with torch

# ---------------------------------------------------------------------------
# Stage 1 — frontend
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend

# Mirrors the repository layout, because vite.config.ts writes to
# `../src/dentist_ai/static/dist` — a flat workdir would land the bundle
# outside the build context.
WORKDIR /app/frontend

# Copied separately so a source-only change reuses the cached install layer.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build && test -f /app/src/dentist_ai/static/dist/.vite/manifest.json


# ---------------------------------------------------------------------------
# Stage 2 — Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/

ARG INSTALL_ML=0
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && if [ "$INSTALL_ML" = "1" ]; then \
           /opt/venv/bin/pip install ".[ml]"; \
       else \
           /opt/venv/bin/pip install "."; \
       fi


# ---------------------------------------------------------------------------
# Stage 3 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp is required by torch/opencv when the ML extra is installed; curl is
# used by the healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin dentist

WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
COPY --chown=dentist:dentist alembic.ini ./
COPY --chown=dentist:dentist migrations/ ./migrations/
COPY --chown=dentist:dentist src/ ./src/
COPY --from=frontend --chown=dentist:dentist \
     /app/src/dentist_ai/static/dist ./src/dentist_ai/static/dist

# Patient images and the SQLite database live here; mount a volume over it.
RUN mkdir -p /app/var/storage /app/models \
    && chown -R dentist:dentist /app/var /app/models

# Never run as root: a container escape should not start with uid 0.
USER dentist

ENV PYTHONPATH=/app/src \
    DENTIST_AI__ENVIRONMENT=production \
    DENTIST_AI__LOG_FORMAT=json \
    DENTIST_AI__STORAGE__ROOT=/app/var/storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# `--proxy-headers` so client IPs in the audit log are real rather than the
# load balancer's.
CMD ["uvicorn", "dentist_ai.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
