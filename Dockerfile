# Dashboard stage: use the committed lockfile, never the checked-in bundle.
FROM node:22-bookworm-slim AS dashboard
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY --from=dashboard /app/src/surreal_memory/server/static/dist ./src/surreal_memory/server/static/dist
RUN pip install --no-cache-dir ".[server,surrealdb,embeddings-openai]"

FROM python:3.11-slim
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/acidkill/surreal-memory" \
      org.opencontainers.image.revision="${VCS_REF}"
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN useradd --uid 1000 --create-home --shell /bin/bash appuser && \
    mkdir -p /data && chown appuser:appuser /data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "surreal_memory.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
