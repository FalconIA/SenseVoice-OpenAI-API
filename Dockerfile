FROM python:3.11-slim AS builder
WORKDIR /app

COPY .python-version pyproject.toml uv.lock ./
COPY src src
ENV UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy
RUN pip install uv && uv sync -v --frozen --no-install-project

FROM python:3.11-slim AS production

LABEL org.opencontainers.image.authors="Pengxuan Men <pengxuan.men@gmail.com>" maintainer="Pengxuan Men <pengxuan.men@gmail.com>"

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

VOLUME /app/logs
EXPOSE 9991

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY --from=mwader/static-ffmpeg:6.1.1 /ffmpeg /usr/local/bin/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "src.main"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -sf http://localhost:${PORT:-9991}/health | grep -qv '"DOWN"'
