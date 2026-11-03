FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Optional provider extras: --build-arg EXTRAS=anthropic (or openai, or anthropic,openai).
ARG EXTRAS=""

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ -n "$EXTRAS" ]; then pip install ".[$EXTRAS]"; else pip install .; fi

RUN useradd --create-home --uid 10001 pulso && mkdir /data && chown pulso /data
USER pulso

# Threads (and paused refund approvals) survive restarts when /data is a volume.
ENV PULSO_CHECKPOINT_DB=/data/threads.sqlite
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "pulso_support.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
