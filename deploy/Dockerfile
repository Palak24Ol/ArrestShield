FROM python:3.10.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    ARRESTSHIELD_HOST=0.0.0.0 \
    ARRESTSHIELD_PORT=8000 \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system arrestshield \
    && useradd --system --gid arrestshield --home-dir /app arrestshield

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
RUN python -m pip install --upgrade pip==26.0.1 \
    && python -m pip install -r requirements/base.txt

COPY pyproject.toml README.md ./
COPY configs configs
COPY scripts scripts
COPY src src

RUN mkdir -p /app/artifacts/models /app/artifacts/pretrained \
    && chown -R arrestshield:arrestshield /app

USER arrestshield

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["python", "scripts/run_api.py"]
