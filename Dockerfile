# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVELIN_TRACKER_DATA_DIR=/data

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        libfreetype6-dev \
        libjpeg62-turbo-dev \
        libpng-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt CITATION.cff ./
COPY javelin_tracker ./javelin_tracker
COPY demo ./demo
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && pip install .

RUN groupadd --system javelin && useradd --system --gid javelin --home /app javelin
RUN mkdir -p /data && chown -R javelin:javelin /data /app

USER javelin

VOLUME ["/data"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["javelin", "--help"]
