FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim-bookworm

# Install libpq-dev for psycopg (psycopg v3 requires libpq)
# Replace ALL apt sources to avoid network issues with default deb.debian.org
RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/* \
    && echo 'deb https://mirrors.huaweicloud.com/debian/ bookworm main' > /etc/apt/sources.list \
    && echo 'deb https://mirrors.huaweicloud.com/debian/ bookworm-updates main' >> /etc/apt/sources.list \
    && echo 'deb https://mirrors.huaweicloud.com/debian-security/ bookworm-security main' >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /uvx /bin/

WORKDIR /app

COPY . .

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--http", "h11"]