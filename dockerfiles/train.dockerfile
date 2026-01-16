FROM ghcr.io/astral-sh/uv:python3.12-bookworm as base 

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src src/

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENV PYTHONPATH=/app

ENTRYPOINT ["uv", "run", "python","src/mlops/train.py"]