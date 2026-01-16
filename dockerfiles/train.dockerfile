FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/mlops/train.py"]