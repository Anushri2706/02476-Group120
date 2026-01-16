FROM ghcr.io/astral-sh/uv:python3.12-bookworm as base 

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/mlops/train.py"]