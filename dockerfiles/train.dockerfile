FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src src/
COPY tasks.py . 
COPY configshydra configshydra/
COPY tests/performancetests/locustfile.py tests/performancetests/locustfile.py/

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENV PYTHONPATH=/app
COPY entrypoint.sh ./
COPY models/latest/best_model.pth models/latest/best_model.pth 
RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]

EXPOSE 8000

CMD ["/bin/bash"]
