FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src src/
COPY tasks.py . 
COPY configshydra configshydra/

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENV PYTHONPATH=/app
COPY entrypoint.sh ./
COPY models/latest/best_model.pth models/latest/best_model.pth 
RUN chmod +x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]

EXPOSE 8000

# Since you want terminal access, we keep the default CMD or set it to bash
CMD ["/bin/bash"]
