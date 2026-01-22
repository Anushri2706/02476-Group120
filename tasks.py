import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops"
PYTHON_VERSION = "3.12"


# Project commands
#* Run this with <inv preprocess-data>, function name in invoke gets mapped with "_" -> "-"
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.data.make_dataset", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.train", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)



@task
def hello_module(ctx: Context) -> None:
    """Run the hello.py script as a module to test imports."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.hello", echo=True, pty=not WINDOWS)



@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.evaluate", echo=True, pty=not WINDOWS)

@task
def visualize(ctx: Context) -> None:
    """Visualize model performance."""
    ctx.run(f"uv run python -m {PROJECT_NAME}.visualize", echo=True, pty=not WINDOWS)

@task
def api(ctx: Context) -> None:
    """Run the FastAPI application locally."""
    # --host 0.0.0.0 is REQUIRED for Docker
    ctx.run(
        f"uv run uvicorn src.{PROJECT_NAME}.api:app --host 0.0.0.0 --port 8000 --reload", 
        echo=True, 
        pty=not WINDOWS
    )