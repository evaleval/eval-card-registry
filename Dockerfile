FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy workspace definition and lockfile first for layer caching
COPY pyproject.toml uv.lock ./
COPY packages/eval-entity-resolver/pyproject.toml packages/eval-entity-resolver/

# Copy source
COPY packages/eval-entity-resolver/src packages/eval-entity-resolver/src
COPY src src

# Install all workspace packages
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV LOCAL_MODE=false

EXPOSE 8000

CMD ["uvicorn", "eval_card_registry.main:app", "--host", "0.0.0.0", "--port", "8000"]
