FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Create non-root user (HF Spaces runs as UID 1000)
RUN useradd -m -u 1000 user

# Copy workspace definition and lockfile first for layer caching
COPY --chown=user pyproject.toml uv.lock ./
COPY --chown=user packages/eval-entity-resolver/pyproject.toml packages/eval-entity-resolver/

# Copy source
COPY --chown=user packages/eval-entity-resolver/src packages/eval-entity-resolver/src
COPY --chown=user src src

# Install all workspace packages
RUN uv sync --no-dev

USER user
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/tmp/hf_cache
ENV LOCAL_MODE=false
ENV READ_ONLY=true

EXPOSE 7860

CMD ["uvicorn", "eval_card_registry.main:app", "--host", "0.0.0.0", "--port", "7860"]
