# Two stages: build the React app with Node, then serve it and the API from one
# Python process. FastAPI mounts frontend/dist, so production is a single origin
# and there is no CORS or second host to configure.

# ---- stage 1: frontend build ------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: python runtime ----------------------------------------------
FROM python:3.13-slim AS runtime

# uv, pinned. Copied from the official image rather than curl|sh.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, as their own layer. `--no-default-groups --group api`
# takes fastapi + uvicorn and the core data stack (polars / numpy / scikit-learn,
# which `nflpred.evaluate` and `nflpred.predlog` import) but leaves out the
# `train` group - xgboost / lightgbm / catboost / shap - which only the fitting
# path uses. `--no-install-project` keeps this layer dependency-only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group api --no-install-project

# The application: the read-only code path and the prediction log it serves.
# A second sync installs the `nflpred` package itself now that its source is here.
COPY src/ ./src/
RUN uv sync --frozen --no-default-groups --group api
COPY data/predictions.sqlite ./data/predictions.sqlite
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "nflpred.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
