"""The app: routers under ``/api``, and the built frontend at ``/`` if present.

Run in development::

    uv run --group api uvicorn nflpred.api.main:app --reload

In production the multi-stage Dockerfile drops the Vite build into
``frontend/dist`` and this module serves it from the same origin, so CORS
matters only for the split dev setup (Vite on :5173, API on :8000). Interactive
API docs at ``/docs``.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nflpred.api.routes import games, performance, weeks
from nflpred.config import FRONTEND_DIST

#: Vite's dev server. The production build is same-origin and needs no entry here.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(
    title="NFL prediction dashboard API",
    summary="Read-only views over data/predictions.sqlite. Never predicts on request.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")
api.include_router(weeks.router, tags=["weeks"])
api.include_router(games.router, tags=["games"])
api.include_router(performance.router, tags=["diagnostics"])
app.include_router(api)

_DIST = FRONTEND_DIST
if (_DIST / "index.html").is_file():
    # Hashed build assets are served directly; every other non-/api path returns
    # index.html so a deep link or a refresh on a client-side route (/model,
    # /game/...) still loads the app. `StaticFiles(html=True)` alone does not do
    # this - it 404s unknown paths - hence the explicit catch-all.
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")
    _ROOT = _DIST.resolve()
    _INDEX = _ROOT / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="unknown API route")
        # Resolve before the containment check: `is_relative_to` is lexical and
        # does not collapse `..`, so `/%2e%2e/%2e%2e/pyproject.toml` would
        # otherwise escape the build tree. Order matters too - prove containment
        # before touching the filesystem.
        candidate = (_ROOT / full_path).resolve()
        if full_path and candidate.is_relative_to(_ROOT) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX)
