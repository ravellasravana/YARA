"""HTTP service over the `YARA` facade.

Run it with ``uvicorn yara.api:app`` (needs ``pip install -e ".[api]"``).

The routes are thin on purpose: request models validate the input, the facade
does the work, and nothing here knows about agents or retrieval. Anything
that would be useful outside HTTP belongs in the facade instead.

Two decisions worth knowing:

- Handlers are plain ``def``, not ``async def``. A research run makes blocking
  LLM calls for seconds at a time; FastAPI runs sync handlers in a threadpool,
  so a long run doesn't freeze the event loop and ``/health`` keeps answering.
  Written as ``async def`` the same code would stall every other request.
- One lock serialises work on the facade. The vector store and BM25 index
  mutate on ingest and aren't safe to read mid-update, and SQLite serialises
  writes anyway. That caps throughput at one run at a time per process, which
  is the honest limit of an in-process index; scaling out means a shared
  vector database, not more threads.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import PlainTextResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('The API needs FastAPI. pip install "yara-research[api]"') from exc

from pydantic import BaseModel, Field

from . import __version__
from .app import YARA
from .briefs.schema import ResearchBrief
from .config import Settings

# Keeps a single request from pinning memory or the embedder. Big corpora
# should go through `yara ingest`, which reads from disk in the same process.
MAX_DOCUMENT_CHARS = 2_000_000
MAX_DOCUMENTS_PER_REQUEST = 100


# ---------- request / response models ----------


class DocumentIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    title: str = ""
    source: str = ""


class IngestRequest(BaseModel):
    """Documents travel as text rather than server-side paths.

    Accepting a path would let any client read any file the server process
    can - config, keys, other users' data. The CLI covers path ingestion for
    whoever already has shell access.
    """

    documents: list[DocumentIn] = Field(min_length=1, max_length=MAX_DOCUMENTS_PER_REQUEST)


class IngestResponse(BaseModel):
    added: int
    corpus_size: int


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    use_tools: bool = False


class DecideRequest(BaseModel):
    """Same keys :meth:`DecisionAgent.execute` reads; ``type`` is implied."""

    data: list[dict[str, Any]] | dict[str, Any]
    criteria: dict[str, float] | None = None
    complexity_bonus: float | None = None
    user_preferences: dict[str, Any] = Field(default_factory=dict)


class Health(BaseModel):
    status: str
    version: str
    provider: str
    corpus_size: int


# ---------- app ----------


def _get_yara(request: Request) -> YARA:
    return request.app.state.yara


# Annotated rather than `y: YARA = Depends(...)`: the dependency lives in the
# type, so the signature isn't lying about having a default value.
YaraDep = Annotated[YARA, Depends(_get_yara)]


def create_app(yara: YARA | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app around an existing facade, or one built at startup.

    Taking the facade as an argument is what makes the API testable: tests pass
    a non-persistent instance with a known corpus and nothing touches disk.
    When the app builds its own, it also closes it on shutdown; one it was
    handed belongs to the caller.
    """
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = yara is None
        app.state.yara = yara if yara is not None else YARA(settings)
        try:
            yield
        finally:
            if owned:
                app.state.yara.close()

    app = FastAPI(
        title="YARA",
        version=__version__,
        description="Multi-agent research assistant with citation-verified briefs.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health)
    def health(y: YaraDep) -> Health:
        # Deliberately lock-free: a liveness probe must answer while a long
        # research run holds the lock. len() on the store is a single read.
        return Health(
            status="ok",
            version=__version__,
            provider=y.provider.name,
            corpus_size=y.corpus_size,
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest, y: YaraDep) -> IngestResponse:
        with lock:
            added = sum(
                y.ingest_text(d.text, title=d.title, source=d.source) for d in body.documents
            )
            return IngestResponse(added=added, corpus_size=y.corpus_size)

    @app.post("/research", response_model=ResearchBrief)
    def research(body: ResearchRequest, y: YaraDep) -> ResearchBrief:
        if not body.question.strip():
            raise HTTPException(422, "question must not be blank")
        with lock:
            if y.corpus_size == 0:
                # 409, not 400: the request is fine, the server's state isn't.
                raise HTTPException(409, "corpus is empty; POST /ingest first")
            return y.research(body.question, use_tools=body.use_tools)

    @app.get("/research/{run_id}", response_model=ResearchBrief)
    def get_brief(run_id: str, y: YaraDep) -> ResearchBrief:
        return _brief_or_404(y, run_id)

    @app.get("/research/{run_id}/markdown", response_class=PlainTextResponse)
    def get_brief_markdown(run_id: str, y: YaraDep) -> PlainTextResponse:
        return PlainTextResponse(
            _brief_or_404(y, run_id).to_markdown(), media_type="text/markdown"
        )

    @app.post("/decide")
    def decide(body: DecideRequest, y: YaraDep) -> dict[str, Any]:
        task = {"type": "decision", **body.model_dump(exclude_none=True)}
        result = y.decide(task)
        if not result.get("recommendations") and "message" in result:
            raise HTTPException(422, result["message"])
        return result

    return app


def _brief_or_404(y: YARA, run_id: str) -> ResearchBrief:
    brief = y.get_brief(run_id)
    if brief is None:
        raise HTTPException(404, f"no completed run with id {run_id!r}")
    return brief


# For `uvicorn yara.api:app`. Cheap to build: the facade (settings, index,
# database) is only created when the server starts, in the lifespan hook.
app = create_app()
