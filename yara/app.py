"""The `YARA` facade: one object that wires the whole system from settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .agents.decision_agent import DecisionAgent
from .briefs.schema import ResearchBrief
from .config import Settings, get_settings
from .llm import get_provider
from .memory.store import MemoryStore
from .orchestrator.orchestrator import Orchestrator
from .retrieval.chunking import Chunk, Document, chunk_document
from .retrieval.embeddings import get_embedder
from .retrieval.ingest import load_path, to_chunks
from .retrieval.retriever import HybridRetriever
from .retrieval.vector_store import VectorStore
from .tools.base import ToolRegistry
from .tools.builtin import CalculatorTool, FetchUrlTool, RecallMemoryTool, SearchCorpusTool

logger = logging.getLogger("yara")


class YARA:
    """Assembled research system.

    >>> yara = YARA()
    >>> yara.ingest_path("examples/corpus")
    >>> brief = yara.research("How does retrieval reduce hallucination?")
    >>> print(brief.to_markdown())
    """

    def __init__(self, settings: Settings | None = None, *, persist: bool = True):
        self.settings = settings or get_settings()
        self.persist = persist

        embedder = get_embedder(self.settings.embedder, dim=self.settings.embedding_dim)

        if persist:
            self.settings.ensure_dirs()
            self.vector_store = VectorStore.load(self.settings.index_path, embedder)
            self.memory = MemoryStore(self.settings.db_path, embedder=embedder)
        else:
            self.vector_store = VectorStore(embedder)
            self.memory = MemoryStore(":memory:", embedder=embedder)

        self.retriever = HybridRetriever(self.vector_store, rrf_k=self.settings.rrf_k)
        self.provider = get_provider(self.settings)
        self.tools = self._build_tools()
        self.orchestrator = Orchestrator(
            self.provider,
            self.retriever,
            self.memory,
            settings=self.settings,
            tools=self.tools,
        )
        self.decision_agent = DecisionAgent()
        logger.info(
            "YARA ready: provider=%s embedder=%s corpus=%d chunks tools=%s",
            self.provider.name, embedder.name, len(self.vector_store), self.tools.names(),
        )

    def _build_tools(self) -> ToolRegistry:
        return ToolRegistry(
            [
                SearchCorpusTool(self.retriever, default_k=self.settings.top_k),
                RecallMemoryTool(self.memory),
                CalculatorTool(),
                FetchUrlTool(
                    self.retriever,
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                ),
            ]
        )

    # ---------- corpus ----------

    def ingest_text(self, text: str, *, title: str = "", source: str = "", **meta: Any) -> int:
        doc = Document.from_text(text, title=title, source=source, **meta)
        return self._add(
            chunk_document(
                doc,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
        )

    def ingest_path(self, path: str | Path) -> int:
        docs = load_path(Path(path))
        chunks = to_chunks(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        logger.info("ingesting %d document(s) -> %d chunk(s)", len(docs), len(chunks))
        return self._add(chunks)

    def _add(self, chunks: list[Chunk]) -> int:
        added = self.retriever.add(chunks)
        if added and self.persist:
            self.vector_store.save(self.settings.index_path)
        return added

    @property
    def corpus_size(self) -> int:
        return len(self.vector_store)

    # ---------- work ----------

    def research(self, question: str, *, use_tools: bool = False) -> ResearchBrief:
        if self.corpus_size == 0:
            logger.warning("corpus is empty; ingest sources before researching")
        return self.orchestrator.run(question, use_tools=use_tools)

    def decide(self, task: dict[str, Any]) -> dict[str, Any]:
        return self.decision_agent.execute(task)

    def get_brief(self, run_id: str) -> ResearchBrief | None:
        record = self.memory.get_run(run_id)
        if record is None or not record.result:
            return None
        return ResearchBrief.model_validate(record.result)

    def close(self) -> None:
        self.memory.close()

    def __enter__(self) -> YARA:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
