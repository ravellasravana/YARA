"""Agent memory. Three tiers in one SQLite file.

- runs/steps: what happened, kept for good. Lets a run be replayed after
  the fact instead of guessing why it produced what it did.
- scratch: working state for one run. How a finding from step 2 reaches
  the synthesiser in step 5 without re-sending the whole history.
- facts: what survives the run, looked up by embedding similarity.

SQLite rather than a service because the whole thing is one file - mount a
volume and the container keeps its memory, point tests at :memory: and they
need no teardown.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..retrieval.embeddings import Embedder, HashingEmbedder

logger = logging.getLogger("yara.memory")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'running',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    result     TEXT,
    error      TEXT,
    meta       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS steps (
    step_id    TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    agent      TEXT NOT NULL,
    action     TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    output     TEXT NOT NULL DEFAULT '{}',
    latency_ms REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, ordinal);

CREATE TABLE IF NOT EXISTS scratch (
    run_id     TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id    TEXT PRIMARY KEY,
    run_id     TEXT,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'finding',
    citations  TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    embedding  BLOB,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind);
"""


@dataclass
class RunRecord:
    run_id: str
    question: str
    status: str
    created_at: float
    updated_at: float
    result: dict[str, Any] | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    fact_id: str
    text: str
    kind: str
    citations: list[str]
    confidence: float
    run_id: str | None = None
    score: float = 0.0


class MemoryStore:
    def __init__(self, db_path: str | Path = ":memory:", embedder: Embedder | None = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- episodic ----------

    def start_run(self, question: str, **meta: Any) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, question, status, created_at, updated_at, meta) "
                "VALUES (?, ?, 'running', ?, ?, ?)",
                (run_id, question, now, now, json.dumps(meta)),
            )
            self._conn.commit()
        return run_id

    def log_step(
        self,
        run_id: str,
        *,
        agent: str,
        action: str,
        payload: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> str:
        step_id = f"step_{uuid.uuid4().hex[:12]}"
        with self._lock:
            # Ordinal derived from what's already stored rather than kept in
            # memory, so ordering survives a restart mid-run.
            row = self._conn.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 AS n FROM steps WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO steps (step_id, run_id, ordinal, agent, action, payload, "
                "output, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    step_id, run_id, row["n"], agent, action,
                    json.dumps(payload or {}, default=str),
                    json.dumps(output or {}, default=str),
                    latency_ms, time.time(),
                ),
            )
            self._conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?", (time.time(), run_id)
            )
            self._conn.commit()
        return step_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status = ?, result = ?, error = ?, updated_at = ? "
                "WHERE run_id = ?",
                (
                    status,
                    json.dumps(result, default=str) if result is not None else None,
                    error,
                    time.time(),
                    run_id,
                ),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            question=row["question"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            meta=json.loads(row["meta"]),
        )

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY ordinal", (run_id,)
            ).fetchall()
        return [
            {
                "step_id": r["step_id"],
                "ordinal": r["ordinal"],
                "agent": r["agent"],
                "action": r["action"],
                "payload": json.loads(r["payload"]),
                "output": json.loads(r["output"]),
                "latency_ms": r["latency_ms"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r for r in (self.get_run(row["run_id"]) for row in rows) if r]

    # ---------- scratchpad ----------

    def scratch_set(self, run_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO scratch (run_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (run_id, key, json.dumps(value, default=str), time.time()),
            )
            self._conn.commit()

    def scratch_get(self, run_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM scratch WHERE run_id = ? AND key = ?", (run_id, key)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def scratch_all(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM scratch WHERE run_id = ?", (run_id,)
            ).fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ---------- semantic ----------

    def remember(
        self,
        text: str,
        *,
        kind: str = "finding",
        citations: list[str] | None = None,
        confidence: float = 0.5,
        run_id: str | None = None,
    ) -> str:
        """Store a fact. Identical text merges instead of duplicating.

        Otherwise every run that rediscovers a claim writes another copy and
        memory grows with run count rather than with knowledge - recall starts
        returning the same sentence three times.
        """
        existing = self._find_by_text(text, kind)
        if existing is not None:
            merged = sorted(set(json.loads(existing["citations"])) | set(citations or []))
            with self._lock:
                self._conn.execute(
                    "UPDATE facts SET citations = ?, confidence = ? WHERE fact_id = ?",
                    (
                        json.dumps(merged),
                        max(existing["confidence"], confidence),
                        existing["fact_id"],
                    ),
                )
                self._conn.commit()
            return existing["fact_id"]

        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        vector = self.embedder.encode([text])[0].astype(np.float32)
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts (fact_id, run_id, text, kind, citations, confidence, "
                "embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_id, run_id, text, kind,
                    json.dumps(citations or []), confidence,
                    vector.tobytes(), time.time(),
                ),
            )
            self._conn.commit()
        return fact_id

    def _find_by_text(self, text: str, kind: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT fact_id, citations, confidence FROM facts WHERE text = ? AND kind = ?",
                (text, kind),
            ).fetchone()

    def recall(
        self, query: str, k: int = 5, *, kind: str | None = None, min_score: float = 0.05
    ) -> list[Fact]:
        """Look up prior facts by similarity.

        min_score is low on purpose. Hashed embeddings put genuinely related
        short texts around 0.1-0.3, and a threshold picked for neural
        embeddings threw away every real hit while the fact count still read
        non-zero - looked like it was working. It's here to drop exact-zero
        non-matches, not to rank.
        """
        sql = "SELECT * FROM facts WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return []

        matrix = np.vstack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        )
        query_vector = self.embedder.encode([query])[0]
        if matrix.shape[1] != query_vector.shape[0]:
            logger.warning("stored fact embeddings have a different dim; skipping recall")
            return []

        scores = matrix @ query_vector
        order = np.argsort(-scores)[: max(k, 0)]
        return [
            Fact(
                fact_id=rows[i]["fact_id"],
                text=rows[i]["text"],
                kind=rows[i]["kind"],
                citations=json.loads(rows[i]["citations"]),
                confidence=rows[i]["confidence"],
                run_id=rows[i]["run_id"],
                score=float(scores[i]),
            )
            for i in order
            if scores[i] >= min_score
        ]

    def count(self, table: str) -> int:
        if table not in {"runs", "steps", "facts", "scratch"}:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
