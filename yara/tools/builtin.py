"""The tools that ship by default."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..memory.store import MemoryStore
from ..retrieval.retriever import HybridRetriever
from .base import Tool, ToolError


class SearchCorpusTool(Tool):
    name = "search_corpus"
    description = (
        "Search the ingested research corpus and return matching passages with "
        "their chunk ids. Always cite a returned chunk id when using a passage."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "k": {
                "type": "integer",
                "description": "Number of passages to return (1-20).",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: HybridRetriever, *, default_k: int = 6):
        self.retriever = retriever
        self.default_k = default_k

    def run(self, query: str, k: int | None = None) -> Any:
        k = max(1, min(int(k or self.default_k), 20))
        results = self.retriever.retrieve(query, k=k)
        if not results:
            return {"query": query, "results": [], "note": "No matching passages found."}
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "source": r.chunk.citation_label(),
                    "score": round(r.score, 5),
                    "text": r.chunk.text,
                }
                for r in results
            ],
        }


class RecallMemoryTool(Tool):
    name = "recall_memory"
    description = (
        "Search durable memory for findings established in earlier runs. Use this "
        "before re-deriving something that may already be known."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for in memory."},
            "k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(self, memory: MemoryStore, *, default_k: int = 5):
        self.memory = memory
        self.default_k = default_k

    def run(self, query: str, k: int | None = None) -> Any:
        facts = self.memory.recall(query, k=max(1, min(int(k or self.default_k), 10)))
        return {
            "query": query,
            "facts": [
                {
                    "text": f.text,
                    "citations": f.citations,
                    "confidence": f.confidence,
                    "similarity": round(f.score, 4),
                    "from_run": f.run_id,
                }
                for f in facts
            ],
        }


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch a web page, extract its readable text, add it to the corpus, and "
        "return the new chunk ids so its content can be cited."
    )
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
        "required": ["url"],
    }

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        chunk_size: int = 900,
        overlap: int = 150,
        allowed_domains: list[str] | None = None,
        max_chars: int = 40_000,
    ):
        self.retriever = retriever
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.allowed_domains = allowed_domains
        self.max_chars = max_chars

    def run(self, url: str) -> Any:
        from urllib.parse import urlparse

        from ..retrieval.chunking import chunk_document
        from ..retrieval.ingest import load_url

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError("only http and https URLs are supported")
        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            raise ToolError(f"domain {parsed.netloc!r} is not on the allowlist")

        try:
            doc = load_url(url)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not fetch {url}: {exc}") from exc

        doc.text = doc.text[: self.max_chars]
        chunks = chunk_document(doc, chunk_size=self.chunk_size, overlap=self.overlap)
        added = self.retriever.add(chunks)
        return {
            "url": url,
            "title": doc.title,
            "chunks_added": added,
            "chunk_ids": [c.chunk_id for c in chunks[:20]],
            "preview": doc.text[:500],
        }


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "pi": math.pi, "e": math.e,
}


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate an arithmetic expression. Use for any numeric comparison or "
        "percentage rather than computing it mentally."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. '(85 - 60) / 60 * 100'",
            }
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> Any:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"invalid expression: {exc}") from exc
        value = self._eval(tree.body)
        return {"expression": expression, "result": value}

    def _eval(self, node: ast.AST) -> Any:
        # Walks the AST by hand. No eval, no exec, no attribute access - the
        # test file has a list of things that must stay rejected.
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ToolError("only numeric literals are allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.Name) and node.id in _FUNCS:
            value = _FUNCS[node.id]
            if callable(value):
                raise ToolError(f"{node.id} must be called with arguments")
            return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _FUNCS.get(node.func.id)
            if not callable(func):
                raise ToolError(f"unsupported function {node.func.id!r}")
            return func(*[self._eval(a) for a in node.args])
        raise ToolError(f"unsupported expression element: {type(node).__name__}")
