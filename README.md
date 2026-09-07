# YARA

A multi-agent research assistant that answers a question from a corpus you
control, and verifies every citation against passages it actually retrieved.

**Status: in progress.** Being built up one piece at a time. The list below is
what exists today, not what is planned.

## Implemented

- **Decision agent** — weighted multi-criteria ranking under hard constraints.
  Deterministic: no model call, so the same input always gives the same
  ranking.
- **Chunking** — splits on markdown headings first, then packs sentences up to
  a size budget rather than cutting on a fixed window. A chunk that ends
  mid-claim can't support the claim it half contains.
- **Hashing embedder** — feature hashing over unigrams and bigrams. No model
  download, no network, no API key.

## Next

Vector store with persistence · BM25 and reciprocal rank fusion · corpus
ingestion · SQLite agent memory · LLM provider layer · tool calling · planner,
analyst, synthesiser and critic agents · orchestration · FastAPI service ·
Docker.

## Running

```bash
pip install -e ".[dev]"
pytest
```

13 tests. No API key required.

## Notes

The embedder uses **unsigned** feature hashing. Signed hashing is the standard
trick and keeps inner products unbiased in expectation, but colliding terms
with opposite signs cancel — a passage containing both query terms scored
exactly `0.0`. Unsigned turns collisions into additive noise instead, which
degrades ranking gracefully rather than destroying a true match. Regression
test in `tests/test_embeddings.py`.

## Author

Jyothi Sai Sravana Ravella —
[GitHub](https://github.com/ravellasravana) ·
[LinkedIn](https://www.linkedin.com/in/sravanaravella)

MIT licensed.
