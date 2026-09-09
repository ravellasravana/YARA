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
- **Vector store** — exact cosine search over a float32 matrix, persisted to
  disk. Reloading with a different embedder discards the index rather than
  comparing vectors that aren't comparable.
- **Hybrid retrieval** — BM25, written out rather than pulled in, fused with
  dense search using reciprocal rank fusion.
- **Ingestion** — single files or whole directories. Text, markdown, HTML,
  and PDF.
- **Agent memory** — SQLite in three tiers: run history, per-run scratchpad,
  and semantic recall across runs.

## Next

LLM provider layer · tool calling · planner, analyst, synthesiser and critic
agents · orchestration with citation verification · CLI · FastAPI service ·
Docker.

## Running

```bash
pip install -e ".[dev]"
pytest
```

55 tests. No API key required.

## Two things worth knowing

**Dense search alone isn't enough.** Embeddings generalise across paraphrase
but go blind on rare literal tokens — identifiers, acronyms, version numbers.
BM25 is the other way round. Reciprocal rank fusion combines the two ranked
lists on *rank* rather than score, so the two scales never need calibrating
against each other, which is why it moves to a new corpus without retuning.
`test_surfaces_lexical_only_match` is the test that justifies the whole
arrangement.

**The embedder uses unsigned feature hashing.** Signed hashing is the standard
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
