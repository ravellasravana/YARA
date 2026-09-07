import numpy as np
import pytest

from yara.retrieval.embeddings import HashingEmbedder


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


def test_deterministic(embedder):
    a = embedder.encode(["grounding reduces hallucination"])
    b = embedder.encode(["grounding reduces hallucination"])
    np.testing.assert_array_equal(a, b)


def test_unit_norm(embedder):
    v = embedder.encode(["one", "two words here", ""])
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0)


def test_related_scores_above_unrelated(embedder):
    v = embedder.encode([
        "retrieval augmented generation grounding",
        "retrieval grounding augmented generation systems",
        "banana bread recipe with walnuts",
    ])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_weights_are_non_negative(embedder):
    assert embedder.encode(["a longer passage with many varied terms in it"]).min() >= 0.0


def test_containing_passage_never_scores_zero():
    """Signed hashing let colliding terms cancel, so a passage holding both
    query terms could score exactly 0.0. Don't let that come back."""
    e = HashingEmbedder(dim=256)
    passage = (
        "Hallucination rates fall when grounding is enforced strictly. BM25 matches "
        "rare literal tokens such as identifier XK9427 reliably. Dense retrieval "
        "generalises across paraphrase instead."
    )
    v = e.encode([passage, "hallucination grounding"])
    assert float(v[0] @ v[1]) > 0.05


def test_rejects_bad_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
