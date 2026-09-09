# Retrieval Augmented Generation

Retrieval augmented generation, usually shortened to RAG, conditions a language model on passages fetched from an external corpus at inference time rather than relying only on parameters fixed at training time. The retrieved passages are placed in the prompt alongside the user question, and the model is instructed to answer from them.

The central motivation is grounding. A model answering from parameters alone has no mechanism to distinguish a fact it memorised accurately from a plausible-sounding reconstruction. Supplying source passages gives the answer something external to be checked against.

Reported hallucination reductions vary widely by domain and by how strictly grounding is enforced. Benchmarks on open-domain question answering commonly report error reductions in the range of twenty to forty percent relative to an unretrieved baseline, though these numbers are sensitive to retrieval quality and to how hallucination is measured.

# Retrieval Quality Bounds Answer Quality

RAG systems inherit the failure modes of their retriever. If the retriever returns passages that do not contain the answer, the generator has two options: admit the gap, or fabricate. Models fine-tuned for helpfulness tend toward the second.

This makes recall the dominant variable in most RAG deployments. A generator that is well aligned but fed poor evidence produces confident wrong answers, and those answers carry citations, which makes them harder to spot than ungrounded ones.

# Hybrid Retrieval

Dense embedding retrieval generalises across paraphrase, matching passages that share meaning without sharing words. It systematically underperforms on rare literal tokens: identifiers, product names, acronyms, and numbers, where the embedding carries little signal.

Sparse lexical methods such as BM25 have the complementary profile. They match exact terms reliably and fail on vocabulary mismatch.

Reciprocal rank fusion combines two ranked lists by summing the reciprocal of each item's rank in each list, offset by a smoothing constant. Because it consumes ranks rather than scores, it requires no calibration between the two scoring scales, which is why it transfers across corpora without tuning.

# Chunking

Chunk size trades recall against precision. Long chunks preserve context and are more likely to contain a complete answer, but they dilute the embedding across multiple topics and waste prompt budget. Short chunks embed more sharply but frequently sever a claim from the evidence supporting it.

Splitting on document structure before splitting on length preserves semantic boundaries that fixed-width windows destroy. Overlap between adjacent chunks reduces the chance that a claim is cut in half at a boundary.
