# Evaluating Retrieval Systems

Retrieval is evaluated separately from generation, because conflating the two hides which component is failing. Recall at k measures whether the passages containing the answer appear in the top k results at all. It sets a ceiling: a generator cannot ground an answer in evidence it was never shown.

Mean reciprocal rank and normalised discounted cumulative gain measure how highly the relevant passages are placed within that set. They matter when prompt budget forces a small k.

# Evaluating Grounded Generation

Citation coverage is the fraction of claims in an output that carry at least one citation resolving to a retrieved passage. It is cheap to compute and catches the most common failure, which is confident assertion with no source.

Coverage says nothing about whether the cited passage actually supports the claim. Attribution accuracy, which requires checking claim against passage, catches that but is far more expensive, usually needing either human annotation or a separate judge model.

Reporting coverage alone overstates system quality. A system can reach full coverage by attaching a nearby citation to every sentence regardless of relevance.

# Deterministic Baselines

An extractive baseline that selects the highest-scoring sentences from retrieved passages, with no generation at all, is a useful floor. It cannot hallucinate, because every output sentence appears verbatim in a source.

Comparing a generative system against this floor separates two things that are otherwise confounded: how much of the quality comes from retrieval, and how much comes from the generator. Systems that do not beat their extractive baseline are paying for generation without getting anything for it.
