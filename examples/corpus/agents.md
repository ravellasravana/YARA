# Multi-Agent Systems

A multi-agent system decomposes a task across several specialised components, each with a narrow responsibility, rather than asking one general component to do everything. Applied to language models, each agent gets its own prompt, its own output schema, and often its own tools.

The argument for decomposition is that a single prompt asked to plan, retrieve, reason, write, and self-check performs each subtask worse than a prompt dedicated to one of them. Narrow prompts are also easier to evaluate, because the output of each stage has a checkable shape.

The cost is coordination overhead. Every additional agent adds latency, token spend, and a new place for a handoff to fail. Systems that decompose too aggressively spend more on message passing than on work.

# Orchestration Patterns

Static pipelines fix the sequence of agents in advance. They are predictable and cheap to debug, and they cannot adapt when a task needs a step the designer did not anticipate.

Dynamic orchestration lets a planner choose the next step at runtime. This adapts to task structure but makes cost unbounded unless the loop is explicitly capped, and it makes failures much harder to reproduce.

A common middle position is a bounded state machine: fixed stages, with iteration permitted inside a stage up to an explicit budget. This keeps worst-case cost predictable while allowing some adaptivity.

# Self-Critique and Verification

Asking a model to review its own output catches some errors, particularly formatting violations and internal contradictions. It is unreliable for factual grounding, because the same distribution that produced an unsupported claim tends to accept it on review.

Verification is more reliable when part of the check is mechanical. Confirming that a cited identifier appears in the set of documents actually retrieved is a set membership test, and it does not depend on the model's judgement at all. Reserving the model for the judgements that genuinely require language understanding, and handling the rest in code, is the more robust split.

# Tool Use

Tool calling lets a model invoke typed functions described in JSON Schema, receive the results, and continue. It moves capabilities the model lacks, such as arithmetic, retrieval, and access to live data, out of the weights and into code.

Unbounded tool loops are the standard failure mode. Without an iteration cap, a model that keeps requesting tools has no stopping condition, and cost per request becomes unbounded.
