# Evaluation

## Method

Three arms answer the same questions. Only retrieval differs: the
answering model, prompt template and generation config are shared
(`medgraph/arms.py`), so a difference between arms is attributable to
retrieval.

- `vector_only` - Chroma similarity search, k=2
- `graph_only` - Neo4j 1-hop neighbours of extracted entities
- `hybrid` - both

Answers come from Groq (`openai/gpt-oss-120b`). Scoring is by Gemini
Flash-Lite, a different provider and family; see
`docs/adr/judge-model-separation.md`. Failed judge calls are recorded as
`score=None` and excluded from averages, never replaced with a default.

## Run 2026-09-15

30 runs, 0 judge failures.

| arm | mean score (1-5) | n |
|---|---|---|
| vector_only | 1.8 | 10 |
| graph_only | 1.0 | 10 |
| hybrid | 1.8 | 10 |

**These numbers do not support any claim about hybrid retrieval.**

The graph held 3 nodes at the time of this run, seeded by hand for a
security test. `triple_count` was 0 for every row, so the graph arm
answered from an empty context and the hybrid arm was identical to the
vector arm - identical enough that the hybrid answers hit the judge
cache created by the vector arm. The run demonstrates that the harness
works; it measures nothing about the system.

A meaningful comparison needs the real corpus ingested (P1-1) and entity
linking that matches question phrasing to node ids (P1-3).

## Limitations

- Ground truths were written by the author of the system.
- One judge model. No inter-judge agreement measured.
- Judge temperature is the provider default; reruns are not
  deterministic on cache misses, and that variance is unmeasured.
- Chroma contains duplicate chunks, so k=2 can return one document
  twice.
- Retrieval failures and genuinely empty results are both recorded as
  `triple_count=0`; they are not distinguishable in the CSV.

## Withdrawn claims

The previously reported 26.5% improvement and 40% to 10% hallucination
reduction came from a script where the answering model graded itself and
failed judge calls silently became a score of 3. Both are withdrawn.