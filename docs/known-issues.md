# Known issues and deferred work

Problems found during remediation that were out of scope for the phase
in which they were found. Each names where it will be handled.

## Retrieval

- Neighbour query uses `MATCH (n)-[r]-(m)`, which is undirected. Each
  edge returns twice, one copy reversed, producing false triples such as
  "GVHD TREATS Cyclosporine". These reach the answering prompt.
  -> direction-aware retrieval (P1-4)
- Chroma contains duplicate chunks. A `k=2` search returned the same
  document twice with an identical score, so one slot was wasted. This
  weakens the vector arm and could make the hybrid arm look better for
  the wrong reason.
  -> dedup during ingestion (P1-1)

## Repo hygiene

- `requirements.txt` is missing `sentence-transformers`, which
  `langchain-huggingface` needs at runtime. A fresh clone fails on
  import. Versions are also unpinned.
  -> P0-4
- No `.env.example`. `NEO4J_DATABASE` is required for Aura instances
  whose database is not named `neo4j`, and is documented nowhere.
  -> P0-4
- `app.py` and `hybrid_rag.py` still build their own connections and
  model instances instead of using `medgraph.stores` and `medgraph.llm`.
  -> P0-4
- `langchain_community` is deprecated. `Neo4jGraph` and `Chroma` should
  move to `langchain-neo4j` and `langchain-chroma`.
  -> P0-4

## Documentation

- README names Llama 3.3 70B as the answering model. That model was
  retired by Groq on 2026-08-16; the project now uses gpt-oss.
  -> P1-8
- README claims a 26.5% improvement and a 40% to 10% hallucination
  reduction. Both are withdrawn; see the judge-model-separation ADR.
  -> P1-8

## Evaluation

- Judge reruns are not deterministic across cache misses, and the
  variance has not been measured.
  -> P1-6
- Ground truths were written by the project author. This bias is
  documented but not addressed.
  -> P1-6
- Application logging is reconfigured by `google_genai` on import, so
  the judge's own log lines are suppressed.
  -> P1-8

Entity matching too strict — sh06 question lo "Graft-Versus-Host Disease (GVHD)" ani undi, graph lo node id "GVHD". CONTAINS match kaledu, triple_count=0 vachindi. Graph lo data unna sare retrieval fail ayindi. → P1-3 (entity linking/aliases)

Graph failure vs empty result distinguish cheyyalem — log lo Neo4j connection drops kanipinchayi (Aura idle connections cut chestundi), and fetch_neighbours() aa exception ni pattukoni [] istundi. CSV lo adi triple_count=0 ga kanipistundi — "data ledu" laaga. Rendintiki ee vyatyasam CSV lo ledu. → graph_error column add cheyyali, P1-4 lo.