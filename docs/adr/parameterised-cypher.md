# Parameterised Cypher instead of string escaping

Status: Accepted
Date: 2026-09-13

## Context

Graph retrieval built Cypher with f-string interpolation in three places:
`hybrid_rag.get_graph_context()`, `app.get_graph_data()` and
`app.get_graph_context_text()`. All three embedded an entity string
directly into the query text.

The entity values come from an LLM extractor, which reads the user's
question. A user can therefore influence them through prompt injection,
which makes the entity string untrusted input.

A `sanitize()` helper stripped possessives and backslash-escaped single
quotes. That is escaping, not parameterisation: it blacklists characters
someone thought of, and misses Unicode quotes, comment syntax and
encoding tricks. The README nonetheless claimed the system "prevents
Cypher injection", so the documentation was wrong.

## Decision

All Cypher now lives in `medgraph/retrieval/graph_store.py` and is sent
with bound parameters:

    graph.query(NEIGHBOURS_QUERY, params={"entity": entity, "limit": limit})

The query text is fixed and parsed before values are bound, so an
injected value can never be parsed as query structure. `sanitize()` was
deleted; the whitespace and possessive handling it did survives as
`normalise()`, named so it is not mistaken for a security control.

## Consequences

- Structural guarantee rather than a filter. Verified: the payload
  `x') MATCH (n) DETACH DELETE n //` returns an empty result and leaves
  the node count unchanged.
- Cypher parameters bind values only, never labels, relationship types
  or property names. Any future dynamic label must use an allowlist.
- Centralising the query removed duplicated Cypher from three call sites.
- An empty-string entity previously matched every node, because
  `CONTAINS ''` is true for all strings. That silently pushed unrelated
  triples into the prompt. `fetch_neighbours()` now returns early on
  empty input.
- The LLM extractor may refuse an injected question, but refusal is
  non-deterministic and is not treated as a control.

## Known issue, not addressed here

`MATCH (n)-[r]-(m)` is undirected, so each edge is returned twice and one
copy is reversed, producing false triples such as "GVHD TREATS
Cyclosporine". This is a retrieval correctness problem, not a security
one, and is deferred to the direction-aware retrieval work.