"""Parameterised Cypher access to the Neo4j knowledge graph.

All Cypher for MedGraph lives here. No f-strings, no .format(),
no string concatenation. Values from the LLM or the user are passed
as query parameters only.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Cypher parameters can replace VALUES only -
# never labels, relationship types or property names.
NEIGHBOURS_QUERY = """
MATCH (n)-[r]-(m)
WHERE toLower(n.id) CONTAINS toLower($entity)
   OR toLower(m.id) CONTAINS toLower($entity)
RETURN n.id AS source, type(r) AS rel, m.id AS target
LIMIT $limit
"""

def normalise(text):
    """Canonicalise an entity string for matching.

    This is NOT a security control - parameterisation provides the
    security guarantee. This only improves match quality by stripping
    possessives and collapsing whitespace.
    """
    if not text:
        return ""
    text = text.replace("'s","").replace("\u2019s", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def fetch_neighbours(graph, entity, limit=10):
    """Fetch 1-hop neighbours for a single entity.

    Returns a list of {source, rel, target} dicts; empty on failure.
    """
    entity = normalise(entity)
    if not entity:
        return []
    try:
        return graph.query(
            NEIGHBOURS_QUERY,
            params={"entity": entity, "limit": int(limit)},
        )
    except Exception as exc:
        logger.warning("graph query failed for entity=%r: %s", entity, exc)
        return []

def fetch_triples(graph, entities, limit=10):
    """Many entites -> flat list of 'Source REL Target' strings."""
    triples = []
    for entity in entities:
        for record in fetch_neighbours(graph, entity, limit):
            triples.append(f"{record['source']} {record['rel']} {record['target']}")
    return triples
