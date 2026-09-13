"""The three retrieval arms compared in the evaluation.

All arms share one prompt template, one answering model and one
generation config. Only the retrieval step differs, so any measured
difference is attributable to retrieval and not to prompting.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from medgraph.llm import answering_llm, fast_llm
from medgraph.retrieval.graph_store import fetch_triples
from medgraph.stores import graph, vector_search

NO_VECTOR = "No literature passages available."
NO_GRAPH = "No graph relationships available."

# Shared by every arm. Do not copy this into an arm.
ANSWER_TEMPLATE = """Answer the question using only the context below.
If the context does not support an answer, say so.

LITERATURE CONTEXT:
{vector_context}

GRAPH CONTEXT:
{graph_context}

Question: {question}
Answer:"""

def extract_entities(question):
    """Pull medical entities out of a question using the fast model."""
    system = (
        "Extract the main medical entities (diseases, drugs, procedures) "
        "from the question as a comma-separated list. Return only the list."
    )
    chain = (
        ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{question}")]
        )
        | fast_llm()
        | StrOutputParser()
    )
    raw = chain.invoke({"question": question})
    return [e.strip() for e in raw.split(",") if e.strip()]

def _answer(question, vector_context, graph_context):
    """Single generation path shared by all arms."""
    chain = (
        ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
        | answering_llm()
        | StrOutputParser()
    )
    return chain.invoke(
        {
            "question": question,
            "vector_context": vector_context,
            "graph_context": graph_context,
        }
    )

def vector_only(question):
    """Retrieve from Chroma only."""
    hits = vector_search(question, k=2)
    context = "\n".join(text for text, _ in hits) or NO_VECTOR
    answer = _answer(question, context, NO_GRAPH)
    return {
        "arm": "vector_only",
        "answer": answer,
        "vector_context": context,
        "graph_context": "",
        "vector_top_score": hits[0][1] if hits else None,
        "triple_count": 0,
        "entities": [],
    }

def graph_only(question):
    """Retrieve from Neo4j only."""
    entities = extract_entities(question)
    triples = fetch_triples(graph(), entities)
    context = "\n".join(triples) or NO_GRAPH
    answer = _answer(question, NO_VECTOR, context)
    return {
        "arm": "graph_only",
        "answer": answer,
        "vector_context": "",
        "graph_context": context,
        "vector_top_score": None,
        "triple_count": len(triples),
        "entities": entities,
    }

def hybrid(question):
    """Retrieve from both stores."""
    entities = extract_entities(question)
    hits = vector_search(question, k=2)
    triples = fetch_triples(graph(), entities)
    vector_context = "\n".join(text for text, _ in hits) or NO_VECTOR
    graph_context = "\n".join(triples) or NO_GRAPH
    answer = _answer(question, vector_context, graph_context)
    return {
        "arm": "hybrid",
        "answer": answer,
        "vector_context": vector_context,
        "graph_context": graph_context,
        "vector_top_score": hits[0][1] if hits else None,
        "triple_count": len(triples),
        "entities": entities,
    }

ARMS = {"vector_only": vector_only, "graph_only": graph_only, "hybrid": hybrid}