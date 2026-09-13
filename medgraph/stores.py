"""Shared database handles.

Connections are created on first use and reused, so scripts and the
evaluation harness do not each open their own.
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph

load_dotenv()

CHROMA_DIR = "./medical_chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_vector_db = None
_graph = None

def vector_db():
    """Chroma handle, created once per process."""
    global _vector_db
    if _vector_db is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vector_db = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
    return _vector_db

def graph():
    """Neo4j handle, created once per process."""
    global _graph
    if _graph is None:
        _graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
    return _graph

def vector_search(question, k=2):
    """Return [(text, score), ...] with the similarity score exposed."""
    hits = vector_db().similarity_search_with_relevance_scores(question, k=k)
    return [(doc.page_content, score) for doc, score in hits]