import os

import streamlit as st
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config

from medgraph.arms import hybrid
from medgraph.config import setup_logging
from medgraph.retrieval.graph_store import fetch_neighbours
from medgraph.stores import graph as get_graph

load_dotenv()
setup_logging()

st.set_page_config(layout="wide", page_title="MedGraph AI", page_icon="🧬")

if not os.getenv("GROQ_API_KEY") or not os.getenv("NEO4J_PASSWORD"):
    st.error("🚨 API keys not found. Copy .env.example to .env and fill it in.")
    st.stop()

try:
    graph = get_graph()
    db_status = "✅ Connected"
except Exception as e:
    st.error(f"Connection failed: {e}")
    st.stop()


def get_graph_data(entities):
    """Build agraph nodes and edges from the entities' neighbours."""
    nodes = []
    edges = []
    node_ids = set()

    for entity in entities:
        for res in fetch_neighbours(graph, entity, limit=20):
            source = res["source"]
            target = res["target"]
            rel = res["rel"]

            if source not in node_ids:
                nodes.append(Node(id=source, label=source, size=25, color="#FF4B4B"))
                node_ids.add(source)
            if target not in node_ids:
                nodes.append(Node(id=target, label=target, size=25, color="#4DFF4B"))
                node_ids.add(target)
            edges.append(Edge(source=source, label=rel, target=target, color="#A0A0A0"))
    return nodes, edges


def hybrid_search_logic(question):
    """Answer using both stores. Shared with the evaluation harness."""
    result = hybrid(question)
    return result["answer"], result["entities"]


# --- UI ---
st.title("🧬 MedGraph: Hybrid Reasoning Engine")
st.caption(f"System Status: {db_status} | Model: openai/gpt-oss-20b")

col1, col2 = st.columns([55, 45])

with col1:
    st.subheader("💬 Clinical Query")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_entities" not in st.session_state:
        st.session_state.last_entities = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Ex: What treats Hirschsprung's disease?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🧠 Triangulating vector and graph data..."):
            answer, entities = hybrid_search_logic(prompt)
            st.session_state.last_entities = entities
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.subheader("🕸️ Neural Association Graph")
    if st.session_state.last_entities:
        with st.expander("Show Debug Details"):
            st.write(f"Entities: {st.session_state.last_entities}")
        with st.spinner("Rendering knowledge graph..."):
            nodes, edges = get_graph_data(st.session_state.last_entities)
            if nodes:
                config = Config(
                    width=600,
                    height=600,
                    directed=True,
                    physics=True,
                    hierarchy=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                )
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.warning("No connections found.")