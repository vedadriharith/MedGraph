import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_agraph import agraph, Node, Edge, Config

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="MedGraph AI", page_icon="🧬")

# Retrieve keys from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not GROQ_API_KEY or not NEO4J_PASSWORD:
    st.error("🚨 API Keys not found! Please create a .env file with your credentials.")
    st.stop()

# --- INITIALIZATION ---
@st.cache_resource
def setup_databases():
    print("Loading Databases...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./medical_chroma_db", embedding_function=embedding_function)
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    # Using Llama 3.1 Instant for speed
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return vector_db, graph, llm

try:
    vector_db, graph, llm = setup_databases()
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    db_status = "✅ Connected"
except Exception as e:
    db_status = f"❌ Error: {e}"

# --- LOGIC ---
def sanitize(text):
    text = text.replace("'s", "") 
    text = text.replace("'", "\\'") 
    return text.strip()

def get_graph_data(entities):
    nodes = []
    edges = []
    node_ids = set()
    
    for entity in entities:
        clean_entity = sanitize(entity)
        query = f"""
        MATCH (n)-[r]-(m)
        WHERE toLower(n.id) CONTAINS toLower('{clean_entity}') OR toLower(m.id) CONTAINS toLower('{clean_entity}')
        RETURN n.id AS source, type(r) AS rel, m.id AS target
        LIMIT 20
        """
        try:
            results = graph.query(query)
            for res in results:
                source = res['source']
                target = res['target']
                rel = res['rel']
                
                if source not in node_ids:
                    nodes.append(Node(id=source, label=source, size=25, color="#FF4B4B"))
                    node_ids.add(source)
                if target not in node_ids:
                    nodes.append(Node(id=target, label=target, size=15, color="#4BFF4B"))
                    node_ids.add(target)
                edges.append(Edge(source=source, label=rel, target=target, color="#A0A0A0"))
        except Exception as e:
            print(f"Viz Error: {e}")
            
    return nodes, edges

def get_graph_context_text(entities):
    context_data = []
    for entity in entities:
        clean_entity = sanitize(entity)
        query = f"""
        MATCH (n)-[r]-(m)
        WHERE toLower(n.id) CONTAINS toLower('{clean_entity}') OR toLower(m.id) CONTAINS toLower('{clean_entity}')
        RETURN n.id AS source, type(r) AS rel, m.id AS target
        LIMIT 10
        """
        try:
            result = graph.query(query)
            for record in result:
                context_data.append(f"{record['source']} {record['rel']} {record['target']}")
        except Exception:
            continue
    return "\n".join(context_data) if context_data else "No direct graph connections found."

def hybrid_search_logic(question):
    # 1. Extract
    system_prompt = "You are a medical entity extractor. Extract the main medical concepts (diseases, drugs, procedures) from the user question. Return ONLY the entities as a comma-separated list."
    extraction_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    extractor = extraction_prompt | llm | StrOutputParser()
    entities = [e.strip() for e in extractor.invoke({"question": question}).split(",")]
    
    # 2. Retrieve
    vector_docs = vector_retriever.invoke(question)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])
    graph_context = get_graph_context_text(entities)
    
    # 3. Answer
    template = """
    You are an advanced AI Medical Assistant. Answer the question using the provided context.
    
    CONTEXT FROM VECTOR DB (Literature):
    {vector_context}
    
    CONTEXT FROM KNOWLEDGE GRAPH (Relationships):
    {graph_context}
    
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"vector_context": vector_context, "graph_context": graph_context, "question": question})
    return response, entities

# --- UI ---
st.title("🧬 MedGraph: Hybrid Reasoning Engine")
st.caption(f"System Status: {db_status} | Model: Llama 3.1 Instant")

col1, col2 = st.columns([55, 45])

with col1:
    st.subheader("💬 Clinical Query")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_entities" not in st.session_state: st.session_state.last_entities = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])
        
    if prompt := st.chat_input("Ex: What treats Hirschsprung's disease?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🧠 Triangulating Vector & Graph Data..."):
            answer, entities = hybrid_search_logic(prompt)
            st.session_state.last_entities = entities
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with col2:
    st.subheader("🕸️ Neural Association Graph")
    if st.session_state.last_entities:
        with st.expander("Show Debug Details"):
            st.write(f"Entities: {st.session_state.last_entities}")
        with st.spinner("Rendering 3D Network..."):
            nodes, edges = get_graph_data(st.session_state.last_entities)
            if nodes:
                config = Config(width=600, height=600, directed=True, physics=True, hierarchy=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.warning("No connections found.")