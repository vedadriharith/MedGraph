import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from data_loader import load_medical_data

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# Retrieve keys securely
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Validation to prevent crashes if keys are missing
if not NEO4J_PASSWORD:
    raise ValueError("❌ NEO4J_PASSWORD not found in .env file")

# 1. Connect to Graph DB
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# 2. Initialize the LLM (Llama 3 via Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 3. Initialize the Graph Transformer
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Disease", "Drug", "Symptom", "Anatomy", "Test"],
    allowed_relationships=["CAUSES", "TREATS", "ASSOCIATED_WITH", "AFFECTS", "PREVENTS"]
)

def build_knowledge_graph():
    # Load raw text
    raw_docs = load_medical_data()
    
    # Process just 50 documents for the demo run
    subset_docs = raw_docs[:50] 
    
    documents = [Document(page_content=text) for text in subset_docs]
    
    print(f"Extracting graph entities from {len(documents)} documents...")
    
    # Convert Text -> Graph Documents
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    
    print(f"Extraction complete. Found {len(graph_documents)} graph structures.")
    print("Pushing to Neo4j Database...")
    
    # Store in Neo4j
    graph.add_graph_documents(graph_documents)
    print("SUCCESS! Knowledge Graph populated.")

if __name__ == "__main__":
    build_knowledge_graph()