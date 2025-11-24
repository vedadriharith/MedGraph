import os
import time
import re
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from data_loader import load_medical_data

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD not found in .env")

# Connect
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Use Instant model for ingestion
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Disease", "Drug", "Symptom", "Anatomy", "Test", "Treatment"],
    allowed_relationships=["CAUSES", "TREATS", "ASSOCIATED_WITH", "AFFECTS", "PREVENTS", "IS_A"]
)

def extract_wait_time(error_message):
    match = re.search(r"try again in (\d+)m(\d+)", str(error_message))
    if match:
        return (int(match.group(1)) * 60) + int(match.group(2)) + 10
    return 60

def process_in_batches(docs, batch_size=5):
    total_docs = len(docs)
    print(f"\n--- STARTING INGESTION: {total_docs} Documents ---")
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        print(f"Processing Batch {(i // batch_size) + 1}...")
        
        success = False
        while not success:
            try:
                lc_docs = [Document(page_content=d) for d in batch]
                graph_docs = llm_transformer.convert_to_graph_documents(lc_docs)
                graph.add_graph_documents(graph_docs)
                print(f"   > Success! Added {len(graph_docs)} sub-graphs.")
                success = True
            except Exception as e:
                if "429" in str(e):
                    wait = extract_wait_time(e)
                    print(f"   > RATE LIMIT. Sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   > ERROR: {e}")
                    success = True # Skip bad batches
        time.sleep(2)

if __name__ == "__main__":
    all_docs = load_medical_data()
    process_in_batches(all_docs, batch_size=5)