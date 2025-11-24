import os
import time
import re
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from data_loader import load_medical_data

# Load secrets
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("❌ NEO4J_PASSWORD not found in .env")

# Connect to Graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# --- USING THE ONLY ACTIVE SMART MODEL ---
print("Initializing Llama-3.3-70b-versatile...")
try:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Disease", "Drug", "Symptom", "Anatomy", "Test", "Treatment"],
    allowed_relationships=["CAUSES", "TREATS", "ASSOCIATED_WITH", "AFFECTS", "PREVENTS", "IS_A"]
)

def clear_database():
    """Wipes the Neo4j database clean before starting."""
    print("\n⚠️  WARNING: Wiping existing Graph Database...")
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        print("✅ Database Cleared. Starting fresh.")
    except Exception as e:
        print(f"❌ Error clearing DB: {e}")

def extract_wait_time(error_message):
    msg = str(error_message)
    match_min = re.search(r"try again in (\d+)m(\d+)", msg)
    match_sec = re.search(r"try again in (\d+\.?\d*)s", msg)
    
    if match_min:
        return (int(match_min.group(1)) * 60) + int(match_min.group(2)) + 5
    if match_sec:
        return float(match_sec.group(1)) + 5
    return 30 

def process_in_batches(docs, start_index=0, limit=50, batch_size=1):
    # CRITICAL: Batch size 1 prevents hitting TPM (Tokens Per Minute) limits
    target_docs = docs[start_index : limit]
    print(f"\n--- FRESH INGESTION: Processing Docs {start_index} to {limit} ---")
    
    for i in range(0, len(target_docs), batch_size):
        current_idx = start_index + i
        batch = target_docs[i : i + batch_size]
        print(f"Processing Batch {(i // batch_size) + 1} (Doc {current_idx})...")
        
        success = False
        while not success:
            try:
                lc_docs = [Document(page_content=d) for d in batch]
                graph_docs = llm_transformer.convert_to_graph_documents(lc_docs)
                graph.add_graph_documents(graph_docs)
                print(f"   > ✅ Success! Graph Updated.")
                success = True
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    wait = extract_wait_time(error_str)
                    print(f"   > ⏳ Quota Limit. Sleeping {wait:.0f}s...")
                    time.sleep(wait)
                elif "model_decommissioned" in error_str:
                     print("   > ❌ Critical: Model Decommissioned. Groq changed models again.")
                     exit()
                else:
                    print(f"   > ❌ Error: {e}")
                    success = True # Skip bad batches
        
        # Polite pause to cool down rate limiter
        time.sleep(1)

if __name__ == "__main__":
    all_docs = load_medical_data()
    # 1. Clear Data
    clear_database()
    # 2. Run 50 docs slowly
    process_in_batches(all_docs, start_index=0, limit=50, batch_size=1)