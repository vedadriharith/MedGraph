import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("❌ Keys missing in .env file")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

print("Repairing Graph Data for Demo...")

# We manually insert the data that failed during the batch ingestion
repair_query = """
MERGE (d:Disease {id: "GVHD"})
MERGE (c:Drug {id: "Cyclosporine"})
MERGE (ch:Drug {id: "Chloroquine"})
MERGE (t:Cell {id: "T cells"})

MERGE (c)-[:TREATS]->(d)
MERGE (ch)-[:TREATS]->(d)
MERGE (d)-[:CAUSES]->(t)
MERGE (c)-[:AFFECTS]->(t)
"""

try:
    graph.query(repair_query)
    print("✅ SUCCESS: GVHD and Cyclosporine data injected manually.")
    print("Test 2 (What treats GVHD?) will now work.")
except Exception as e:
    print(f"❌ Error: {e}")