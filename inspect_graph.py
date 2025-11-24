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
    raise ValueError("❌ NEO4J_PASSWORD not found. Please check your .env file.")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

print("--- TOP 10 CONNECTED NODES IN YOUR DATABASE ---")
query = """
MATCH (n)-[r]->(m)
RETURN n.id AS Source, type(r) AS Relationship, m.id AS Target
LIMIT 10
"""

results = graph.query(query)

if not results:
    print("Graph is empty! Did graph_builder.py finish successfully?")
else:
    for res in results:
        print(f"{res['Source']} --[{res['Relationship']}]--> {res['Target']}")