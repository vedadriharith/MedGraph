import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_loader import load_medical_data

# 1. Load Data
raw_docs = load_medical_data()

# 2. Split Text (Chunks)
# Medical texts are dense; we need smaller chunks with overlap to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.create_documents(raw_docs)

print(f"Split into {len(docs)} chunks. Embedding now... (this may take a minute)")

# 3. Create Vector Store (ChromaDB)
# We use a free, open-source medical-friendly embedding model
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_function, 
    persist_directory="./medical_chroma_db"
)

print("Vector Database created successfully!")

# 4. Test Retrieval
query = "do statins increase risk of diabetes?"
results = db.similarity_search(query, k=3)

print("\n--- TEST QUERY RESULTS ---")
for i, res in enumerate(results):
    print(f"\nResult {i+1}: {res.page_content[:300]}...")