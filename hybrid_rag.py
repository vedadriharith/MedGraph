import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./medical_chroma_db", embedding_function=embedding_function)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def sanitize(text):
    return text.replace("'", "\\'").replace("'s", "").strip()

def get_graph_context(question):
    # Simplified extraction for CLI
    system_prompt = "Extract medical entities from the question as a comma-separated list."
    extractor = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")]) | llm | StrOutputParser()
    entities = [e.strip() for e in extractor.invoke({"question": question}).split(",")]
    
    context_data = []
    for entity in entities:
        clean = sanitize(entity)
        query = f"MATCH (n)-[r]-(m) WHERE toLower(n.id) CONTAINS toLower('{clean}') OR toLower(m.id) CONTAINS toLower('{clean}') RETURN n.id, type(r), m.id LIMIT 5"
        try:
            result = graph.query(query)
            for r in result: context_data.append(f"{r['n.id']} {r['type(r)']} {r['m.id']}")
        except: continue
    return "\n".join(context_data) if context_data else "No connections."

def hybrid_search(question):
    vector_context = "\n".join([doc.page_content for doc in vector_retriever.invoke(question)])
    graph_context = get_graph_context(question)
    
    template = "Answer using context.\nVector: {v}\nGraph: {g}\nQuestion: {q}"
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"v": vector_context, "g": graph_context, "q": question})

if __name__ == "__main__":
    print(hybrid_search("What treats Hirschsprung Disease?"))