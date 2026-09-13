import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from medgraph.retrieval.graph_store import fetch_triples

# Load secrets
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

# Setup Databases
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./medical_chroma_db", embedding_function=embedding_function)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

# Use the SMART model for the reasoning engine
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def get_graph_context(entities):
    """Retrieves structured data from Neo4j."""
    triples = fetch_triples(graph, entities)
    if not triples:
        return "No direct graph connections found."
    return "\n".join(triples)

def hybrid_search(question):
    """The core RAG pipeline: Extract -> Retrieve (Vector+Graph) -> Generate."""
    
    # 1. Extract Entities
    system_prompt = "Extract the main medical entities (diseases, drugs) from the question as a comma-separated list."
    extractor = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")]) | llm | StrOutputParser()
    entities = [e.strip() for e in extractor.invoke({"question": question}).split(",")]
    
    # 2. Retrieve Context
    # Vector
    vector_docs = vector_retriever.invoke(question)
    vector_context = "\n".join([doc.page_content for doc in vector_docs])
    
    # Graph
    graph_context = get_graph_context(entities)
    
    # 3. Generate Answer
    template = """
    Answer the question using the provided context.
    
    VECTOR CONTEXT (Literature):
    {vector_context}
    
    GRAPH CONTEXT (Relationships):
    {graph_context}
    
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "vector_context": vector_context,
        "graph_context": graph_context,
        "question": question
    })

if __name__ == "__main__":
    # Quick Test
    print(hybrid_search("What treats GVHD?"))