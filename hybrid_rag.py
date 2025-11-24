import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load secrets
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Setup Databases
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./medical_chroma_db", embedding_function=embedding_function)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 2})

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Use the SMART model for the reasoning engine
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def sanitize(text):
    """Cleans entities to match Neo4j format."""
    text = text.replace("'s", "") 
    text = text.replace("'", "\\'") 
    return text.strip()

def get_graph_context(entities):
    """Retrieves structured data from Neo4j."""
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
    
    if not context_data:
        return "No direct graph connections found."
    return "\n".join(context_data)

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