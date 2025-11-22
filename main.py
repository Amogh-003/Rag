from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

DB_PATH = "vector_db"
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3.2", temperature=0)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def query(q):
    context_docs = retriever.invoke(q)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": q})
    print("Answer:", response.content)

if __name__ == "__main__":
    while True:
        q = input("\nAsk (or 'quit'): ")
        if q.lower() in ["quit", "exit"]:
            break
        query(q)