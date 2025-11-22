from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

DATA_PATH = "documents"
DB_PATH = "vector_db"

def load_documents():
    loaders = {
        ".pdf": PyPDFDirectoryLoader,
        ".txt": TextLoader,
    }
    docs = []
    for file in os.listdir(DATA_PATH):
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            loader = loaders[ext](os.path.join(DATA_PATH, file) if ext == ".txt" else DATA_PATH)
            docs.extend(loader.load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def main():
    docs = load_documents()
    chunks = split_docs(docs)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(DB_PATH):
        Chroma(persist_directory=DB_PATH, embedding_function=embedding).delete_collection()
    Chroma.from_documents(chunks, embedding, persist_directory=DB_PATH)
    print(f"Indexed {len(chunks)} chunks into {DB_PATH}")

if __name__ == "__main__":
    main()