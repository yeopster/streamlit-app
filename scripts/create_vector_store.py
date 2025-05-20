from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from chunk_documents import chunk_documents
import os

def create_vector_store(chunks, model_name="all-MiniLM-L6-v2", store_path="data/vector_store"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vector_store.save_local(store_path)
    print(f"Vector store saved to {store_path}")

if __name__ == "__main__":
    chunks = chunk_documents("data/processed/")
    create_vector_store(chunks)
