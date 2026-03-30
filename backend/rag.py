from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Free embeddings - no API key needed
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None


def add_papers_to_store(papers: list[dict]):
    global vector_store
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    for p in papers:
        chunks = splitter.split_text(p["summary"])
        texts.extend(chunks)

    if vector_store is None:
        vector_store = FAISS.from_texts(texts, embeddings)
    else:
        vector_store.add_texts(texts)
    print(f"✅ Added {len(texts)} chunks to vector store")


def retrieve_context(query: str, k: int = 3) -> str:
    if vector_store is None:
        return "No papers loaded yet."
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])