from rag import add_papers_to_store, retrieve_context

# Fake papers to test
papers = [
    {"summary": "RAG combines retrieval with generation for better LLM responses."},
    {"summary": "LangGraph enables stateful multi-agent workflows using graphs."},
    {"summary": "FAISS is a fast vector similarity search library by Facebook AI."},
]

add_papers_to_store(papers)
result = retrieve_context("how does retrieval augmented generation work?")
print("\n🔍 Retrieved Context:\n")
print(result)