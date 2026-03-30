from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from rag import add_papers_to_store, retrieve_context
from typing import TypedDict
import arxiv
import os
import time

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)


# ── State ──────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    papers: list
    context: str
    summaries: list
    report: str


# ── Node 1: Search ArXiv ───────────────────────────────
def search_node(state: AgentState) -> AgentState:
    print(f"\n🔍 Searching ArXiv for: {state['query']}")

    for attempt in range(3):
        try:
            search = arxiv.Search(
                query=state["query"],
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            papers = []
            for paper in search.results():
                papers.append({
                    "title": paper.title,
                    "summary": paper.summary,
                    "pdf_url": paper.pdf_url,
                    "published": str(paper.published),
                    "authors": [a.name for a in paper.authors[:3]]
                })
            print(f"✅ Found {len(papers)} papers")
            add_papers_to_store(papers)
            return {"papers": papers}
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(5)

    print("❌ ArXiv unavailable after 3 attempts")
    return {"papers": []}


# ── Node 2: Retrieve RAG Context ───────────────────────
def rag_node(state: AgentState) -> AgentState:
    print("\n📚 Retrieving RAG context...")
    context = retrieve_context(state["query"])
    print(f"✅ Context retrieved ({len(context)} chars)")
    return {"context": context}


# ── Node 3: Summarize Papers ───────────────────────────
def summarize_node(state: AgentState) -> AgentState:
    print("\n✍️  Summarizing papers...")
    summaries = []

    if not state["papers"]:
        return {"summaries": ["No papers found. ArXiv may be temporarily unavailable."]}

    for paper in state["papers"][:1]:
        try:
            response = llm.invoke([
                HumanMessage(content=(
                    f"Summarize this research paper in 3 bullet points:\n\n"
                    f"Title: {paper['title']}\n\n"
                    f"Abstract: {paper['summary'][:2000]}"
                ))
            ])
            summaries.append(response.content)
            print(f"  ✅ Summarized: {paper['title'][:50]}...")
        except Exception as e:
            print(f"  ⚠️ Summarization failed: {e}")
            summaries.append(f"Could not summarize: {paper['title']}")

    return {"summaries": summaries}


# ── Node 4: Generate Final Report ─────────────────────
def report_node(state: AgentState) -> AgentState:
    print("\n📝 Generating final report...")
    combined = "\n\n---\n\n".join(state["summaries"])

    try:
        response = llm.invoke([
            HumanMessage(content=(
                f"Write a comprehensive research report on: '{state['query']}'\n\n"
                f"RAG Context:\n{state['context']}\n\n"
                f"Paper Summaries:\n{combined}\n\n"
                f"Structure: Overview → Key Findings → Trends → Conclusion"
            ))
        ])
        print("✅ Report generated!")
        return {"report": response.content}
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return {"report": f"Report generation failed: {str(e)}"}


# ── Build Graph ────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("search",    search_node)
    graph.add_node("rag",       rag_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("report",    report_node)

    graph.set_entry_point("search")
    graph.add_edge("search",    "rag")
    graph.add_edge("rag",       "summarize")
    graph.add_edge("summarize", "report")
    graph.add_edge("report",    END)

    return graph.compile()


agent = build_agent()