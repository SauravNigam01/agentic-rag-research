from fastmcp import FastMCP
import arxiv
from groq import Groq
import os

mcp = FastMCP("Research Tools")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search ArXiv for research papers on a given topic."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in search.results():
        results.append({
            "title": paper.title,
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published),
            "authors": [a.name for a in paper.authors[:3]]
        })
    return results


@mcp.tool()
def summarize_paper(text: str) -> str:
    """Summarize a research paper using LLM."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Give a clear, concise summary in bullet points."
            },
            {
                "role": "user",
                "content": f"Summarize this research paper:\n\n{text[:4000]}"
            }
        ]
    )
    return response.choices[0].message.content


@mcp.tool()
def generate_report(summaries: list[str], topic: str) -> str:
    """Generate a structured research report from multiple paper summaries."""
    combined = "\n\n---\n\n".join(summaries)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a senior research analyst. Write structured, professional reports."
            },
            {
                "role": "user",
                "content": (
                    f"Write a comprehensive research report on the topic: '{topic}'\n\n"
                    f"Based on these paper summaries:\n\n{combined}\n\n"
                    f"Include: Overview, Key Findings, Trends, and Conclusion."
                )
            }
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)