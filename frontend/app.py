import streamlit as st
import httpx

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="🔬 Agentic RAG Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────
st.title("🔬 Agentic RAG Research Assistant")
st.markdown("*Powered by LangGraph + Groq + FAISS + ArXiv*")
st.divider()

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    backend_url = st.text_input("Backend URL", value="http://localhost:8000")
    st.markdown("---")
    st.markdown("### 🧠 How it works")
    st.markdown("""
    1. 🔍 **Search** — Queries ArXiv for papers
    2. 📚 **RAG** — Stores in FAISS vector DB
    3. ✍️ **Summarize** — LLM summarizes each paper
    4. 📝 **Report** — Generates full research report
    """)
    st.markdown("---")
    st.markdown("Built with **FastMCP + LangGraph + Groq**")

# ── Main Input ─────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Enter your research topic:",
        placeholder="e.g. Retrieval Augmented Generation, Vision Transformers, RLHF..."
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🚀 Research", use_container_width=True)

# ── Research Logic ─────────────────────────────────────
if search_btn and query:
    with st.status("🤖 Agent working...", expanded=True) as status:
        st.write("🔍 Searching ArXiv for papers...")
        st.write("📚 Building RAG vector store...")
        st.write("✍️ Summarizing papers with Groq LLM...")
        st.write("📝 Generating research report...")

        try:
            response = httpx.post(
                f"{backend_url}/research",
                json={"query": query},
                timeout=300.0
            )
            data = response.json()
            status.update(label="✅ Research complete!", state="complete")
        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"Failed to connect to backend: {e}")
            st.stop()

    st.divider()

    # ── Report ─────────────────────────────────────────
    st.subheader("📄 Research Report")
    st.markdown(data["report"])

    st.divider()

    # ── Papers ─────────────────────────────────────────
    st.subheader(f"📚 Papers Found ({len(data['papers'])})")
    for i, paper in enumerate(data["papers"], 1):
        with st.expander(f"{i}. {paper['title']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
                st.markdown(f"**Published:** {paper.get('published', 'N/A')[:10]}")
                st.markdown("**Abstract:**")
                st.write(paper["summary"][:500] + "...")
            with col2:
                st.link_button("📎 View PDF", paper["pdf_url"])

    st.divider()

    # ── RAG Context ────────────────────────────────────
    with st.expander("🔎 RAG Context Used"):
        st.text(data["context"])

elif search_btn and not query:
    st.warning("⚠️ Please enter a research topic first.")