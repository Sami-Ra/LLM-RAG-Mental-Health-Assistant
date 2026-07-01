import streamlit as st
from pathlib import Path

from config import DATA_DIR, LLM_MODEL, EMBEDDING_MODEL
from utils.rag_pipeline import RAGPipeline
from utils.helpers import format_source_title

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="LLM-RAG Mental Health Assistant",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
    }

    [data-testid="stAppViewContainer"] > .main {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }

    .block-container {
        flex: 1;
        display: flex;
        flex-direction: column;
          padding-top: 1rem !important;
    }

    .footer {
        margin-top: 40px;
        padding: 20px 0;
        border-top: 1px solid #e6e6e6;
        text-align: center;
        color: gray;
        font-size: 0.9rem;
    }

    /* Make the question box look similar to chat_input */
    div[data-testid="stTextInput"] input {
        border-radius: 14px;
        padding: 18px;
        font-size: 20px;
        color: black !important;
    }

    /* Make the label above the textbox bigger and black */
    label[data-testid="stWidgetLabel"] p {
        color: black !important;
        font-size: 22px !important;
        font-weight: 600 !important;
    }

    /* Optional: darker placeholder */
    div[data-testid="stTextInput"] input::placeholder {
        color: #666666 !important;
        font-size: 18px;
    }

    div[data-testid="stForm"] {
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Cached Resources
# ============================================================

@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

rag = load_rag_pipeline()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

   # st.title("🧠 Mental Health RAG")

 #   st.markdown("---")

    st.subheader("📚 Indexed Documents")

    pdf_files = sorted(Path(DATA_DIR).glob("*.pdf"))

    if pdf_files:
        for pdf in pdf_files:
            st.write(f"✓ {pdf.stem.replace('_', ' ')}")
    else:
        st.info("No PDF documents found in `data/pdfs/`.")

    st.markdown("---")

    st.subheader("🔄 Knowledge Base")

    if st.button("Build / Refresh Index", use_container_width=True):
        if not pdf_files:
            st.error("No PDF files found. Add PDFs to `data/pdfs/` first.")
        else:
            with st.spinner("Building vector database..."):
                try:
                    stats = rag.build_index()
                    st.success(
                        f"Indexed {stats['pages']} pages into {stats['chunks']} chunks."
                    )
                except Exception as e:
                    st.error(f"Failed to build index: {e}")

    st.markdown("---")

    st.subheader("📊 Statistics")

    st.metric("PDF Documents", len(pdf_files))
    index_status = "Ready" if rag.vector_store.is_built else "Not built"
    st.markdown(
    f"""
    <div style="margin-bottom:20px;">
        <div style="font-size:16px; color:#555;">
            <strong>Vector Index</strong>
        </div>
        <div style="font-size:24px; font-weight:600;">
            {index_status}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
    st.caption(f"LLM: {LLM_MODEL}")
    st.caption(f"Embedding Model: {EMBEDDING_MODEL}")
    st.caption("Vector Store: FAISS")

    st.markdown("---")

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.subheader("ℹ About")

    st.caption(
        "This application demonstrates Retrieval-Augmented Generation (RAG) "
        "using evidence-based mental health resources."
    )

# ============================================================
# Main Page
# ============================================================

st.title("🧠 LLM-RAG Mental Health Assistant")
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
   
This assistant provides evidence-based guidance using Retrieval-Augmented Generation (RAG). The knowledge base consists of trusted mental health resources that focus primarily on supporting **university students mental well-being**.  
 
Ask questions about evidence-based mental health resources for university students.

Responses are grounded in trusted PDF documents and include supporting evidence from the knowledge base.
"""
)

 

if not rag.vector_store.is_built:
    st.warning(
        "⚠️ The knowledge base has not been built yet. "
        "Click **Build / Refresh Index** in the sidebar to get started."
    )

# ============================================================
# Helper
# ============================================================

def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    st.divider()
    st.subheader("📚 Supporting Evidence")
    st.caption("The response above is grounded in the following retrieved passages.")

    for source in sources:
        title = format_source_title(source["source"])

        with st.expander(f"📄 {title}"):
            st.markdown(f"**Page:** {source['page']}")
            st.markdown(f"**Similarity:** {source['similarity']}%")
            st.divider()
            st.text(source["text"])

# ============================================================
# Conversation History
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

# ============================================================
# Question Box
# ============================================================

with st.form("chat_form", clear_on_submit=True):

    st.subheader("Ask a question")
    st.caption("Ask a question about anxiety, stress, CBT, or well-being.")

    question = st.text_area(
        "",
        height=100,
        placeholder="Type your question here...",
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button("Send")

if submitted and question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("Searching evidence-based resources..."):

            try:
                result = rag.ask(question)
                answer = result["answer"]
                sources = result.get("sources", [])

            except (ValueError, RuntimeError) as e:
                answer = f"An error occurred: {e}"
                sources = []

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )

# ============================================================
# Footer
# ============================================================

st.markdown(
    """
<div class="footer">

Built with <strong>Python</strong> •
<strong>Streamlit</strong> •
<strong>OpenAI</strong> •
<strong>Sentence Transformers</strong> •
<strong>FAISS</strong> •
<strong>PyMuPDF</strong>

Developed by <strong>Samira Rasouli</strong>

</div>
""",
    unsafe_allow_html=True,
)