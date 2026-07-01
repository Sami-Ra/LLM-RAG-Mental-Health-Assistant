import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About")

st.markdown(
    """
## LLM-RAG Mental Health Assistant

This application demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline
applied to evidence-based mental health resources.

---

### How It Works

1. PDF documents are loaded and parsed with **PyMuPDF**.
2. Documents are divided into overlapping text chunks.
3. Each chunk is converted into a dense vector embedding using **Sentence Transformers**.
4. Embeddings are indexed in a **FAISS** vector database.
5. User queries are embedded into the same vector space.
6. The most relevant chunks are retrieved and used as context.
7. An **OpenAI GPT** model generates a grounded response.
8. Supporting source passages are displayed alongside the answer.

---

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Interface | Streamlit |
| LLM | OpenAI GPT-4o Mini |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| PDF Processing | PyMuPDF |
| Text Splitting | LangChain Text Splitters |

---

### Author

**Samira Rasouli**

Postdoctoral Researcher — Artificial Intelligence & Machine Learning
University of Waterloo
"""
)
