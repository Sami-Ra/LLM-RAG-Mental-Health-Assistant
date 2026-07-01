from config import DATA_DIR, TOP_K

from utils.pdf_loader import PDFLoader
from utils.chunker import DocumentChunker
from utils.embeddings import EmbeddingService
from utils.vector_store import VectorStore
from utils.llm import LLMService
from utils.prompts import build_prompt
from utils.helpers import build_context


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm = LLMService()

        try:
            self.vector_store.load()
            print("Vector index loaded successfully.")
        except FileNotFoundError:
            print("No existing vector index found. Please build the index.")

    def build_index(self) -> dict:
        """
        Build a new FAISS vector index from all PDFs in the data directory.

        Returns
        -------
        dict
            {"pages": int, "chunks": int}

        Raises
        ------
        ValueError
            If no PDF documents are found in the data directory.
        """
        print("Loading PDF documents...")
        loader = PDFLoader(DATA_DIR)
        documents = loader.load_documents()

        if not documents:
            raise ValueError(
                f"No readable documents found in {DATA_DIR}. "
                "Please add PDF files before building the index."
            )

        print("Chunking documents...")
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        print("Generating embeddings...")
        embeddings = self.embedder.encode_documents(chunks)

        print("Building FAISS index...")
        self.vector_store.build(embeddings, chunks)
        self.vector_store.save()

        print("Index built successfully.")
        return {"pages": len(documents), "chunks": len(chunks)}

    def ask(self, question: str) -> dict:
        """
        Answer a question using Retrieval-Augmented Generation.

        Parameters
        ----------
        question : str

        Returns
        -------
        dict
            {"question": str, "answer": str, "sources": list[dict]}
        """
        question = question.strip()

        if not self.vector_store.is_built:
            return {
                "question": question,
                "answer": (
                    "The knowledge base has not been built yet. "
                    "Please click **Build / Refresh Index** in the sidebar "
                    "to index your PDF documents."
                ),
                "sources": [],
            }

        query_embedding = self.embedder.encode_query(question)
        retrieved_docs = self.vector_store.search(query_embedding, TOP_K)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": (
                    "No relevant information was found in the knowledge base "
                    "for your question."
                ),
                "sources": [],
            }

        context = build_context(retrieved_docs)
        messages = build_prompt(context, question)
        answer = self.llm.generate_response(messages)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
        }
