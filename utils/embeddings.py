import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingService:
    """Creates sentence embeddings for document chunks and user queries."""

    def __init__(self) -> None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode_documents(self, chunks: list[dict]) -> np.ndarray:
        """
        Generate normalized embeddings for a list of document chunks.

        Parameters
        ----------
        chunks : list[dict]
            Each entry must contain a "text" key.

        Returns
        -------
        np.ndarray
            Shape (n_chunks, embedding_dim).

        Raises
        ------
        ValueError
            If the chunks list is empty.
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding.")

        texts = [chunk["text"] for chunk in chunks]

        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate a normalized embedding for a user query.

        Parameters
        ----------
        query : str

        Returns
        -------
        np.ndarray
            Shape (embedding_dim,).

        Raises
        ------
        ValueError
            If the query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
