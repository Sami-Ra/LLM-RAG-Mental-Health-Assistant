import faiss
import pickle
from pathlib import Path
import numpy as np

from config import INDEX_FILE, METADATA_FILE


class VectorStore:
    """Stores document embeddings in a FAISS vector database."""

    def __init__(self) -> None:
        self.index = None
        self.metadata: list[dict] = []

    @property
    def is_built(self) -> bool:
        """Return True if the vector index is loaded and ready for search."""
        return self.index is not None

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Build a FAISS inner-product index from document embeddings.

        Embeddings must be L2-normalized so that inner product equals cosine similarity.
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.asarray(embeddings, dtype=np.float32))
        self.metadata = metadata

    def save(self) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        """
        Load a previously saved FAISS index and metadata from disk.

        Raises
        ------
        FileNotFoundError
            If either the index or the metadata file is missing.
        """
        if not Path(INDEX_FILE).exists() or not Path(METADATA_FILE).exists():
            raise FileNotFoundError(
                "Vector index not found. Please build the index first."
            )
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """
        Retrieve the top-K most similar chunks for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
        top_k : int

        Returns
        -------
        list[dict]
            Each entry is a metadata dict with an added "similarity" key (0–100 %).

        Raises
        ------
        RuntimeError
            If the index has not been built or loaded.
        """
        if self.index is None:
            raise RuntimeError(
                "Vector index is not available. Please build the index first."
            )

        distances, indices = self.index.search(
            np.asarray([query_embedding], dtype=np.float32),
            top_k,
        )

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            document = self.metadata[idx].copy()
            document["similarity"] = round(float(distance) * 100, 2)
            results.append(document)

        return results
