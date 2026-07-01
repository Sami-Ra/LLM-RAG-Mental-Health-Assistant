from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
    """Splits document pages into overlapping text chunks."""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Split page documents into overlapping text chunks.

        Parameters
        ----------
        documents : list[dict]
            Each entry: {"source": str, "page": int, "text": str}

        Returns
        -------
        list[dict]
            Each entry adds "chunk_id" to the document metadata.
        """
        chunks: list[dict] = []

        for document in documents:
            text_chunks = self.splitter.split_text(document["text"])
            for chunk_id, chunk in enumerate(text_chunks):
                chunks.append({
                    "text": chunk,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_id": chunk_id,
                })

        return chunks
