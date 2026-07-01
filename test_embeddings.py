from config import DATA_DIR

from utils.pdf_loader import PDFLoader
from utils.chunker import DocumentChunker
from utils.embeddings import EmbeddingService

loader = PDFLoader(DATA_DIR)

documents = loader.load_documents()

chunker = DocumentChunker()

chunks = chunker.chunk_documents(documents)

embedder = EmbeddingService()

embeddings = embedder.encode_documents(chunks)

print()

print("Embeddings shape:")

print(embeddings.shape)