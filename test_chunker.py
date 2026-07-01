from config import DATA_DIR

from utils.pdf_loader import PDFLoader
from utils.chunker import DocumentChunker

loader = PDFLoader(DATA_DIR)

documents = loader.load_documents()

chunker = DocumentChunker()

chunks = chunker.chunk_documents(documents)

print(f"Pages: {len(documents)}")
print(f"Chunks: {len(chunks)}")

print()

print(chunks[0])