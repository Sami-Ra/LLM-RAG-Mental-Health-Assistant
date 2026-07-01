from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- API ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Models ----------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ---------- Directories ----------

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data" / "pdfs"

DATABASE_DIR = BASE_DIR / "database"

DATABASE_DIR.mkdir(exist_ok=True)

# ---------- Vector Store ----------

INDEX_FILE = DATABASE_DIR / "faiss_index.bin"

METADATA_FILE = DATABASE_DIR / "metadata.pkl"

# ---------- Retrieval ----------

TOP_K = 5

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# ---------- LLM ----------

TEMPERATURE = 0.2

MAX_TOKENS = 1024
