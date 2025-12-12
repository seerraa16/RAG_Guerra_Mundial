from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "wiki_clean"
VECTOR_INDEX_DIR = PROJECT_ROOT / "vector_index"
RAG_DOCUMENTS_FILE = PROCESSED_DATA_DIR / "ww2_docs.jsonl"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
