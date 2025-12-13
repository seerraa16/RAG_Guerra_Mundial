import faiss
import pickle
import json
from sentence_transformers import SentenceTransformer
from config import VECTOR_INDEX_DIR, CHUNKS_FILE

VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Cargar chunks
docs = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"[INFO] Chunks cargados: {len(docs)}")

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["texto"] for doc in docs]
embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

# Guardar embeddings
with open(VECTOR_INDEX_DIR / "embeddings.pkl", "wb") as f:
    pickle.dump({"docs": docs, "embeddings": embeddings}, f)

# FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, str(VECTOR_INDEX_DIR / "faiss_index.idx"))

print("[OK] FAISS index reconstruido con chunks")
