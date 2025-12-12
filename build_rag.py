import faiss
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import VECTOR_INDEX_DIR, RAG_DOCUMENTS_FILE

VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Cargar documentos ya descargados por tu api_wiki.py
docs = []
with open(RAG_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

# Generar embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [doc["texto"] for doc in docs]
embeddings = model.encode(texts).astype("float32")

# Guardar embeddings + docs
with open(VECTOR_INDEX_DIR / "embeddings.pkl", "wb") as f:
    pickle.dump({"docs": docs, "embeddings": embeddings}, f)

# Crear índice FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, str(VECTOR_INDEX_DIR / "faiss_index.idx"))

print("RAG index construido correctamente.")
