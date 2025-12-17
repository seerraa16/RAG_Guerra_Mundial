import faiss
import pickle
import json
from sentence_transformers import SentenceTransformer
from config import VECTOR_INDEX_DIR, PROCESSED_DATA_DIR
import ollama  # cliente directamente

VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Cargar chunks de todos los archivos
# -----------------------------
docs = []

for file in ["ww2_docs.jsonl", "ww2_analysis.jsonl", "ww2_extra.jsonl"]:
    path = PROCESSED_DATA_DIR / file
    if not path.exists():
        print(f"[WARN] Archivo no encontrado: {file}, se saltará.")
        continue

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Línea {line_num} en {file} ignorada por JSON inválido: {e}")
                continue

print(f"[INFO] Total de chunks cargados: {len(docs)}")

# -----------------------------
# Crear embeddings
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["texto"] for doc in docs]
embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

# Guardar embeddings
with open(VECTOR_INDEX_DIR / "embeddings.pkl", "wb") as f:
    pickle.dump({"docs": docs, "embeddings": embeddings}, f)

# -----------------------------
# Reconstruir índice FAISS
# -----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, str(VECTOR_INDEX_DIR / "faiss_index.idx"))
print("[OK] FAISS index reconstruido con todos los chunks disponibles")

# -----------------------------
# Instancia Ollama y bucle interactivo
# -----------------------------
ollama_client = ollama  # cliente de Ollama

print("=== RAG Segunda Guerra Mundial ===")
print("Escribe tu pregunta y presiona Enter. Para salir, escribe 'salir' o 'exit'.\n")

while True:
    query = input("Pregunta: ").strip()
    if query.lower() in ["salir", "exit"]:
        break
    if not query:
        continue

    query_emb = model.encode([query]).astype("float32")
    k = 5
    D, I = index.search(query_emb, k)
    context = "\n\n".join([docs[i]["texto"] for i in I[0]])

    response = ollama_client.chat(
        model="llama3:latest",
        messages=[
            {"role":"system","content":"""
Eres un experto en la Segunda Guerra Mundial. Responde únicamente usando el contexto que te proporciono.
- Si la información no está en el contexto, dilo claramente y no inventes hechos.
- Prioriza hechos históricos verificables y fechas exactas.
- Explica conceptos complejos de forma simple.
- Puedes dar ejemplos o anécdotas si están en el contexto.
- Mantén respuestas concisas, claras y estructuradas.
"""},
            {"role":"user","content":f"Contexto:\n{context}\n\nPregunta: {query}"}
        ]
    )

    # Solo mostrar el texto
    print("\n=== RESPUESTA ===")
    print(response.message if hasattr(response, "message") else "No se recibió respuesta del modelo.")
    print("\n" + "="*50 + "\n")
