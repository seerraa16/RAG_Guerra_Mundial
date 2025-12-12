import faiss
import pickle
from config import VECTOR_INDEX_DIR
from sentence_transformers import SentenceTransformer
import ollama

# ----------------
# Cargar embeddings y documentos
# ----------------
with open(VECTOR_INDEX_DIR / "embeddings.pkl", "rb") as f:
    data = pickle.load(f)
docs = data["docs"]
embeddings = data["embeddings"]

index = faiss.read_index(str(VECTOR_INDEX_DIR / "faiss_index.idx"))

# ----------------
# Modelo de embeddings
# ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------
# Instancia Ollama
# ----------------
ollama_client = ollama

# ----------------
# Bucle interactivo
# ----------------
print("=== RAG Segunda Guerra Mundial ===")
print("Escribe tu pregunta y presiona Enter. Para salir, escribe 'salir' o 'exit'.\n")

while True:
    query = input("Pregunta: ").strip()
    if query.lower() in ["salir", "exit"]:
        print("Saliendo...")
        break
    if not query:
        continue

    # Embedding de la pregunta
    query_emb = model.encode([query]).astype("float32")

    # Búsqueda FAISS
    k = 5
    D, I = index.search(query_emb, k)
    context = "\n\n".join([docs[i]["texto"] for i in I[0]])

    # Consulta a Ollama
    response = ollama_client.chat(
        model="llama3:latest",
        messages=[
            {"role":"system","content":"Responde solo basado en el contexto proporcionado."},
            {"role":"user","content":f"Contexto:\n{context}\n\nPregunta: {query}"}
        ]
    )

    # Mostrar respuesta
    print("\n=== RESPUESTA ===")
    if hasattr(response, "message"):
        print(response.message)
    else:
        print("No se recibió respuesta del modelo.")
    print("\n" + "="*50 + "\n")
