import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import VECTOR_INDEX_DIR
import ollama

# -----------------------------
# Cargar embeddings y documentos
# -----------------------------
@st.cache_resource
def load_rag():
    with open(VECTOR_INDEX_DIR / "embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    docs = data["docs"]
    embeddings = data["embeddings"]

    index = faiss.read_index(str(VECTOR_INDEX_DIR / "faiss_index.idx"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return docs, embeddings, index, model

docs, embeddings, index, model = load_rag()

# Cliente Ollama
ollama_client = ollama

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="RAG WW2 -- Alejandro Serrano",
    page_icon="📜",
    layout="centered"
)

st.title("📜 RAG WW2 -- Alejandro Serrano")
st.markdown(
    """
    💬 Pregunta lo que quieras sobre la **Segunda Guerra Mundial**.
    Las respuestas se basan únicamente en los documentos procesados.
    """
)

# Inicializar el historial en session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input de la pregunta
query = st.text_input("Escribe tu pregunta aquí:")

if st.button("Enviar") and query:
    with st.spinner("🔍 Buscando en los documentos..."):
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
                {"role":"system","content":"""
Eres un experto en la Segunda Guerra Mundial. Responde únicamente usando el contexto que te proporciono.
- Si la información no está en el contexto, dilo claramente y no inventes hechos.
- Prioriza hechos históricos verificables y fechas exactas.
- Explica conceptos complejos de forma simple.
- Puedes dar ejemplos o anécdotas si están en el contexto.
- Mantén respuestas concisas, claras y estructuradas.
""" },
                {"role":"user","content":f"Contexto:\n{context}\n\nPregunta: {query}"}
            ]
        )

        # Extraer solo el texto limpio
        if hasattr(response, "message") and "content" in response.message:
            answer = response.message["content"]
        elif hasattr(response, "text"):
            answer = response.text
        else:
            answer = "No se recibió respuesta del modelo."

        # Guardar pregunta y respuesta en el historial
        st.session_state.chat_history.append({"question": query, "answer": answer})

# Mostrar todo el historial de chat
for chat in st.session_state.chat_history:
    st.markdown(f"**Pregunta:** {chat['question']}")
    st.success(chat["answer"])
