import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import VECTOR_INDEX_DIR
import ollama

# -----------------------------
# Configuración general
# -----------------------------
BG_IMAGE = "https://raw.githubusercontent.com/seerraa16/RAG_Guerra_Mundial/main/images/ww2_bg.jpg"

st.set_page_config(
    page_title="RAG WW2 -- Alejandro Serrano",
    page_icon="",
    layout="centered"
)

# -----------------------------
# Estilos CSS (FINALES)
# -----------------------------
st.markdown(
    f"""
    <style>
    /* Ocultar header y footer de Streamlit */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Quitar espacio superior que deja el header */
    .block-container {{
        padding-top: 2rem;
    }}

    /* Fondo completo */
    .stApp {{
        background: linear-gradient(
            rgba(0,0,0,0.7),
            rgba(0,0,0,0.7)
        ), url("{BG_IMAGE}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Texto general */
    h1, h2, h3, p, label {{
        color: #ffffff !important;
    }}

    /* Botón Enviar */
    div.stButton > button {{
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
        border: none;
    }}

    div.stButton > button:hover {{
        background-color: #e0e0e0 !important;
        color: #000000 !important;
    }}

    /* Caja de respuesta */
    .chat-box {{
        background-color: #2a2a2a;
        color: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #c9a227;
        line-height: 1.5;
    }}

    /* Forzar texto blanco dentro de respuestas */
    .chat-box * {{
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Cargar RAG
# -----------------------------
@st.cache_resource
def load_rag():
    with open(VECTOR_INDEX_DIR / "embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    docs = data["docs"]
    index = faiss.read_index(str(VECTOR_INDEX_DIR / "faiss_index.idx"))
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return docs, index, model

docs, index, model = load_rag()

# -----------------------------
# UI
# -----------------------------
st.title("RAG WW2 -- Alejandro Serrano")
st.write(
    "Chat histórico basado **exclusivamente en documentos** sobre la Segunda Guerra Mundial."
)

# Historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
query = st.text_input("Escribe tu pregunta histórica")

if st.button("Enviar"):
    if query.strip():
        with st.spinner("🔍 Buscando en los documentos..."):
            # Embedding
            q_emb = model.encode([query]).astype("float32")

            # FAISS
            _, I = index.search(q_emb, 5)
            context = "\n\n".join(docs[i]["texto"] for i in I[0])

            # Ollama
            response = ollama.chat(
                model="llama3:latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en la Segunda Guerra Mundial. "
                            "Responde solo usando el contexto proporcionado. "
                            "Si no hay información suficiente, dilo claramente."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Contexto:\n{context}\n\nPregunta: {query}",
                    },
                ],
            )

            answer = response["message"]["content"]

            # Guardar chat
            st.session_state.chat_history.append(
                {"question": query, "answer": answer}
            )

# -----------------------------
# Mostrar chat
# -----------------------------
for chat in st.session_state.chat_history:
    st.markdown(f"**🧑 Tú:** {chat['question']}")
    st.markdown(
        f"""
        <div class="chat-box">
        🤖 <b>RAG WW2</b><br><br>
        {chat['answer']}
        </div>
        """,
        unsafe_allow_html=True
    )
