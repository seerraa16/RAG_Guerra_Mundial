# RAG Segunda Guerra Mundial

Sistema de **Retrieval-Augmented Generation (RAG)** que responde preguntas históricas sobre la Segunda Guerra Mundial basándose **exclusivamente** en un corpus documental propio (artículos de Wikipedia + análisis histórico redactado a mano), evitando que el modelo "invente" hechos.

El flujo combina:

- **Recuperación semántica** con embeddings (`sentence-transformers`) y un índice vectorial **FAISS**.
- **Generación de respuestas** con un LLM local servido por **Ollama** (`llama3`), al que solo se le pasa como contexto los fragmentos relevantes recuperados.
- Una **interfaz web** hecha con **Streamlit** y un **modo consola** alternativo.

## ¿Cómo funciona?

1. **Descarga de datos** ([api_wiki.py](api_wiki.py))
   Descarga artículos de Wikipedia (en inglés) sobre la Segunda Guerra Mundial a partir de una lista curada de ~70 términos clave (batallas, operaciones militares, líderes, conferencias, consecuencias, etc.), usando la API pública de Wikipedia. El resultado se guarda como `wiki_docs.jsonl`.

2. **Limpieza y troceado (chunking)** ([clean_and_chunk.py](clean_and_chunk.py))
   Limpia el texto (quita referencias `[1]`, saltos de línea redundantes, etc.) y lo divide en fragmentos (*chunks*) de tamaño manejable por párrafos, para facilitar una recuperación más precisa.

3. **Corpus documental** ([wiki_clean/](wiki_clean/))
   - `ww2_docs.jsonl` — artículos completos descargados de Wikipedia.
   - `ww2_chunks.jsonl` — fragmentos generados a partir de los documentos.
   - `ww2_analysis.jsonl` — análisis histórico adicional redactado manualmente en formato pregunta/respuesta (por ejemplo, sobre la invasión de Polonia o errores logísticos alemanes en el frente oriental), para enriquecer el contexto más allá de lo que ofrece Wikipedia.

4. **Construcción del índice vectorial** ([build_rag.py](build_rag.py))
   Carga todos los chunks/documentos disponibles, genera sus embeddings con el modelo `all-MiniLM-L6-v2` y construye un índice **FAISS** (`IndexFlatL2`) que se guarda en [vector_index/](vector_index/) junto con los embeddings serializados (`embeddings.pkl`).

5. **Consulta del RAG**
   - [query_rag.py](query_rag.py): versión de **consola interactiva**. Reconstruye el índice, recibe preguntas por teclado, recupera los `k=5` fragmentos más relevantes y genera la respuesta con Ollama (`llama3:latest`), restringida al contexto recuperado.
   - [app.py](app.py): versión **web con Streamlit**, con una interfaz de chat estilizada (fondo temático, historial de conversación) que realiza el mismo proceso de búsqueda semántica + generación.

## Estructura del proyecto

```text
.
├── api_wiki.py          # Descarga de artículos de Wikipedia
├── clean_and_chunk.py   # Limpieza y troceado de texto
├── build_rag.py         # Construcción del índice FAISS
├── query_rag.py         # Consulta del RAG por consola
├── app.py                # Interfaz web (Streamlit)
├── config.py             # Rutas y parámetros de configuración
├── wiki_raw/              # Datos crudos descargados
├── wiki_clean/            # Datos limpios, chunks y análisis manual (.jsonl)
├── vector_index/          # Índice FAISS y embeddings serializados
├── images/                # Recursos gráficos (fondo de la interfaz)
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
pip install streamlit
```

Además es necesario tener [Ollama](https://ollama.com/) instalado y el modelo `llama3` descargado:

```bash
ollama pull llama3
```

## Uso

1. (Opcional) Descargar/actualizar el corpus de Wikipedia:

   ```bash
   python api_wiki.py
   python clean_and_chunk.py
   ```

2. Construir el índice vectorial:

   ```bash
   python build_rag.py
   ```

3. Consultar el sistema:
   - Por consola:

     ```bash
     python query_rag.py
     ```

   - Por interfaz web:

     ```bash
     streamlit run app.py
     ```

## Tecnologías utilizadas

- **Python**
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) para embeddings
- [FAISS](https://github.com/facebookresearch/faiss) para búsqueda de similitud vectorial
- [Ollama](https://ollama.com/) (`llama3`) como LLM local para generación
- [Streamlit](https://streamlit.io/) para la interfaz web
- API pública de Wikipedia para la obtención de datos
