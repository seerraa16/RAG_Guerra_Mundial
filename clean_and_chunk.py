import json
import re
from config import PROCESSED_DATA_DIR, RAG_DOCUMENTS_FILE

OUTPUT_FILE = PROCESSED_DATA_DIR / "ww2_chunks.jsonl"

def clean_text(text: str) -> str:
    # Quitar referencias tipo [1], [23]
    text = re.sub(r"\[\d+\]", "", text)

    # Quitar múltiples saltos de línea
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Quitar espacios raros
    text = text.strip()

    return text


def chunk_by_paragraphs(text: str, max_length=800):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= max_length:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with open(RAG_DOCUMENTS_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in f_in:
            doc = json.loads(line)

            raw_text = doc["texto"]
            cleaned = clean_text(raw_text)

            chunks = chunk_by_paragraphs(cleaned)

            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "id": f"{doc['id']}_chunk_{i}",
                    "texto": chunk,
                    "metadata": doc.get("metadata", {})
                }
                f_out.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[OK] Chunks generados: {total_chunks}")
    print(f"[OK] Archivo: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
