# api_wiki.py - Descarga artículos de Wikipedia para RAG

import requests
import json
import time
from pathlib import Path

# Importar la carpeta de datos procesados desde config.py
from config import PROCESSED_DATA_DIR

DATA_PROCESSED = PROCESSED_DATA_DIR

# IMPORTANTE: pon aquí algo identificable tuyo (recomendado por Wikipedia)
HEADERS = {
    "User-Agent": "RAG_2_Guerra_mundial (contact: aserrcat@myuax.com)"
}

# Lista de keywords/artículos de Wikipedia
KEYWORDS = [
    "World War II",
    "Second World War",
    "Invasion of Poland",
    "Battle of France",
    "Battle of Britain",
    "Operation Sea Lion",
    "Operation Barbarossa",
    "Operation Typhoon",
    "Operation Torch",
    "Operation Husky",
    "Operation Overlord",
    "Operation Bagration",
    "Operation Market Garden",
    "Pearl Harbor",
    "Battle of Moscow",
    "Battle of Stalingrad",
    "Battle of Kursk",
    "D-Day",
    "Normandy landings",
    "Battle of Midway",
    "Battle of Guadalcanal",
    "Battle of Iwo Jima",
    "Battle of Okinawa",
    "Winston Churchill",
    "Franklin D. Roosevelt",
    "Joseph Stalin",
    "Adolf Hitler",
    "Benito Mussolini",
    "Hideki Tojo",
    "Nazi Germany",
    "Fascist Italy",
    "Imperial Japan",
    "Allies of World War II",
    "Axis powers",
    "Holocaust",
    "Nazi concentration camps",
    "Final Solution",
    "War crimes in World War II",
    "Nuremberg trials",
    "German war economy",
    "British war economy",
    "American war production",
    "Manhattan Project",
    "Atomic bombings of Hiroshima and Nagasaki",
    "Firebombing of Tokyo",
    "European theatre of World War II",
    "Pacific War",
    "Eastern Front",
    "Western Front",
    "North African Campaign",
    "Italian Campaign",
    "Post–World War II",
    "Consequences of World War II",
    "Division of Germany",
    "Cold War origins",
    "United Nations",
    "Heinrich Himmler",
    "Reinhard Heydrich",
    "Joseph Goebbels",
    "Hermann Göring",
    "SS",
    "Gestapo",
    "Wehrmacht",
    "Red Army",
    "Royal Air Force",
    "Molotov–Ribbentrop Pact",
    "Yalta Conference",
    "Potsdam Conference",
    "Tehran Conference",
    "Luftwaffe",
    "Panzer divisions",
    "Battle of El Alamein",
    "Battle of the Bulge",
]

# Función para descargar un artículo de Wikipedia
def fetch_wiki_page(title: str, lang: str = "en") -> dict | None:
    """
    Descarga una página de Wikipedia en texto plano (extract),
    resolviendo redirecciones. Devuelve un dict listo para usar en RAG.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,      # texto sin HTML
        "redirects": 1,        # seguir redirecciones
        "titles": title,
    }

    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))

    if "missing" in page:
        print(f"[WARN] Página no encontrada para: {title}")
        return None

    extract = (page.get("extract") or "").strip()
    if not extract:
        print(f"[WARN] Página sin extracto para: {title}")
        return None

    normalized_title = page.get("title", title)
    pageid = page.get("pageid")
    doc_id = f"wiki_{pageid}" if pageid is not None else f"wiki_{normalized_title.replace(' ', '_')}"

    doc = {
        "id": doc_id,
        "texto": extract,
        "fuente": "wikipedia",
        "metadata": {
            "title": normalized_title,
            "lang": lang,
            "pageid": pageid,
            "url": f"https://{lang}.wikipedia.org/?curid={pageid}" if pageid else None,
            "original_query": title,
        },
    }
    return doc

# Función principal
def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "wiki_docs.jsonl"

    docs_guardados = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for kw in KEYWORDS:
            try:
                print(f"[INFO] Descargando: {kw} ...")
                doc = fetch_wiki_page(kw, lang="en")
                if doc is None:
                    continue

                f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                docs_guardados += 1
                time.sleep(0.5)  # pequeña pausa

            except requests.HTTPError as e:
                print(f"[HTTP ERROR] '{kw}': {e}")
            except Exception as e:
                print(f"[ERROR] Problema con '{kw}': {e}")

    print(f"[DONE] Documentos de Wikipedia guardados en: {out_path}")
    print(f"[DONE] Total de documentos guardados: {docs_guardados}")

if __name__ == "__main__":
    main()
