import requests
import time
import os
import re
import html
import json
from urllib.parse import quote

# =========================
# CONFIGURACIÓN
# =========================

API_KEY = ""

SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ARTICLE_URL = "https://api.elsevier.com/content/article/doi"

HEADERS = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json"
}

OUTPUT_ROOT = "elsevier_output"

JOURNALS = [
    "Neural Networks",
    "Applied Ergonomics",
    "Journal of Visual Communication and Image Representation",
    "Expert Systems with Applications"
]

START_YEAR = 2020
END_YEAR = 2024

REQUEST_DELAY = 0.01
OPENALEX_DELAY = 0.01

# =========================
# PREFIJOS DOI ACEPTADOS
# =========================
ACCEPTED_DOI_PREFIXES = [
    "10.1016/" # ScienceDirect/Elsevier
]


# =========================
# UTILIDADES
# =========================

def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_sciencedirect_link(core: dict) -> str | None:
    for link in core.get("link", []):
        if link.get("@rel") == "scidir":
            return link.get("@href")
    return None


# =========================
# SCOPUS SEARCH
# =========================

def search_articles(journal: str, year: int, start: int = 0, count: int = 25):
    query = f'SRCTITLE("{journal}") AND PUBYEAR = {year}'
    params = {
        "query": query,
        "start": start,
        "count": count
    }

    r = requests.get(SEARCH_URL, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# =========================
# ARTICLE RETRIEVAL
# =========================

def retrieve_article_by_doi(doi: str):
    encoded_doi = quote(doi, safe="")
    url = f"{ARTICLE_URL}/{encoded_doi}"

    r = requests.get(
        url,
        headers=HEADERS,
        params={"view": "META_ABS"},
        timeout=20
    )
    r.raise_for_status()
    return r.json()


# =========================
# OPENALEX KEYWORDS
# =========================

def get_openalex_keywords(doi: str) -> list[str]:
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []

        data = r.json()
        keywords = set()

        for kw in data.get("keywords", []):
            if kw.get("display_name"):
                keywords.add(kw["display_name"])

        for concept in data.get("concepts", []):
            if concept.get("display_name"):
                keywords.add(concept["display_name"])

        return sorted(keywords)

    except Exception:
        return []


# =========================
# NORMALIZACIÓN JSON
# =========================

def article_to_json(data: dict) -> dict:
    core = data["full-text-retrieval-response"]["coredata"]
    doi = core.get("prism:doi", "")

    article = {
        "title": clean_text(core.get("dc:title", "")),
        "journal": core.get("prism:publicationName", ""),
        "year": core.get("prism:coverDate", "")[:4],
        "doi": doi,
        "abstract": clean_text(core.get("dc:description", "")),
        "link": extract_sciencedirect_link(core),
        "keywords": get_openalex_keywords(doi),
        "authors": []
    }

    creators = core.get("dc:creator", [])
    if isinstance(creators, dict):
        creators = [creators]

    for creator in creators:
        name = creator.get("$", "")
        if "," in name:
            surname, given = name.split(",", 1)
            article["authors"].append({
                "surname": surname.strip(),
                "given_name": given.strip()
            })

    return article


# =========================
# PIPELINE PRINCIPAL
# =========================

def collect():
    ensure_dir(OUTPUT_ROOT)

    for journal in JOURNALS:
        journal_dir = os.path.join(OUTPUT_ROOT, sanitize(journal))
        ensure_dir(journal_dir)

        print(f"\nRevista: {journal}")

        for year in range(START_YEAR, END_YEAR + 1):
            year_dir = os.path.join(journal_dir, str(year))
            ensure_dir(year_dir)

            output_file = os.path.join(
                year_dir,
                f"{sanitize(journal)}_{year}.json"
            )

            print(f"  Año {year}")

            articles_json = []
            start = 0
            total_found = 0
            total_kept = 0

            while True:
                results = search_articles(journal, year, start=start)
                entries = results["search-results"].get("entry", [])

                if not entries:
                    break

                for e in entries:
                    doi = e.get("prism:doi")
                    if not doi:
                        continue

                    total_found += 1

                    if not any(doi.startswith(p) for p in ACCEPTED_DOI_PREFIXES):
                        continue

                    try:
                        print(f"    DOI aceptado: {doi}")
                        article_raw = retrieve_article_by_doi(doi)
                        articles_json.append(article_to_json(article_raw))
                        total_kept += 1
                        time.sleep(REQUEST_DELAY + OPENALEX_DELAY)
                    except Exception as ex:
                        print(f"    Error con DOI {doi}: {ex}")

                start += len(entries)

            if articles_json:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(articles_json, f, indent=2, ensure_ascii=False)

                print(f"    Guardado: {output_file}")
                print(f"    Artículos Scopus: {total_found}")
                print(f"    Artículos aceptados: {total_kept}")
            else:
                print("    No se encontraron artículos aceptados.")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    collect()
    print("\n   Proceso finalizado correctamente.")
