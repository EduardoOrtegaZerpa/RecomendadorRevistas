import re
from typing import List

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def join_keywords(keywords: List[str]) -> str:
    if not keywords:
        return ""
    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    return " ".join(keywords)

def folder_to_label(folder_name: str) -> str:
    # "1 Applied Ergonomics" -> "Applied Ergonomics"
    m = re.match(r"^\s*\d+\s+(.*)$", folder_name)
    return m.group(1).strip() if m else folder_name.strip()

def build_text(title: str, abstract: str, keywords: List[str]) -> str:
    title = clean_text(title)
    abstract = clean_text(abstract)
    kw = join_keywords(keywords)
    return f"{title}. {abstract}. Keywords: {kw}".strip()

def shorten_labels(labels, max_len=28):
    short = []
    for l in labels:
        if len(l) > max_len:
            short.append(l[:max_len - 1] + "…")
        else:
            short.append(l)
    return short
