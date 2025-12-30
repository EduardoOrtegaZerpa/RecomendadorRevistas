import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from utils import folder_to_label, build_text

def load_json_file(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset_from_folders(data_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    if not data_root.exists():
        raise FileNotFoundError(f"No existe la ruta: {data_root}")

    for journal_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        label = folder_to_label(journal_dir.name)

        json_files = sorted(journal_dir.glob("*.json"))
        if not json_files:
            continue

        for jf in json_files:
            try:
                items = load_json_file(jf)
            except Exception as e:
                raise RuntimeError(f"Error leyendo {jf}: {e}") from e

            for it in items:
                title = it.get("title", "")
                abstract = it.get("abstract", "")
                keywords = it.get("keywords", [])

                text = build_text(title, abstract, keywords)

                rows.append({
                    "text": text,
                    "label": label,
                    "journal": it.get("journal", label),
                    "year": str(it.get("year", "")),
                    "doi": it.get("doi", ""),
                    "link": it.get("link", "")
                })

    df = pd.DataFrame(rows)
    df = df[df["text"].astype(str).str.len() > 5].reset_index(drop=True)
    return df

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
