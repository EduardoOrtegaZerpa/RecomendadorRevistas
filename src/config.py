from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASSICAL_DIR = MODELS_DIR / "classical"
BERT_DIR = MODELS_DIR / "bert"

REPORTS_DIR = OUTPUTS_DIR / "reports"

for p in [MODELS_DIR, OUTPUTS_DIR, CLASSICAL_DIR, BERT_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
