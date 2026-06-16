from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]  # src/ → music_tfg/
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
EDA_OUTPUTS_DIR = OUTPUTS_DIR / "eda"


def ensure_directories() -> None:
    for directory in (PROCESSED_DATA_DIR, EDA_OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)