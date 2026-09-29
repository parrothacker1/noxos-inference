import tomllib
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent


def load_config() -> dict:
    with open(TRAINING_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)
