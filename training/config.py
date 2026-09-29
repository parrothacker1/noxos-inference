import os
import tomllib
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent


def load_config() -> dict:
    path = TRAINING_DIR / os.environ.get("NOXOS_AE_CONFIG", "config.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)
