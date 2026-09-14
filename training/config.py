import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(REPO_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)
