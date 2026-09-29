import hashlib
import sys

import pandas as pd

from config import REPO_ROOT

CATEGORY_PERMISSION = "Manifest Permission"
CATEGORY_INTENT = "Intent"
CATEGORY_API = "API call signature"
CATEGORY_COMMAND = "Commands signature"


def fetch_verified(url_key: str, file_key: str, sha_key: str, cfg: dict):
    path = REPO_ROOT / cfg["raw_dir"] / cfg[file_key]
    if not path.exists():
        sys.exit(f'missing {path} — fetch first: curl -sL "{cfg[url_key]}" -o {path}')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == cfg[sha_key], f"{path.name} sha256 {got} does not match the pinned {cfg[sha_key]}"
    return path


def load_drebin(cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    features_path = fetch_verified("features_url", "features_file", "features_sha256", cfg)
    categories_path = fetch_verified("categories_url", "categories_file", "categories_sha256", cfg)

    raw = pd.read_csv(features_path, low_memory=False)
    label = raw["class"].map({"S": 1, "B": 0})
    assert label.notna().all(), "unexpected class value"
    features = raw.drop(columns="class").apply(pd.to_numeric, errors="coerce")

    valid = features.notna().all(axis=1)
    print(f"dropped {int((~valid).sum())} rows with non-numeric feature values (out of {len(features)})", file=sys.stderr)
    features = features[valid].astype(int).reset_index(drop=True)
    label = label[valid].astype(int).reset_index(drop=True)

    cat_table = pd.read_csv(categories_path, header=None, names=["feature", "category"])
    by_category: dict[str, list[str]] = {}
    for feature, category in zip(cat_table["feature"], cat_table["category"]):
        if feature in features.columns:
            by_category.setdefault(category, []).append(feature)

    covered = {f for cols in by_category.values() for f in cols}
    missing = set(features.columns) - covered
    assert not missing, f"features without a category: {sorted(missing)}"
    return features, label, by_category
