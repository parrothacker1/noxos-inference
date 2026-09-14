import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import REPO_ROOT, load_config

CATEGORICAL_COLUMNS = ["proto"]


def raw_file_path(raw_dir: Path, url: str) -> Path:
    return raw_dir / url.rsplit("/", 1)[-1]


def load_raw(raw_dir: Path, urls: list[str]) -> pd.DataFrame:
    paths = [raw_file_path(raw_dir, url) for url in urls]
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = [f'curl -sL "{u}" -o {p}' for u, p in zip(urls, paths) if p in missing]
        sys.exit("missing raw dataset file(s) — fetch first:\n" + "\n".join(lines))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def parse_port(value) -> float:
    text = str(value).strip()
    try:
        return float(int(text))
    except ValueError:
        pass
    try:
        return float(int(text, 16) & 0xFFFF)
    except ValueError:
        return np.nan


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["label"] = df["label"].astype(int)
    x["proto"] = df["proto"].astype(str)
    x["dst_port"] = df["dsport"].map(parse_port)
    x["src_byte_count"] = df["sbytes"].astype(float)
    x["src_packet_count"] = df["Spkts"].astype(float)
    x["dst_byte_count"] = df["dbytes"].astype(float)
    x["dst_packet_count"] = df["Dpkts"].astype(float)
    x["duration_millis"] = df["dur"].astype(float) * 1000.0
    x["handshake_latency_millis"] = df["tcprtt"].astype(float) * 1000.0
    x["smean"] = (x["src_byte_count"] / x["src_packet_count"].replace(0, np.nan)).fillna(0.0)
    x["dmean"] = (x["dst_byte_count"] / x["dst_packet_count"].replace(0, np.nan)).fillna(0.0)

    before = len(x)
    x = x.dropna(subset=["dst_port"])
    dropped = before - len(x)
    print(f"dropped {dropped} rows with unparseable dst_port (out of {before})", file=sys.stderr)

    return x


def build_features(x: pd.DataFrame, feature_columns: list[str], categories: dict[str, list[str]] | None = None):
    features = x[feature_columns].copy()
    learned_categories = categories or {}
    for col in CATEGORICAL_COLUMNS:
        if col not in learned_categories:
            learned_categories[col] = sorted(features[col].astype(str).unique().tolist())
        code_by_value = {value: code for code, value in enumerate(learned_categories[col])}
        features[col] = features[col].astype(str).map(code_by_value).fillna(-1).astype(int)
    return features, learned_categories


def main():
    config = load_config()
    raw_dir = REPO_ROOT / config["dataset"]["raw_dir"]
    model_out = REPO_ROOT / config["paths"]["model_out"]
    metrics_out = REPO_ROOT / config["paths"]["metrics_out"]
    train_cfg = config["training"]

    raw_df = load_raw(raw_dir, config["dataset"]["urls"])
    dataset = build_dataset(raw_df)

    feature_columns = [c for c in dataset.columns if c != "label"]
    y = dataset["label"]

    train_df, test_df, y_train, y_test = train_test_split(
        dataset, y, test_size=train_cfg["test_size"], random_state=train_cfg["random_state"], stratify=y
    )

    x_train, categories = build_features(train_df, feature_columns)
    x_test, _ = build_features(test_df, feature_columns, categories=categories)

    model = XGBClassifier(
        n_estimators=train_cfg["n_estimators"],
        max_depth=train_cfg["max_depth"],
        learning_rate=train_cfg["learning_rate"],
        eval_metric="logloss",
        random_state=train_cfg["random_state"],
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    normal_df = dataset[dataset["label"] == 0]
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]
    numeric_defaults = {c: float(normal_df[c].median()) for c in numeric_columns}
    categorical_defaults = {c: normal_df[c].astype(str).mode().iloc[0] for c in CATEGORICAL_COLUMNS}

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "categories": categories,
            "numeric_defaults": numeric_defaults,
            "categorical_defaults": categorical_defaults,
        },
        model_out,
    )

    metrics_out.write_text(
        "# Network classifier — held-out metrics\n\n"
        f"Trained on the full raw UNSW-NB15 dataset ({len(dataset)} rows after cleaning), "
        f"10-feature set locked in `knowledge-graph/noxos-inference/TASKS.md` "
        f"(\"Real training spec, LOCKED 2026-09-14\"), "
        f"{int((1 - train_cfg['test_size']) * 100)}/{int(train_cfg['test_size'] * 100)} stratified split, "
        f"`random_state={train_cfg['random_state']}`.\n\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- F1 (binary, attack=1): {f1:.4f}\n\n"
        "Regenerate with `uv run training/train_network_model.py` after fetching the "
        "dataset per `config.toml`'s `[dataset]` section (see `training/FEATURES.md`).\n"
    )

    print(json.dumps({"accuracy": accuracy, "f1": f1, "rows": len(dataset)}, indent=2))


if __name__ == "__main__":
    main()
