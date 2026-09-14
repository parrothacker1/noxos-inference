import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import REPO_ROOT, load_config

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
DROP_COLUMNS = ["id", "attack_cat", "label"]


def load_dataset(raw_csv, dataset_url) -> pd.DataFrame:
    if not raw_csv.exists():
        sys.exit(f'missing {raw_csv} — fetch it first:\ncurl -sL "{dataset_url}" -o {raw_csv}')
    return pd.read_csv(raw_csv)


def build_features(df: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    feature_columns = [c for c in df.columns if c not in DROP_COLUMNS]
    x = df[feature_columns].copy()

    learned_categories = categories or {}
    for col in CATEGORICAL_COLUMNS:
        if col not in learned_categories:
            learned_categories[col] = sorted(x[col].astype(str).unique().tolist())
        code_by_value = {value: code for code, value in enumerate(learned_categories[col])}
        x[col] = x[col].astype(str).map(code_by_value).fillna(-1).astype(int)

    return x, feature_columns, learned_categories


def main():
    config = load_config()
    raw_csv = REPO_ROOT / config["dataset"]["raw_dir"] / "UNSW_NB15_training-set.csv"
    model_out = REPO_ROOT / config["paths"]["model_out"]
    metrics_out = REPO_ROOT / config["paths"]["metrics_out"]
    train_cfg = config["training"]

    df = load_dataset(raw_csv, config["dataset"]["url"])
    y = df["label"].astype(int)

    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=train_cfg["test_size"], random_state=train_cfg["random_state"], stratify=y
    )

    x_train, feature_columns, categories = build_features(train_df)
    x_test, _, _ = build_features(test_df, categories=categories)

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

    normal_df = df[df["label"] == 0]
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
        f"Trained on UNSW-NB15 (`UNSW_NB15_training-set.csv`, {len(df)} rows), "
        f"{int((1 - train_cfg['test_size']) * 100)}/{int(train_cfg['test_size'] * 100)} stratified split, "
        f"`random_state={train_cfg['random_state']}`.\n\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- F1 (binary, attack=1): {f1:.4f}\n\n"
        "Regenerate with `uv run training/train_network_model.py` after fetching the "
        "dataset per `config.toml`'s `[dataset]` section (see `training/FEATURES.md`).\n"
    )

    print(json.dumps({"accuracy": accuracy, "f1": f1, "rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
