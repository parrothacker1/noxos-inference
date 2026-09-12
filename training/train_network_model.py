import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
MODEL_OUT = REPO_ROOT / "service" / "model_network.joblib"
METRICS_OUT = REPO_ROOT / "training" / "METRICS.md"

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
DROP_COLUMNS = ["id", "attack_cat", "label"]


def load_dataset() -> pd.DataFrame:
    if not RAW_CSV.exists():
        sys.exit(
            f"missing {RAW_CSV} — fetch it first:\n"
            f'curl -sL "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv" -o {RAW_CSV}'
        )
    return pd.read_csv(RAW_CSV)


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
    df = load_dataset()
    y = df["label"].astype(int)

    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train, feature_columns, categories = build_features(train_df)
    x_test, _, _ = build_features(test_df, categories=categories)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    normal_df = df[df["label"] == 0]
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]
    numeric_defaults = {c: float(normal_df[c].median()) for c in numeric_columns}
    categorical_defaults = {c: normal_df[c].astype(str).mode().iloc[0] for c in CATEGORICAL_COLUMNS}

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "categories": categories,
            "numeric_defaults": numeric_defaults,
            "categorical_defaults": categorical_defaults,
        },
        MODEL_OUT,
    )

    METRICS_OUT.write_text(
        "# Network classifier — held-out metrics\n\n"
        f"Trained on UNSW-NB15 (`UNSW_NB15_training-set.csv`, {len(df)} rows), "
        f"80/20 stratified split, `random_state=42`.\n\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- F1 (binary, attack=1): {f1:.4f}\n\n"
        "Regenerate with `python training/train_network_model.py` after fetching the "
        "dataset (see `training/FEATURES.md`).\n"
    )

    print(json.dumps({"accuracy": accuracy, "f1": f1, "rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
