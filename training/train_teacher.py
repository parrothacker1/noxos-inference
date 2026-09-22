import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from config import REPO_ROOT, TRAINING_DIR, load_config

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
    models_dir = TRAINING_DIR / config["paths"]["models_dir"]
    oof_out = TRAINING_DIR / config["paths"]["oof_out"]
    metrics_out = TRAINING_DIR / config["paths"]["metrics_out"]
    train_cfg = config["training"]
    n_splits = train_cfg["n_splits"]

    raw_df = load_raw(raw_dir, config["dataset"]["urls"])
    dataset = build_dataset(raw_df).reset_index(drop=True)
    feature_columns = [c for c in dataset.columns if c != "label"]

    groups = dataset.groupby(feature_columns, sort=False).ngroup()
    print(f"{groups.nunique()} distinct feature patterns across {len(dataset)} rows", file=sys.stderr)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=train_cfg["random_state"])
    oof_proba = np.full(len(dataset), np.nan)
    categories = None

    models_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset, dataset["label"], groups=groups)):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        leaked = train_groups & test_groups
        assert not leaked, f"fold {fold}: {len(leaked)} feature patterns leaked across train/held-out — the group split is broken"

        train_df = dataset.iloc[train_idx]
        test_df = dataset.iloc[test_idx]
        x_train, categories = build_features(train_df, feature_columns)
        x_test, _ = build_features(test_df, feature_columns, categories=categories)

        model = XGBClassifier(
            n_estimators=train_cfg["n_estimators"],
            max_depth=train_cfg["max_depth"],
            learning_rate=train_cfg["learning_rate"],
            subsample=train_cfg["subsample"],
            colsample_bytree=train_cfg["colsample_bytree"],
            eval_metric="logloss",
            random_state=train_cfg["random_state"],
        )
        model.fit(x_train, train_df["label"])
        oof_proba[test_idx] = model.predict_proba(x_test)[:, 1]

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "categorical_columns": CATEGORICAL_COLUMNS,
                "categories": categories,
            },
            models_dir / f"teacher_fold_{fold}.joblib",
        )
        fold_attack_rate_train = train_df["label"].mean()
        fold_attack_rate_test = test_df["label"].mean()
        print(
            f"fold {fold}: {len(train_idx)} train / {len(test_idx)} held-out, "
            f"attack rate train={fold_attack_rate_train:.4f} held-out={fold_attack_rate_test:.4f}",
            file=sys.stderr,
        )

    assert not np.isnan(oof_proba).any(), "every row must get exactly one out-of-fold prediction"

    y = dataset["label"].to_numpy()
    predictions = (oof_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    oof_df = dataset[feature_columns + ["label"]].copy()
    oof_df["oof_attack_probability"] = oof_proba
    oof_df.to_parquet(oof_out, index=False)

    metrics_out.write_text(
        "# Teacher ensemble — out-of-fold metrics\n\n"
        f"5 XGBoost models ({train_cfg['n_estimators']} trees each, `subsample="
        f"{train_cfg['subsample']}`/`colsample_bytree={train_cfg['colsample_bytree']}` — each "
        f"tree also bags a random row/feature subset on top of the fold split, for extra "
        f"ensemble diversity), one per `StratifiedGroupKFold(n_splits={n_splits})` fold, "
        f"grouped by the exact 10-feature pattern per row (same leak-free methodology locked "
        f"in for the single-split model, see `knowledge-graph/noxos-inference/TASKS.md`). "
        f"Every row's reported prediction "
        f"comes from the one fold-model that never saw its feature-pattern during training — "
        f"this is a real held-out evaluation over the entire {len(dataset)}-row dataset, not a "
        f"single 80/20 split.\n\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- Precision: {precision:.4f}\n"
        f"- Recall: {recall:.4f}\n"
        f"- F1 (binary, attack=1): {f1:.4f}\n\n"
        f"`{oof_out.name}` holds every row's out-of-fold attack probability — this is the "
        "distillation target for the on-device student model (`student-xgboost` branch), not "
        "the raw teacher predict_proba on data a fold-model has already seen.\n"
    )

    print(json.dumps({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "rows": len(dataset)}, indent=2))


if __name__ == "__main__":
    main()
