import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBRegressor

from config import REPO_ROOT, TRAINING_DIR, load_config

CATEGORICAL_COLUMNS = ["proto"]


def build_features(x: pd.DataFrame, feature_columns: list[str], categories: dict[str, list[str]] | None = None):
    features = x[feature_columns].copy()
    learned_categories = categories or {}
    for col in CATEGORICAL_COLUMNS:
        if col not in learned_categories:
            learned_categories[col] = sorted(features[col].astype(str).unique().tolist())
        code_by_value = {value: code for code, value in enumerate(learned_categories[col])}
        features[col] = features[col].astype(str).map(code_by_value).fillna(-1).astype(int)
    return features, learned_categories


def bucket_verdict(attack_probability) -> np.ndarray:
    return np.where(attack_probability < 0.4, "allow", np.where(attack_probability > 0.6, "block", "uncertain"))


def group_split(df: pd.DataFrame, feature_columns: list[str], n_splits: int, random_state: int):
    groups = df.groupby(feature_columns, sort=False).ngroup()
    print(f"{groups.nunique()} distinct feature patterns across {len(df)} rows", file=sys.stderr)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    leaked = train_groups & test_groups
    assert not leaked, f"{len(leaked)} feature patterns leaked across the train/held-out split — the group split is broken"

    return df.iloc[train_idx], df.iloc[test_idx]


def main():
    config = load_config()
    train_cfg = config["training"]

    oof_df = pd.read_parquet(TRAINING_DIR / config["teacher_oof"]["path"])
    feature_columns = [c for c in oof_df.columns if c not in ("label", "oof_attack_probability")]

    train_df, test_df = group_split(oof_df, feature_columns, train_cfg["n_splits"], train_cfg["random_state"])
    x_train, categories = build_features(train_df, feature_columns)
    x_test, _ = build_features(test_df, feature_columns, categories=categories)

    model = XGBRegressor(
        n_estimators=train_cfg["n_estimators"],
        max_depth=train_cfg["max_depth"],
        learning_rate=train_cfg["learning_rate"],
        subsample=train_cfg["subsample"],
        colsample_bytree=train_cfg["colsample_bytree"],
        objective="reg:logistic",
        random_state=train_cfg["random_state"],
    )
    model.fit(x_train, train_df["oof_attack_probability"])

    teacher_proba_test = test_df["oof_attack_probability"].to_numpy()
    student_proba_test = model.predict(x_test)
    y_test = test_df["label"].to_numpy()

    fidelity_rmse = math.sqrt(mean_squared_error(teacher_proba_test, student_proba_test))
    bucket_agreement = float(np.mean(bucket_verdict(student_proba_test) == bucket_verdict(teacher_proba_test)))

    def label_metrics(proba):
        pred = (proba >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }

    teacher_metrics = label_metrics(teacher_proba_test)
    student_metrics = label_metrics(student_proba_test)

    normal_df = train_df[train_df["label"] == 0]
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]
    numeric_defaults = {c: float(normal_df[c].median()) for c in numeric_columns}
    categorical_defaults = {c: normal_df[c].astype(str).mode().iloc[0] for c in CATEGORICAL_COLUMNS}

    models_dir = TRAINING_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out = TRAINING_DIR / config["paths"]["model_out"]
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

    metrics_out = TRAINING_DIR / config["paths"]["metrics_out"]
    metrics_out.write_text(
        "# Student model — distilled from the teacher's out-of-fold predictions\n\n"
        f"`XGBRegressor(objective=\"reg:logistic\", n_estimators={train_cfg['n_estimators']}, "
        f"max_depth={train_cfg['max_depth']})` — {train_cfg['n_estimators']} trees vs. the teacher's "
        f"5x{500}, trained to regress against `teacher-xgboost`'s leak-free out-of-fold attack "
        f"probability (not the true label directly — this is distillation, not a second from-scratch "
        f"classifier). Held out via the same `StratifiedGroupKFold` grouping as the teacher, so this "
        f"evaluation is on feature-patterns the student's own training never saw.\n\n"
        "## Distillation fidelity (student vs. teacher, same held-out rows)\n\n"
        f"- RMSE (student probability vs. teacher OOF probability): {fidelity_rmse:.4f}\n"
        f"- Verdict-bucket agreement (allow/uncertain/block, student vs. teacher): {bucket_agreement:.4f}\n\n"
        "## Real-world correctness (both models vs. true label, same held-out rows)\n\n"
        "| | Teacher (OOF) | Student (distilled) |\n"
        "|---|---|---|\n"
        f"| Accuracy | {teacher_metrics['accuracy']:.4f} | {student_metrics['accuracy']:.4f} |\n"
        f"| Precision | {teacher_metrics['precision']:.4f} | {student_metrics['precision']:.4f} |\n"
        f"| Recall | {teacher_metrics['recall']:.4f} | {student_metrics['recall']:.4f} |\n"
        f"| F1 | {teacher_metrics['f1']:.4f} | {student_metrics['f1']:.4f} |\n\n"
        "**Train/production-drift caveat, per `knowledge-graph/noxos-inference/TASKS.md`**: the teacher "
        "numbers above come from 5 separate fold-models' OOF predictions, not a single full-data "
        "production teacher — don't assume they're interchangeable artifacts once a real production "
        "teacher is trained and deployed.\n"
    )

    print(json.dumps({
        "fidelity_rmse": fidelity_rmse,
        "bucket_agreement": bucket_agreement,
        "teacher": teacher_metrics,
        "student": student_metrics,
        "rows": len(oof_df),
    }, indent=2))


if __name__ == "__main__":
    main()
