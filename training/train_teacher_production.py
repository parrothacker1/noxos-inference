import json
import sys

import joblib
from xgboost import XGBClassifier

from config import REPO_ROOT, TRAINING_DIR, load_config
from train_teacher import CATEGORICAL_COLUMNS, build_dataset, build_features, load_raw


def main():
    config = load_config()
    raw_dir = REPO_ROOT / config["dataset"]["raw_dir"]
    train_cfg = config["training"]

    raw_df = load_raw(raw_dir, config["dataset"]["urls"])
    dataset = build_dataset(raw_df).reset_index(drop=True)
    feature_columns = [c for c in dataset.columns if c != "label"]

    x, categories = build_features(dataset, feature_columns)

    model = XGBClassifier(
        n_estimators=train_cfg["n_estimators"],
        max_depth=train_cfg["max_depth"],
        learning_rate=train_cfg["learning_rate"],
        subsample=train_cfg["subsample"],
        colsample_bytree=train_cfg["colsample_bytree"],
        eval_metric="logloss",
        random_state=train_cfg["random_state"],
    )
    model.fit(x, dataset["label"])

    normal_df = dataset[dataset["label"] == 0]
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]
    numeric_defaults = {c: float(normal_df[c].median()) for c in numeric_columns}
    categorical_defaults = {c: normal_df[c].astype(str).mode().iloc[0] for c in CATEGORICAL_COLUMNS}

    models_dir = TRAINING_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "categories": categories,
            "numeric_defaults": numeric_defaults,
            "categorical_defaults": categorical_defaults,
        },
        models_dir / "teacher_production.joblib",
    )
    print(json.dumps({"rows": len(dataset), "trees": train_cfg["n_estimators"]}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
