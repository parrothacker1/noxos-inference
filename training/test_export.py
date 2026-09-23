import json
import math

import joblib
import numpy as np

from config import REPO_ROOT, TRAINING_DIR, load_config
from train_teacher import build_dataset, build_features, load_raw


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def eval_tree(node: dict, features: dict[str, float]) -> float:
    if "leaf" in node:
        return node["leaf"]
    value = features.get(node["feature"])
    if value is None:
        go_left = node["default_left"]
    else:
        go_left = np.float32(value) < np.float32(node["threshold"])
    return eval_tree(node["left"] if go_left else node["right"], features)


def predict_from_export(exported: dict, features: dict[str, float]) -> float:
    full_features = dict(exported["feature_defaults"])
    full_features.update(features)
    margin = exported["base_score"] + sum(eval_tree(t, full_features) for t in exported["trees"])
    return sigmoid(margin)


def main():
    config = load_config()
    exported = json.loads((TRAINING_DIR / "models" / "teacher_ondevice.json").read_text())
    bundle = joblib.load(TRAINING_DIR / "models" / "teacher_production.joblib")

    raw_dir = REPO_ROOT / config["dataset"]["raw_dir"]
    raw_df = load_raw(raw_dir, config["dataset"]["urls"])
    dataset = build_dataset(raw_df)
    sample = dataset.sample(30, random_state=7)
    x, _ = build_features(sample, bundle["feature_columns"], categories=bundle["categories"])
    real_probas = bundle["model"].predict_proba(x)[:, 1]

    max_diff = 0.0
    for i, (_, row) in enumerate(x.iterrows()):
        features = {k: float(v) for k, v in row.items()}
        exported_proba = predict_from_export(exported, features)
        max_diff = max(max_diff, abs(exported_proba - real_probas[i]))

    assert max_diff < 0.001, f"exported-JSON prediction drifted from the real model: max_diff={max_diff}"
    print(f"OK: exported JSON matches real model.predict_proba() within {max_diff:.6f} on 30 real sampled rows")


if __name__ == "__main__":
    main()
