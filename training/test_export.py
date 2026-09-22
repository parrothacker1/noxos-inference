import json
import math

import joblib
import numpy as np
import pandas as pd

from config import TRAINING_DIR, load_config
from train_student import build_features


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
    exported = json.loads((TRAINING_DIR / config["paths"]["ondevice_out"]).read_text())
    bundle = joblib.load(TRAINING_DIR / config["paths"]["model_out"])

    oof_df = pd.read_parquet(TRAINING_DIR / config["teacher_oof"]["path"])
    sample = oof_df.sample(30, random_state=7)
    x, _ = build_features(sample, bundle["feature_columns"], categories=bundle["categories"])
    real_preds = bundle["model"].predict(x)

    max_diff = 0.0
    for i, (_, row) in enumerate(x.iterrows()):
        features = {k: float(v) for k, v in row.items()}
        exported_pred = predict_from_export(exported, features)
        max_diff = max(max_diff, abs(exported_pred - real_preds[i]))

    assert max_diff < 0.001, f"exported-JSON prediction drifted from the real model: max_diff={max_diff}"
    print(f"OK: exported JSON matches real model.predict() within {max_diff:.6f} on 30 real sampled rows")


if __name__ == "__main__":
    main()
