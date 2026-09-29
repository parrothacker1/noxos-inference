import json
import math

import joblib
import numpy as np

from config import TRAINING_DIR, load_config
from data import load_drebin
from export_file_model import export


def eval_tree(node: dict, features: dict) -> float:
    if "leaf" in node:
        return node["leaf"]
    value = features.get(node["feature"])
    go_left = node["default_left"] if value is None else np.float32(value) < np.float32(node["threshold"])
    return eval_tree(node["left"] if go_left else node["right"], features)


def predict_from_export(exported: dict, features: dict) -> float:
    full = {**exported["feature_defaults"], **features}
    margin = exported["base_score"] + sum(eval_tree(t, full) for t in exported["trees"])
    return 1.0 / (1.0 + math.exp(-margin))


def main():
    config = load_config()
    bundle = joblib.load(TRAINING_DIR / config["paths"]["model_out"])
    exported = export(bundle)
    (TRAINING_DIR / config["paths"]["ondevice_out"]).write_text(json.dumps(exported))

    features, _, _ = load_drebin(config["dataset"])
    sample = features[bundle["feature_columns"]].sample(n=200, random_state=7)
    real = bundle["model"].predict_proba(sample)[:, 1]
    got = np.array([predict_from_export(exported, row.to_dict()) for _, row in sample.iterrows()])

    max_diff = float(np.max(np.abs(real - got)))
    bucket = lambda p: np.where(p < 0.4, 0, np.where(p > 0.6, 2, 1))
    agree = float(np.mean(bucket(real) == bucket(got)))
    print(json.dumps({"max_diff": max_diff, "bucket_agreement": agree}))
    assert max_diff < 1e-4, f"exported trees diverge from the real model by {max_diff}"
    assert agree == 1.0
    print(f"OK — exported JSON matches model.predict_proba() within {max_diff:.8f} on 200 real rows")


if __name__ == "__main__":
    main()
