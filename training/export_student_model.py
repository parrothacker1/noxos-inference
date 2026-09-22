import json
import math

import joblib

from config import REPO_ROOT, TRAINING_DIR, load_config


def transform_node(node: dict) -> dict:
    if "leaf" in node:
        return {"leaf": node["leaf"]}

    children_by_id = {child["nodeid"]: child for child in node["children"]}
    return {
        "feature": node["split"],
        "threshold": node["split_condition"],
        "default_left": node["missing"] == node["yes"],
        "left": transform_node(children_by_id[node["yes"]]),
        "right": transform_node(children_by_id[node["no"]]),
    }


def export(model_bundle: dict) -> dict:
    booster = model_bundle["model"].get_booster()
    trees = [transform_node(json.loads(t)) for t in booster.get_dump(dump_format="json", with_stats=False)]

    cfg = json.loads(booster.save_config())
    base_score_prob = float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base_score_margin = math.log(base_score_prob / (1 - base_score_prob))

    feature_defaults = dict(model_bundle["numeric_defaults"])
    categories = model_bundle["categories"]
    for col, default_value in model_bundle["categorical_defaults"].items():
        feature_defaults[col] = float(categories[col].index(default_value))

    return {
        "base_score": base_score_margin,
        "feature_defaults": feature_defaults,
        "trees": trees,
        "categories": categories,
    }


def main():
    config = load_config()
    model_bundle = joblib.load(TRAINING_DIR / config["paths"]["model_out"])
    exported = export(model_bundle)

    out_path = TRAINING_DIR / config["paths"]["ondevice_out"]
    out_path.write_text(json.dumps(exported))
    print(f"wrote {out_path} ({len(exported['trees'])} trees, {len(exported['feature_defaults'])} features)")


if __name__ == "__main__":
    main()
