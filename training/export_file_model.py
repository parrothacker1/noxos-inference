import json
import math

import joblib

from config import TRAINING_DIR, load_config


def transform_node(node: dict) -> dict:
    if "leaf" in node:
        return {"leaf": node["leaf"], "cover": node["cover"]}

    children_by_id = {child["nodeid"]: child for child in node["children"]}
    return {
        "feature": node["split"],
        "threshold": node["split_condition"],
        "default_left": node["missing"] == node["yes"],
        "cover": node["cover"],
        "left": transform_node(children_by_id[node["yes"]]),
        "right": transform_node(children_by_id[node["no"]]),
    }


def export(model_bundle: dict) -> dict:
    booster = model_bundle["model"].get_booster()
    trees = [transform_node(json.loads(t)) for t in booster.get_dump(dump_format="json", with_stats=True)]

    cfg = json.loads(booster.save_config())
    base_score_prob = float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base_score_margin = math.log(base_score_prob / (1 - base_score_prob))

    return {
        "base_score": base_score_margin,
        "feature_defaults": model_bundle["numeric_defaults"],
        "feature_scale": model_bundle["numeric_scale"],
        "trees": trees,
        "categories": model_bundle["categories"],
    }


def main():
    config = load_config()
    bundle = joblib.load(TRAINING_DIR / config["paths"]["model_out"])
    exported = export(bundle)
    out = TRAINING_DIR / config["paths"]["ondevice_out"]
    out.write_text(json.dumps(exported))
    print(f"wrote {out} ({len(exported['trees'])} trees, {len(exported['feature_defaults'])} features)")


if __name__ == "__main__":
    main()
