import json
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

ONDEVICE_JSON = REPO_ROOT / "service" / "model_ondevice.json"
MODEL_PATH = REPO_ROOT / "service" / "model_network.joblib"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def eval_tree(node: dict, features: dict[str, float]) -> float:
    if "leaf" in node:
        return node["leaf"]
    value = features.get(node["feature"])
    if value is None:
        go_left = node["default_left"]
    else:
        go_left = value < node["threshold"]
    return eval_tree(node["left"] if go_left else node["right"], features)


def predict_from_export(exported: dict, features: dict[str, float]) -> float:
    full_features = dict(exported["feature_defaults"])
    full_features.update(features)
    margin = exported["base_score"] + sum(eval_tree(t, full_features) for t in exported["trees"])
    return sigmoid(margin)


def raw_dataset_ready() -> bool:
    from config import load_config

    cfg = load_config()
    raw_dir = REPO_ROOT / cfg["dataset"]["raw_dir"]
    return all((raw_dir / url.rsplit("/", 1)[-1]).exists() for url in cfg["dataset"]["urls"])


@unittest.skipUnless(ONDEVICE_JSON.exists() and MODEL_PATH.exists() and raw_dataset_ready(), "export/model/dataset not built yet")
class ExportMatchesModelTest(unittest.TestCase):
    def setUp(self):
        import joblib

        from config import load_config
        from train_network_model import build_dataset, load_raw

        self.exported = json.loads(ONDEVICE_JSON.read_text())
        self.bundle = joblib.load(MODEL_PATH)

        cfg = load_config()
        raw_dir = REPO_ROOT / cfg["dataset"]["raw_dir"]
        raw_df = load_raw(raw_dir, cfg["dataset"]["urls"])
        self.dataset = build_dataset(raw_df)

    def test_exported_json_has_the_locked_contract_shape(self):
        self.assertIn("base_score", self.exported)
        self.assertIn("feature_defaults", self.exported)
        self.assertIn("trees", self.exported)
        self.assertIsInstance(self.exported["trees"], list)
        self.assertGreater(len(self.exported["trees"]), 0)

    def test_reimplemented_tree_eval_closely_matches_real_predict_proba(self):
        from train_network_model import build_features

        bundle = self.bundle
        sample = self.dataset.sample(30, random_state=7)
        x, _ = build_features(sample, bundle["feature_columns"], categories=bundle["categories"])
        real_probas = bundle["model"].predict_proba(x)[:, 1]

        max_diff = 0.0
        bucket_agreements = 0
        for i, (_, row) in enumerate(x.iterrows()):
            features = {k: float(v) for k, v in row.items()}
            exported_proba = predict_from_export(self.exported, features)
            diff = abs(exported_proba - real_probas[i])
            max_diff = max(max_diff, diff)

            def bucket(p):
                return "allow" if p < 0.4 else "block" if p > 0.6 else "uncertain"

            if bucket(exported_proba) == bucket(real_probas[i]):
                bucket_agreements += 1

        self.assertLess(max_diff, 0.2, "exported-JSON prediction drifted too far from the real model")
        self.assertGreaterEqual(bucket_agreements, 28, "verdict bucket disagreed on more than 2/30 real samples")


if __name__ == "__main__":
    unittest.main()
