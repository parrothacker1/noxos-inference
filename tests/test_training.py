import re
import unittest
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "service" / "model_network.joblib"
METRICS_PATH = REPO_ROOT / "training" / "METRICS.md"


class TrainingArtifactTest(unittest.TestCase):
    def setUp(self):
        if not MODEL_PATH.exists():
            self.skipTest("model_network.joblib not built yet — run training/train_network_model.py first")
        self.bundle = joblib.load(MODEL_PATH)

    def test_artifact_has_expected_keys(self):
        for key in ("model", "feature_columns", "categorical_columns", "categories", "numeric_defaults", "categorical_defaults"):
            self.assertIn(key, self.bundle)

    def test_defaults_only_row_predicts_a_valid_probability(self):
        import pandas as pd

        bundle = self.bundle
        row = {}
        for col in bundle["feature_columns"]:
            if col in bundle["categorical_columns"]:
                value = bundle["categorical_defaults"][col]
                categories = bundle["categories"][col]
                row[col] = categories.index(value) if value in categories else -1
            else:
                row[col] = bundle["numeric_defaults"][col]
        frame = pd.DataFrame([row], columns=bundle["feature_columns"])

        proba = bundle["model"].predict_proba(frame)[0][1]
        self.assertGreaterEqual(proba, 0.0)
        self.assertLessEqual(proba, 1.0)
        self.assertLess(proba, 0.4, "an all-normal-defaults request should read as low attack probability")

    def test_held_out_metrics_meet_a_sane_floor(self):
        if not METRICS_PATH.exists():
            self.skipTest("METRICS.md not written yet")
        text = METRICS_PATH.read_text()
        accuracy = float(re.search(r"Accuracy:\s*([0-9.]+)", text).group(1))
        f1 = float(re.search(r"F1.*?:\s*([0-9.]+)", text).group(1))
        self.assertGreater(accuracy, 0.85)
        self.assertGreater(f1, 0.85)


if __name__ == "__main__":
    unittest.main()
