import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from config import TRAINING_DIR, load_config
from data import CATEGORY_INTENT, CATEGORY_PERMISSION, load_drebin

HIGH_RISK_PERMISSION_NAMES = [
    "SEND_SMS", "RECEIVE_SMS", "READ_SMS", "RECEIVE_MMS", "READ_CALL_LOG", "PROCESS_OUTGOING_CALLS",
    "CALL_PHONE", "READ_CONTACTS", "RECORD_AUDIO", "SYSTEM_ALERT_WINDOW", "REQUEST_INSTALL_PACKAGES",
    "BIND_ACCESSIBILITY_SERVICE", "BIND_DEVICE_ADMIN", "BIND_NOTIFICATION_LISTENER_SERVICE",
]


def make_model(cfg: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        eval_metric="logloss",
        random_state=cfg["random_state"],
    )


def metrics(y, proba=None, pred=None) -> dict:
    if pred is None:
        pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }
    if proba is not None:
        out["auc"] = roc_auc_score(y, proba)
    return {k: float(v) for k, v in out.items()}


def grouped_oof(features: pd.DataFrame, y: pd.Series, groups: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[dict]]:
    splitter = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["random_state"])
    oof = np.full(len(features), np.nan)
    per_fold = []
    for fold, (tr, te) in enumerate(splitter.split(features, y, groups=groups)):
        assert not (set(groups[tr]) & set(groups[te])), f"fold {fold}: a feature pattern leaked across train/held-out"
        model = make_model(cfg)
        model.fit(features.iloc[tr], y.iloc[tr])
        oof[te] = model.predict_proba(features.iloc[te])[:, 1]
        per_fold.append(metrics(y.iloc[te].to_numpy(), oof[te]))
    assert not np.isnan(oof).any()
    return oof, per_fold


def main():
    config = load_config()
    cfg = config["training"]
    features, y, by_category = load_drebin(config["dataset"])

    permissions = by_category[CATEGORY_PERMISSION]
    intents = by_category[CATEGORY_INTENT]
    sets = {
        "permissions": permissions,
        "permissions+intents": permissions + intents,
        "all_215 (not deployable: API calls need DEX analysis)": list(features.columns),
    }

    groups = features.groupby(list(features.columns), sort=False).ngroup().to_numpy()
    conflict = pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].nunique()
    print(f"{len(features)} rows, {len(np.unique(groups))} distinct full feature patterns, "
          f"{int((conflict > 1).sum())} patterns carry both labels", file=sys.stderr)

    results = {}
    for name, cols in sets.items():
        pattern_groups = features[cols].groupby(list(cols), sort=False).ngroup().to_numpy()
        oof, per_fold = grouped_oof(features[cols], y, pattern_groups, cfg)
        results[name] = {
            "columns": len(cols),
            "distinct_patterns": int(len(np.unique(pattern_groups))),
            "overall": metrics(y.to_numpy(), oof),
            "folds_f1": [round(f["f1"], 4) for f in per_fold],
            "folds_f1_std": float(np.std([f["f1"] for f in per_fold])),
        }
        print(name, json.dumps(results[name]["overall"]), file=sys.stderr)

    available = [p for p in HIGH_RISK_PERMISSION_NAMES if p in permissions]
    high_risk_count = features[available].sum(axis=1)
    rule = (high_risk_count >= 4).astype(int)
    results["baseline: payload-style rule (>=4 high-risk permissions)"] = {
        "columns": len(available),
        "high_risk_names_present_in_drebin": available,
        "overall": metrics(y.to_numpy(), pred=rule.to_numpy()),
    }

    deployed = cfg["deployed_feature_set"]
    deployed_cols = sets[deployed]
    model = make_model(cfg)
    model.fit(features[deployed_cols], y)
    importance = pd.Series(model.feature_importances_, index=deployed_cols).sort_values(ascending=False)

    normal = features[y == 0]
    (TRAINING_DIR / config["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": deployed_cols,
            "categorical_columns": [],
            "categories": {},
            "numeric_defaults": {c: 0.0 for c in deployed_cols},
            "numeric_scale": {c: float(normal[c].std()) or 1.0 for c in deployed_cols},
            "categorical_defaults": {},
        },
        TRAINING_DIR / config["paths"]["model_out"],
    )

    write_metrics(config, results, importance, deployed, len(features), int(y.sum()), len(np.unique(groups)))
    print(json.dumps({k: v["overall"] for k, v in results.items()}, indent=2))


def write_metrics(config, results, importance, deployed, rows, malware, distinct):
    cfg = config["training"]
    lines = []
    for name, r in results.items():
        o = r["overall"]
        auc = f"{o['auc']:.4f}" if "auc" in o else "n/a"
        lines.append(f"| {name} | {r['columns']} | {o['precision']:.4f} | {o['recall']:.4f} | {o['f1']:.4f} | {auc} |")
    top = "\n".join(f"- `{n}`: {v:.4f}" for n, v in importance.head(12).items())
    text = (
        "# File classifier (Drebin-215, APK permission features)\n\n"
        f"XGBoost ({cfg['n_estimators']} trees, depth {cfg['max_depth']}), trained on the public Drebin-215 feature CSV: "
        f"{rows} apps ({malware} malware / {rows - malware} benign), {distinct} distinct feature patterns. "
        "Evaluated with `StratifiedGroupKFold` (5-fold, out-of-fold over every row) grouped by exact feature pattern within each "
        "feature set, so no pattern is in both train and held-out.\n\n"
        "| feature set | columns | precision | recall | F1 | AUC |\n|---|---|---|---|---|---|\n"
        + "\n".join(lines)
        + f"\n\nDeployed feature set: `{deployed}` (the only one `noxos-payload` can extract today).\n\n"
        "## Most important features in the deployed model\n\n" + top + "\n\n"
        "Caveats: Drebin is a 2010-2012 era dataset, so a modern benign app's permission set may sit outside its distribution; "
        "the public CSV has 9,476 benign apps, not the 123K in the original paper. F1 here is on this dataset only, not on captured Warden traffic.\n"
    )
    (TRAINING_DIR / config["paths"]["metrics_out"]).write_text(text)


if __name__ == "__main__":
    main()
