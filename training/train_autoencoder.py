import json
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from config import REPO_ROOT, TRAINING_DIR, load_config
from dataset import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, build_dataset, load_raw
from encoding import encode, fit_categories, numeric_stats
from model import Autoencoder, reconstruction_error, split_targets


def group_split(df, n_splits: int, random_state: int):
    groups = df.groupby(FEATURE_COLUMNS, sort=False).ngroup()
    print(f"{groups.nunique()} distinct feature patterns across {len(df)} rows", file=sys.stderr)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    leaked = train_groups & test_groups
    assert not leaked, f"{len(leaked)} feature patterns leaked across the train/held-out split — the group split is broken"

    return df.iloc[train_idx], df.iloc[test_idx]


@torch.no_grad()
def score(model, x: torch.Tensor) -> np.ndarray:
    model.eval()
    numeric_recon, logits = model(x)
    numeric_true, idx = split_targets(x, model.numeric_dim, model.categorical_dims)
    return reconstruction_error(numeric_recon, numeric_true, logits, idx).numpy()


def sweep_thresholds(errors: np.ndarray, y: np.ndarray, low: int, high: int) -> list[dict]:
    normal_errors = errors[y == 0]
    rows = []
    for p in range(low, high + 1):
        threshold = float(np.percentile(normal_errors, p))
        flagged = (errors > threshold).astype(int)
        rows.append({
            "percentile": p,
            "threshold": threshold,
            "precision": float(precision_score(y, flagged, zero_division=0)),
            "recall": float(recall_score(y, flagged, zero_division=0)),
            "f1": float(f1_score(y, flagged, zero_division=0)),
            "accuracy": float(accuracy_score(y, flagged)),
            "flagged_rate": float(flagged.mean()),
        })
    return rows


def main():
    config = load_config()
    raw_dir = REPO_ROOT / config["dataset"]["raw_dir"]
    train_cfg = config["training"]
    models_dir = TRAINING_DIR / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw(raw_dir, config["dataset"]["urls"])
    dataset = build_dataset(raw_df).reset_index(drop=True)

    train_df, test_df = group_split(dataset, train_cfg["n_splits"], train_cfg["random_state"])
    normal_train_df = train_df[train_df["label"] == 0].reset_index(drop=True)
    print(
        f"train: {len(train_df)} rows ({len(normal_train_df)} normal, used for fitting); "
        f"held-out: {len(test_df)} rows ({int((test_df['label'] == 0).sum())} normal / "
        f"{int((test_df['label'] == 1).sum())} attack)",
        file=sys.stderr,
    )

    categories = fit_categories(normal_train_df, train_cfg["categorical_top_n"])
    categorical_dims = [len(categories[col]) for col in CATEGORICAL_COLUMNS]
    mean, std = numeric_stats(normal_train_df)

    x_train = torch.from_numpy(encode(normal_train_df, mean, std, categories))

    torch.manual_seed(train_cfg["random_state"])
    model = Autoencoder(
        numeric_dim=len(NUMERIC_COLUMNS),
        categorical_dims=categorical_dims,
        hidden_dim=train_cfg["hidden_dim"],
        bottleneck_dim=train_cfg["bottleneck_dim"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    loader = DataLoader(TensorDataset(x_train), batch_size=train_cfg["batch_size"], shuffle=True)

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for (x_batch,) in loader:
            numeric_recon, logits = model(x_batch)
            numeric_true, idx = split_targets(x_batch, model.numeric_dim, model.categorical_dims)
            loss = reconstruction_error(numeric_recon, numeric_true, logits, idx).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        print(f"epoch {epoch + 1}/{train_cfg['epochs']}: mean training loss {epoch_loss / len(x_train):.4f}", file=sys.stderr)

    x_test = torch.from_numpy(encode(test_df, mean, std, categories))
    test_errors = score(model, x_test)
    y_test = test_df["label"].to_numpy()

    sweep = sweep_thresholds(test_errors, y_test, train_cfg["threshold_percentile_min"], train_cfg["threshold_percentile_max"])
    best = max(sweep, key=lambda r: r["f1"])
    normal_test_errors = test_errors[y_test == 0]
    attack_test_errors = test_errors[y_test == 1]

    torch.save(
        {
            "state_dict": model.state_dict(),
            "numeric_dim": len(NUMERIC_COLUMNS),
            "categorical_dims": categorical_dims,
            "hidden_dim": train_cfg["hidden_dim"],
            "bottleneck_dim": train_cfg["bottleneck_dim"],
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "categories": categories,
            "numeric_mean": mean,
            "numeric_std": std,
            "reconstruction_threshold": best["threshold"],
        },
        TRAINING_DIR / config["paths"]["model_out"],
    )

    sweep_lines = "\n".join(
        f"| {r['percentile']} | {r['threshold']:.5f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['flagged_rate']:.4f} |"
        for r in sweep
        if r["percentile"] % 2 == 0 or r["percentile"] == best["percentile"]
    )
    category_lines = "\n".join(f"  - `{col}`: {', '.join(categories[col])}" for col in CATEGORICAL_COLUMNS)

    metrics_out = TRAINING_DIR / config["paths"]["metrics_out"]
    metrics_out.write_text(
        "# Autoencoder — unsupervised anomaly-detection tier (Tier 1)\n\n"
        f"Trained on normal traffic only ({len(normal_train_df)} rows, `label == 0`), never sees an attack label. "
        f"One shared encoder/bottleneck; decoder heads: linear/MSE over the {len(NUMERIC_COLUMNS)} z-score-standardized "
        f"numeric features, plus one softmax/cross-entropy head per categorical feature ({', '.join(CATEGORICAL_COLUMNS)}). "
        f"Architecture: input({len(NUMERIC_COLUMNS) + sum(categorical_dims)}) -> hidden({train_cfg['hidden_dim']}) -> "
        f"bottleneck({train_cfg['bottleneck_dim']}) -> hidden({train_cfg['hidden_dim']}) -> "
        f"[numeric({len(NUMERIC_COLUMNS)}), {', '.join(f'{c}({d})' for c, d in zip(CATEGORICAL_COLUMNS, categorical_dims))}], "
        f"LeakyReLU, {train_cfg['epochs']} epochs, Adam lr={train_cfg['learning_rate']}.\n\n"
        "Reconstruction error per row = **mean** over numeric features of squared error + sum over categorical heads of "
        "per-row cross-entropy (unweighted).\n\n"
        "## Encoding\n\n"
        "- `dst_port`: `log1p` before standardization.\n"
        "- Categorical features bucketed to the top "
        f"{train_cfg['categorical_top_n']} values by frequency in the normal-only training split plus `other`:\n"
        f"{category_lines}\n"
        "- `dttl` (destination TTL) and `state` (connection state) were added after a feature sweep — the original "
        "10 features capped F1 at ~0.75 (see `knowledge-graph/noxos-inference/TASKS.md`).\n\n"
        "## Held-out evaluation (`StratifiedGroupKFold`, groups = exact feature pattern over all model inputs)\n\n"
        f"- Held-out rows: {len(test_df)} ({int((y_test == 0).sum())} normal, {int((y_test == 1).sum())} attack)\n"
        f"- Mean reconstruction error: normal {normal_test_errors.mean():.4f} (median {np.median(normal_test_errors):.4f}), "
        f"attack {attack_test_errors.mean():.4f} (median {np.median(attack_test_errors):.4f})\n"
        f"- Chosen threshold: {best['percentile']}th percentile of normal held-out error = {best['threshold']:.5f} "
        "(F1-maximizing point of the sweep below; note it is selected on the same held-out set it is reported on, so "
        "treat the F1 as slightly optimistic)\n\n"
        "## At the chosen threshold\n\n"
        f"- Accuracy: {best['accuracy']:.4f}\n"
        f"- Precision: {best['precision']:.4f}\n"
        f"- Recall: {best['recall']:.4f}\n"
        f"- F1: {best['f1']:.4f}\n"
        f"- Flagged rate: {best['flagged_rate']:.4f}\n\n"
        "## Threshold sweep (even percentiles + the chosen one)\n\n"
        "| percentile | threshold | precision | recall | F1 | flagged rate |\n|---|---|---|---|---|---|\n"
        f"{sweep_lines}\n\n"
        "Not a like-for-like comparison with the teacher/student models: this is an unsupervised anomaly detector "
        "scored against labels it never trained on.\n"
    )

    print(json.dumps({
        "chosen": best,
        "normal_error_mean": float(normal_test_errors.mean()),
        "attack_error_mean": float(attack_test_errors.mean()),
        "rows_total": len(dataset),
        "rows_train_normal": len(normal_train_df),
        "rows_held_out": len(test_df),
    }, indent=2))


if __name__ == "__main__":
    main()
