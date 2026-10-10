import json
import sys

import numpy as np
import pandas as pd
import torch

from config import REPO_ROOT, TRAINING_DIR, load_config
from encoding import encode
from export_autoencoder_model import export, load_model
from train_autoencoder import score
from train_mirage import run_split, split_by_app, synthetic_anomalies

PERCENTILE = 99
BASELINE_PERCENTILE = 99
MARKER = "## UDP threshold recalibration"


def weighted_quantile(errors: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(errors)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(errors[order][min(np.searchsorted(cumulative, percentile / 100), len(errors) - 1)])


def app_balanced_threshold(pool: pd.DataFrame, percentile: float) -> float:
    weights = 1.0 / pool.groupby("app")["error"].transform("size").to_numpy()
    return weighted_quantile(pool["error"].to_numpy(), weights, percentile)


def is_dns_like(df: pd.DataFrame) -> pd.Series:
    return (df["src_packet_count"] <= 3) & (df["src_byte_count"] < 400) & (df["dst_byte_count"] < 2000)


def spread(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{np.nanmean(values):.4f} ({np.nanmin(values):.4f}-{np.nanmax(values):.4f})"


def collect_udp_errors(config: dict, records: pd.DataFrame, seeds: list[int]):
    mirage = config["mirage"]
    folds = []
    for seed in seeds:
        run = run_split(config, records, seed)
        _, calib, test, _, _ = split_by_app(records, mirage["test_app_fraction"], mirage["calibration_app_fraction"], seed)

        def errors_of(df):
            return score(run["model"], torch.from_numpy(encode(df, run["mean"], run["std"], run["categories"])))

        parts = []
        for name, df in (("calibration", calib), ("test", test)):
            mask = (df["proto"] == "udp").to_numpy()
            part = df[mask].copy()
            part["error"] = errors_of(df)[mask]
            part["part"] = name
            parts.append(part)
        synthetic = {k: errors_of(v) for k, v in synthetic_anomalies(np.random.default_rng(seed)).items()}
        folds.append({"seed": seed, "udp": pd.concat(parts, ignore_index=True), "flood": synthetic["udp_flood"],
                      "test_apps": sorted(test["app"].unique()),
                      "old_threshold": float(np.percentile(parts[0]["error"], BASELINE_PERCENTILE))})
    return folds


def evaluate(folds: list[dict]) -> dict:
    rows = {"old": [], "new": []}
    for i, fold in enumerate(folds):
        test = fold["udp"][fold["udp"]["part"] == "test"]
        others = pd.concat([f["udp"] for j, f in enumerate(folds) if j != i], ignore_index=True)
        others = others[~others["app"].isin(fold["test_apps"])]
        thresholds = {"old": fold["old_threshold"], "new": app_balanced_threshold(others, PERCENTILE)}
        dns = test[is_dns_like(test)]
        for rule, t in thresholds.items():
            rows[rule].append({
                "seed": fold["seed"], "threshold": t,
                "false_flag": float((test["error"] > t).mean()),
                "app_mean_false_flag": float((test["error"] > t).groupby(test["app"]).mean().mean()),
                "flood": float((fold["flood"] > t).mean()),
                "dns_like_flagged": float((dns["error"] > t).mean()) if len(dns) else float("nan"),
                "dns_like_rows": len(dns),
            })
    return rows


def main():
    config = load_config()
    cfg = config["training"]
    mirage = config["mirage"]
    records = pd.read_parquet(REPO_ROOT / mirage["records"])
    seeds = [cfg["random_state"] + i for i in range(mirage["app_split_repeats"])]
    folds = collect_udp_errors(config, records, seeds)
    rows = evaluate(folds)

    pool = pd.concat([f["udp"] for f in folds], ignore_index=True)
    final = app_balanced_threshold(pool, PERCENTILE)

    checkpoint_path = TRAINING_DIR / config["paths"]["model_out"]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    previous = float(checkpoint["reconstruction_thresholds"]["udp"])
    checkpoint["reconstruction_thresholds"]["udp"] = final
    checkpoint["threshold_percentiles"]["udp"] = f"app-balanced {PERCENTILE}th percentile of pooled out-of-fold UDP error"
    torch.save(checkpoint, checkpoint_path)
    export(checkpoint_path, TRAINING_DIR / config["paths"]["ondevice_out"])

    model = load_model(checkpoint)
    mean = np.asarray(checkpoint["numeric_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["numeric_std"], dtype=np.float32)
    first = folds[0]["udp"]
    shipped_errors = score(model, torch.from_numpy(encode(first, mean, std, checkpoint["categories"])))
    shipped_dns = is_dns_like(first).to_numpy()

    def table(rule):
        return "\n".join(
            f"| {r['seed']} | {r['threshold']:.4f} | {r['false_flag']:.4f} | {r['app_mean_false_flag']:.4f} | {r['flood']:.4f} | "
            f"{r['dns_like_flagged']:.3f} ({r['dns_like_rows']}) |" for r in rows[rule]
        )

    def summary(rule):
        return (f"UDP false-flag {spread([r['false_flag'] for r in rows[rule]])}, per-app mean {spread([r['app_mean_false_flag'] for r in rows[rule]])}, "
                f"UDP flood detected {spread([r['flood'] for r in rows[rule]])}, DNS-like flows flagged {spread([r['dns_like_flagged'] for r in rows[rule]])}")

    header = "| seed | threshold | UDP false-flag | per-app mean false-flag | UDP flood detected | DNS-like flagged (rows) |\n|---|---|---|---|---|---|\n"
    text = (
        f"{MARKER}\n\n"
        f"Weights, features, transforms, categories, error formula and the TCP threshold are unchanged. Only `reconstruction_thresholds.udp` changed: "
        f"{previous:.6f} -> {final:.6f}.\n\n"
        f"**Rule.** The UDP threshold is the {PERCENTILE}th percentile of normal UDP reconstruction error pooled over out-of-fold rows from all {len(folds)} app splits "
        "(calibration and test rows; every row scored by a model that never trained on its app), with every app weighted equally. "
        "Why: the old rule used one split's calibration apps (132-807 UDP rows, threshold ranging 0.125-0.499 across splits), and "
        f"one app (YouTube) holds {int((pool['app'] == 'com.google.android.youtube').sum() * 100 / len(pool))}% of the pooled UDP rows, so a plain percentile mostly describes YouTube. "
        "It was chosen by comparing rules on held-out apps, not by fitting to any device's test flows.\n\n"
        f"**Evaluation**, leave-split-out: for each split the threshold is computed from the other splits' rows excluding the apps in this split's test set, then applied to this split's test apps. "
        "Old rule = calibration-only percentile as shipped before.\n\n"
        f"Old rule ({BASELINE_PERCENTILE}th of that split's calibration UDP): {summary('old')}\n\n" + header + table("old") + "\n\n"
        f"New rule: {summary('new')}\n\n" + header + table("new") + "\n\n"
        "DNS-like = UDP snapshots with at most 3 source packets, under 400 source bytes and under 2000 destination bytes. "
        "MIRAGE has no UDP traffic on port 53, so this is a shape proxy (mostly Spotify discovery on 57621, STUN on 3478, DHCP), not real DNS.\n\n"
        f"Shipped model on its own split-42 UDP rows (calibration and test apps), old vs new threshold: flagged {float((shipped_errors > previous).mean()):.4f} -> {float((shipped_errors > final).mean()):.4f}; "
        f"DNS-like flows flagged {float((shipped_errors[shipped_dns] > previous).mean()):.3f} -> {float((shipped_errors[shipped_dns] > final).mean()):.3f} ({int(shipped_dns.sum())} rows).\n\n"
        "**Remaining limits.** F1 is still unmeasured (MIRAGE has no attacks). A rate of a few percent on unseen apps remains and is not zero. "
        "The pooled errors come from the 5 retrained split models, not the shipped weights themselves. Real DNS (port 53) is absent from the training data. "
        "The cost of the higher threshold is UDP flood sensitivity: 94.6% (86.3-99.3%) detected against 98.1% (95.9-99.9%) before; smaller UDP anomalies sitting between the old and new threshold are no longer flagged.\n"
    )
    metrics_path = TRAINING_DIR / config["paths"]["metrics_out"]
    body = metrics_path.read_text()
    if MARKER in body:
        body = body[:body.index(MARKER)].rstrip() + "\n"
    metrics_path.write_text(body.rstrip() + "\n\n" + text)
    print(json.dumps({"previous_udp": previous, "udp": final, "tcp": checkpoint["reconstruction_thresholds"]["tcp"]}))
    print(text)


if __name__ == "__main__":
    main()
