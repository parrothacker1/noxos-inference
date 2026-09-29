import json
import sys

import numpy as np
import pandas as pd
import torch

from config import REPO_ROOT, TRAINING_DIR, load_config
from dataset import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_dataset, load_raw
from encoding import encode, fit_categories, numeric_stats
from train_autoencoder import fit, score

CITATION = (
    "Giuseppe Aceto, Domenico Ciuonzo, Antonio Montieri, Valerio Persico and Antonio Pescapè, "
    '"MIRAGE: Mobile-app Traffic Capture and Ground-truth Creation", 4th IEEE International Conference on '
    "Computing, Communications and Security (ICCCS 2019). Dataset licence: CC BY-NC-ND 4.0."
)


def split_by_app(df: pd.DataFrame, test_fraction: float, calibration_fraction: float, seed: int):
    apps = sorted(df["app"].unique())
    order = np.random.default_rng(seed).permutation(len(apps))
    n_test = max(1, round(len(apps) * test_fraction))
    n_calib = max(1, round(len(apps) * calibration_fraction))
    test_apps = {apps[i] for i in order[:n_test]}
    calib_apps = {apps[i] for i in order[n_test:n_test + n_calib]}
    is_test = df["app"].isin(test_apps)
    is_calib = df["app"].isin(calib_apps)
    return df[~is_test & ~is_calib], df[is_calib], df[is_test], sorted(calib_apps), sorted(test_apps)


def flag_rates(errors: np.ndarray, protos: pd.Series, threshold: float) -> dict:
    flagged = errors > threshold
    out = {"all": {"rows": int(len(errors)), "flagged": float(flagged.mean())}}
    for proto in ("tcp", "udp"):
        mask = (protos == proto).to_numpy()
        if mask.any():
            out[proto] = {"rows": int(mask.sum()), "flagged": float(flagged[mask].mean())}
    return out


def synthetic_anomalies(rng: np.random.Generator, n: int = 2000) -> dict[str, pd.DataFrame]:
    seconds = rng.uniform(0, 30, n)
    zeros = np.zeros(n)

    def frame(proto, port, src_bytes, src_packets, dst_bytes, handshake):
        return pd.DataFrame({
            "proto": proto, "dst_port": port, "src_byte_count": src_bytes, "src_packet_count": src_packets,
            "dst_byte_count": dst_bytes, "duration_millis": seconds * 1000, "handshake_latency_millis": handshake,
            "smean": src_bytes / np.maximum(src_packets, 1),
        })

    syn_packets = rng.integers(1, 3, n).astype(float)
    flood_packets = rng.integers(500, 5000, n).astype(float)
    upload_packets = rng.integers(5000, 50000, n).astype(float)
    return {
        "syn_scan (tcp, random port, 1-2 packets, no reply)": frame("tcp", rng.integers(1, 65535, n), syn_packets * 60, syn_packets, zeros, zeros),
        "udp_flood (udp, thousands of packets, no reply)": frame("udp", rng.integers(1, 65535, n), flood_packets * rng.uniform(512, 1400, n), flood_packets, zeros, zeros),
        "big_upload (tcp 443, tens of thousands of packets up)": frame("tcp", np.full(n, 443), upload_packets * 1400, upload_packets, rng.uniform(1e3, 1e5, n), rng.uniform(10, 60, n)),
    }


def unsw_cross_check(config: dict, model, mean, std, categories, threshold: float) -> dict:
    raw = load_raw(REPO_ROOT / config["dataset"]["raw_dir"], config["dataset"]["urls"])
    unsw = build_dataset(raw)
    unsw = unsw[unsw["proto"].isin(["tcp", "udp"])].reset_index(drop=True)
    errors = score(model, torch.from_numpy(encode(unsw, mean, std, categories)))
    out = {}
    for label, name in ((0, "normal"), (1, "attack")):
        mask = (unsw["label"] == label).to_numpy()
        out[name] = flag_rates(errors[mask], unsw.loc[mask, "proto"], threshold)
    return out


def run_split(config: dict, records: pd.DataFrame, seed: int) -> dict:
    cfg = dict(config["training"], random_state=seed)
    mirage = config["mirage"]
    train_df, calib_df, test_df, calib_apps, test_apps = split_by_app(
        records, mirage["test_app_fraction"], mirage["calibration_app_fraction"], seed
    )
    print(f"seed {seed}: train {len(train_df)} / calibration {len(calib_df)} / test {len(test_df)} snapshots; "
          f"test apps {test_apps}; calibration apps {calib_apps}", file=sys.stderr)

    categories = fit_categories(train_df, cfg["categorical_top_n"])
    categorical_dims = [len(categories[c]) for c in CATEGORICAL_COLUMNS]
    mean, std = numeric_stats(train_df)
    model = fit(torch.from_numpy(encode(train_df, mean, std, categories)), categorical_dims, cfg)

    calib_errors = score(model, torch.from_numpy(encode(calib_df, mean, std, categories)))
    test_errors = score(model, torch.from_numpy(encode(test_df, mean, std, categories)))
    table = {p: flag_rates(test_errors, test_df["proto"], float(np.percentile(calib_errors, p))) for p in [95, 97, 99, 99.5]}
    return {
        "seed": seed, "model": model, "mean": mean, "std": std, "categories": categories,
        "categorical_dims": categorical_dims, "calib_errors": calib_errors, "table": table,
        "train_df": train_df, "calib_df": calib_df, "test_df": test_df,
        "calib_apps": calib_apps, "test_apps": test_apps,
    }


def main():
    config = load_config()
    cfg = config["training"]
    mirage = config["mirage"]
    records = pd.read_parquet(REPO_ROOT / mirage["records"])

    seeds = [cfg["random_state"] + i for i in range(mirage.get("app_split_repeats", 5))]
    runs = [run_split(config, records, seed) for seed in seeds]
    primary = runs[0]
    model, mean, std, categories = primary["model"], primary["mean"], primary["std"], primary["categories"]
    categorical_dims = primary["categorical_dims"]
    train_df, calib_df, test_df = primary["train_df"], primary["calib_df"], primary["test_df"]
    calib_apps, test_apps, table = primary["calib_apps"], primary["test_apps"], primary["table"]
    calib_errors = primary["calib_errors"]

    chosen = mirage["threshold_percentile"]
    threshold = float(np.percentile(calib_errors, chosen))

    rng = np.random.default_rng(cfg["random_state"])
    synthetic = {}
    for name, frame in synthetic_anomalies(rng).items():
        errors = score(model, torch.from_numpy(encode(frame, mean, std, categories)))
        synthetic[name] = float((errors > threshold).mean())

    cross = unsw_cross_check(config, model, mean, std, categories, threshold)

    checkpoint = {
        "state_dict": model.state_dict(),
        "numeric_dim": len(NUMERIC_COLUMNS),
        "categorical_dims": categorical_dims,
        "hidden_dim": cfg["hidden_dim"],
        "bottleneck_dim": cfg["bottleneck_dim"],
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categories": categories,
        "numeric_mean": mean,
        "numeric_std": std,
        "reconstruction_threshold": threshold,
    }
    torch.save(checkpoint, TRAINING_DIR / config["paths"]["model_out"])

    write_metrics(config, records, train_df, calib_df, test_df, test_apps, calib_apps, table, chosen, threshold, synthetic, cross, runs)
    print(json.dumps({"threshold": threshold, "test_flag_rates": table[chosen], "synthetic_detection": synthetic, "unsw_cross_check": cross}, indent=2))


def write_metrics(config, records, train_df, calib_df, test_df, test_apps, calib_apps, table, chosen, threshold, synthetic, cross, runs):
    cfg = config["training"]
    rows = "\n".join(
        f"| {p} | {r['all']['flagged']:.4f} | {r.get('tcp', {}).get('flagged', float('nan')):.4f} | {r.get('udp', {}).get('flagged', float('nan')):.4f} |"
        for p, r in table.items()
    )
    syn = "\n".join(f"- {name}: {rate:.4f} flagged" for name, rate in synthetic.items())
    unsw = "\n".join(
        f"- UNSW-NB15 {name} rows: " + ", ".join(f"{k} {v['flagged']:.4f} of {v['rows']}" for k, v in r.items())
        for name, r in cross.items()
    )
    proto_counts = records["proto"].value_counts().to_dict()
    spread = "\n".join(
        f"| {r['seed']} | {', '.join(r['test_apps'])} | {r['table'][chosen]['all']['flagged']:.4f} | "
        f"{r['table'][chosen].get('tcp', {}).get('flagged', float('nan')):.4f} | {r['table'][chosen].get('udp', {}).get('flagged', float('nan')):.4f} |"
        for r in runs
    )
    overall = [r["table"][chosen]["all"]["flagged"] for r in runs]
    text = (
        "# Autoencoder trained on MIRAGE-2019 (real Android app traffic, normal only)\n\n"
        "Normal-traffic anomaly detector: no attack label is used. One shared encoder/bottleneck, MSE head over the "
        f"{len(NUMERIC_COLUMNS)} log-transformed, z-scored numeric features and a softmax head for `proto`; per-row error = mean "
        f"numeric squared error + cross-entropy. Architecture input -> {cfg['hidden_dim']} -> {cfg['bottleneck_dim']} -> "
        f"{cfg['hidden_dim']} -> outputs, LeakyReLU, {cfg['epochs']} epochs, Adam lr {cfg['learning_rate']}.\n\n"
        "## Data and features\n\n"
        f"- Source: MIRAGE-2019, the downloadable release: {records['app'].nunique()} Android apps on {records['device'].nunique()} devices (the paper describes 40 apps; this public release is a subset). {CITATION}\n"
        "- **The licence is non-commercial and no-derivatives. A commercial product must retrain on Warden's own logged traffic.**\n"
        f"- {len(records)} destination snapshots ({proto_counts}), from {records['capture'].nunique()} captures and {records['app'].nunique()} apps.\n"
        "- One record per destination IP per capture, at a random snapshot time in 0-30 s, aggregating that destination's flows. "
        "Per-packet detail exists only for each flow's first 32 packets; beyond that, counts are interpolated linearly from flow totals.\n"
        "- Flows have no absolute start time, so a destination's flows are assumed to start together (this overcounts if they start later).\n"
        "- Upstream IP bytes are estimated as payload plus the flow's average header overhead (per-packet IP length is not stored).\n"
        "- No TCP flags in the data, so **`state` is not a feature** here. `dttl` is not a feature (not capturable on Android).\n"
        "- Mostly HTTPS over TCP; UDP is thin. Captures are single-app sessions, so 'normal' means these 40 apps.\n\n"
        "## Split (by app, so the test measures unseen apps)\n\n"
        f"- Train {len(train_df)}, calibration {len(calib_df)} ({', '.join(calib_apps)}), test {len(test_df)} ({', '.join(test_apps)}).\n"
        f"- Threshold: {chosen}th percentile of calibration-app reconstruction error = {threshold:.5f}.\n\n"
        "## False-flag rate on held-out normal traffic (unseen apps)\n\n"
        "| calibration percentile | all | TCP | UDP |\n|---|---|---|---|\n" + rows + "\n\n"
        "If the model generalises across apps, the rate should be close to 1 minus the percentile. A larger number means unseen apps look unusual.\n\n"
        f"### Spread across {len(runs)} different app splits (each retrains from scratch; the shipped model is the first)\n\n"
        f"At the {chosen}th percentile the false-flag rate on unseen apps ranged from {min(overall):.4f} to {max(overall):.4f} (mean {np.mean(overall):.4f}); with only {records['app'].nunique()} apps, which apps are held out matters a lot.\n\n"
        "| seed | held-out apps | all | TCP | UDP |\n|---|---|---|---|---|\n" + spread + "\n\n"
        "## Sanity checks, NOT validation\n\n"
        "Synthetic anomaly shapes I defined in feature space (they show the model is not blind to extreme shapes, nothing more):\n\n" + syn + "\n\n"
        "UNSW-NB15 rows scored with this model at the same threshold (different network, whole-flow records, so units differ; treat as a rough shift indicator only):\n\n" + unsw + "\n\n"
        "## Limits\n\n"
        "- There is no attack traffic in MIRAGE, so detection ability in this domain is unmeasured.\n"
        "- Trained on 3 devices and 40 apps; another phone, OS version or app mix may shift the distribution.\n"
        "- The threshold changes on every retrain; read `reconstruction_threshold` from the model file.\n"
        "- Superseded once a model trained on Warden-captured traffic exists.\n"
    )
    (TRAINING_DIR / config["paths"]["metrics_out"]).write_text(text)


if __name__ == "__main__":
    main()
