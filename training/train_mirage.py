import json
import sys

import numpy as np
import pandas as pd
import torch

from config import REPO_ROOT, TRAINING_DIR, load_config
from dataset import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from encoding import encode, fit_categories, numeric_stats, transform_map
from train_autoencoder import fit, score

UT_CITATION = (
    'Yuqiang Heng, Vikram Chandrasekhar, Jeffrey G. Andrews, "UTMobileNetTraffic2021: A Labeled Public Network Traffic Dataset", '
    "IEEE Networking Letters 3(3), 2021 (curated copy on figshare, GPL 3.0+)."
)
CITATION = (
    "Giuseppe Aceto, Domenico Ciuonzo, Antonio Montieri, Valerio Persico and Antonio Pescapè, "
    '"MIRAGE: Mobile-app Traffic Capture and Ground-truth Creation", 4th IEEE International Conference on '
    "Computing, Communications and Security (ICCCS 2019). Dataset licence: CC BY-NC-ND 4.0."
)
MIN_CALIBRATION_ROWS = 200
PROTOS = ("tcp", "udp")


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


def per_protocol_thresholds(errors: np.ndarray, protos: pd.Series, percentile: float) -> dict[str, float]:
    pooled = float(np.percentile(errors, percentile))
    out = {}
    for proto in PROTOS:
        mask = (protos == proto).to_numpy()
        out[proto] = float(np.percentile(errors[mask], percentile)) if mask.sum() >= MIN_CALIBRATION_ROWS else pooled
    return out


def flagged(errors: np.ndarray, protos: pd.Series, thresholds: dict[str, float]) -> np.ndarray:
    limit = np.where((protos == "tcp").to_numpy(), thresholds["tcp"], thresholds["udp"])
    return errors > limit


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
        "syn_scan": frame("tcp", rng.integers(1, 65535, n), syn_packets * 60, syn_packets, zeros, zeros),
        "udp_flood": frame("udp", rng.integers(1, 65535, n), flood_packets * rng.uniform(512, 1400, n), flood_packets, zeros, zeros),
        "big_upload": frame("tcp", np.full(n, 443), upload_packets * 1400, upload_packets, rng.uniform(1e3, 1e5, n), rng.uniform(10, 60, n)),
    }


def run_split(config: dict, records: pd.DataFrame, seed: int) -> dict:
    cfg = dict(config["training"], random_state=seed)
    mirage = config["mirage"]
    train_df, calib_df, test_df, calib_apps, test_apps = split_by_app(
        records, mirage["test_app_fraction"], mirage["calibration_app_fraction"], seed
    )
    print(f"seed {seed}: train {len(train_df)} / calibration {len(calib_df)} / test {len(test_df)}; "
          f"test apps {test_apps}; calibration apps {calib_apps}", file=sys.stderr)

    categories = fit_categories(train_df, cfg["categorical_top_n"])
    categorical_dims = [len(categories[c]) for c in CATEGORICAL_COLUMNS]
    mean, std = numeric_stats(train_df)
    model = fit(torch.from_numpy(encode(train_df, mean, std, categories)), categorical_dims, cfg)

    def errors_of(df):
        return score(model, torch.from_numpy(encode(df, mean, std, categories)))

    calib_errors, test_errors = errors_of(calib_df), errors_of(test_df)
    synthetic_errors = {name: errors_of(frame) for name, frame in synthetic_anomalies(np.random.default_rng(seed)).items()}
    synthetic_protos = {name: frame["proto"] for name, frame in synthetic_anomalies(np.random.default_rng(seed)).items()}

    results = {}
    for p in mirage["threshold_percentiles"]:
        thresholds = per_protocol_thresholds(calib_errors, calib_df["proto"], p)
        test_flags = flagged(test_errors, test_df["proto"], thresholds)
        rates = {"all": float(test_flags.mean())}
        for proto in PROTOS:
            mask = (test_df["proto"] == proto).to_numpy()
            rates[proto] = float(test_flags[mask].mean()) if mask.any() else float("nan")
        detection = {name: float(flagged(e, synthetic_protos[name], thresholds).mean()) for name, e in synthetic_errors.items()}
        results[p] = {"thresholds": thresholds, "false_flag": rates, "detection": detection}

    return {
        "seed": seed, "model": model, "mean": mean, "std": std, "categories": categories,
        "categorical_dims": categorical_dims, "results": results,
        "calib_udp_rows": int((calib_df["proto"] == "udp").sum()),
        "train_rows": len(train_df), "calib_rows": len(calib_df), "test_rows": len(test_df),
        "calib_apps": calib_apps, "test_apps": test_apps,
    }


def spread(runs: list[dict], p: float, getter) -> str:
    values = np.array([getter(r["results"][p]) for r in runs], dtype=float)
    return f"{np.nanmean(values):.4f} ({np.nanmin(values):.4f}-{np.nanmax(values):.4f})"


def main():
    config = load_config()
    cfg = config["training"]
    mirage = config["mirage"]
    records = pd.read_parquet(REPO_ROOT / mirage["records"]).assign(source="mirage")
    if config["utmobilenet"]["enabled"]:
        ut = pd.read_parquet(REPO_ROOT / config["utmobilenet"]["records"]).assign(source="utmobilenet")
        records = pd.concat([records, ut], ignore_index=True)

    seeds = [cfg["random_state"] + i for i in range(mirage["app_split_repeats"])]
    runs = [run_split(config, records, seed) for seed in seeds]
    primary = runs[0]
    chosen = {"tcp": mirage["threshold_percentile_tcp"], "udp": mirage["threshold_percentile_udp"]}
    thresholds = {proto: primary["results"][chosen[proto]]["thresholds"][proto] for proto in PROTOS}

    checkpoint = {
        "state_dict": primary["model"].state_dict(),
        "numeric_dim": len(NUMERIC_COLUMNS),
        "categorical_dims": primary["categorical_dims"],
        "hidden_dim": cfg["hidden_dim"],
        "bottleneck_dim": cfg["bottleneck_dim"],
        "numeric_columns": NUMERIC_COLUMNS,
        "numeric_transforms": transform_map(),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categories": primary["categories"],
        "numeric_mean": primary["mean"],
        "numeric_std": primary["std"],
        "reconstruction_thresholds": thresholds,
        "threshold_percentiles": chosen,
    }
    torch.save(checkpoint, TRAINING_DIR / config["paths"]["model_out"])
    write_metrics(config, records, runs, chosen)
    print(json.dumps({"chosen_percentiles": chosen, "thresholds": thresholds}, indent=2))


def write_metrics(config, records, runs, chosen):
    cfg = config["training"]
    mirage = config["mirage"]
    primary = runs[0]
    percentiles = mirage["threshold_percentiles"]
    ff_rows = "\n".join(
        f"| {p} | {spread(runs, p, lambda r: r['false_flag']['all'])} | {spread(runs, p, lambda r: r['false_flag']['tcp'])} | {spread(runs, p, lambda r: r['false_flag']['udp'])} |"
        for p in percentiles
    )
    det_rows = "\n".join(
        f"| {p} | {spread(runs, p, lambda r: r['detection']['syn_scan'])} | {spread(runs, p, lambda r: r['detection']['udp_flood'])} | {spread(runs, p, lambda r: r['detection']['big_upload'])} |"
        for p in percentiles
    )
    split_rows = "\n".join(
        f"| {r['seed']} | {', '.join(r['test_apps'])} | {r['calib_udp_rows']} | {r['results'][chosen['tcp']]['false_flag']['tcp']:.4f} | {r['results'][chosen['udp']]['false_flag']['udp']:.4f} |"
        for r in runs
    )
    shipped_tcp = spread(runs, chosen["tcp"], lambda r: r["false_flag"]["tcp"])
    shipped_udp = spread(runs, chosen["udp"], lambda r: r["false_flag"]["udp"])
    shipped_syn = spread(runs, chosen["tcp"], lambda r: r["detection"]["syn_scan"])
    shipped_upload = spread(runs, chosen["tcp"], lambda r: r["detection"]["big_upload"])
    shipped_flood = spread(runs, chosen["udp"], lambda r: r["detection"]["udp_flood"])
    ut_count = int((records["source"] == "utmobilenet").sum())
    ut_line = (
        f"- Source 2: UTMobileNetTraffic2021, {ut_count} snapshots from emulated Android app interactions, 85% of its flows UDP and mostly DNS. "
        f"{UT_CITATION} Apps present in both datasets are treated as one app for splitting.\n"
        if ut_count else ""
    )
    transforms = ", ".join(f"{c}: {t}" for c, t in transform_map().items())
    thr = {proto: primary["results"][chosen[proto]]["thresholds"][proto] for proto in PROTOS}
    text = (
        "# Autoencoder trained on MIRAGE-2019 (real Android app traffic, normal only)\n\n"
        "Normal-traffic anomaly detector, no attack label used. Shared encoder/bottleneck, MSE head over the "
        f"{len(NUMERIC_COLUMNS)} transformed, z-scored numeric features and a softmax head for `proto`; per-row error = mean numeric "
        f"squared error + cross-entropy. Input -> {cfg['hidden_dim']} -> {cfg['bottleneck_dim']} -> {cfg['hidden_dim']} -> outputs, "
        f"LeakyReLU, {cfg['epochs']} epochs, Adam lr {cfg['learning_rate']}.\n\n"
        "## Data and features\n\n"
        f"- Source 1: MIRAGE-2019, the downloadable release: {records[records['source'] == 'mirage']['app'].nunique()} Android apps on 2 devices "
        f"(the paper describes 40 apps; this public release is a subset). {CITATION}\n"
        f"{ut_line}"
        "- **The licence is non-commercial and no-derivatives. A commercial product must retrain on Warden's own logged traffic.**\n"
        f"- {len(records)} destination snapshots ({records['proto'].value_counts().to_dict()}) from {records['capture'].nunique()} captures.\n"
        "- One record per destination IP per capture, at a random snapshot time in 0-30 s, aggregating that destination's flows. "
        "Per-packet detail exists only for each flow's first 32 packets; later counts are interpolated linearly from flow totals. "
        "Flows have no absolute start time, so a destination's flows are assumed to start together. Upstream IP bytes are payload plus the "
        "flow's average header overhead.\n"
        "- No TCP flags in the data, so `state` is not a feature; `dttl` is not a feature. Mostly HTTPS over TCP; UDP is thin.\n"
        f"- Input transforms (also written into the export as `numeric_transforms`): {transforms}.\n\n"
        f"## Thresholds: one per protocol, from calibration apps of that protocol\n\n"
        f"Shipped thresholds: TCP = {chosen['tcp']}th percentile of TCP calibration error = {thr['tcp']:.5f}; "
        f"UDP = {chosen['udp']}th percentile of UDP calibration error = {thr['udp']:.5f} "
        f"(a protocol with fewer than {MIN_CALIBRATION_ROWS} calibration rows falls back to the pooled percentile). "
        "UDP uses a looser percentile because its calibration data is thin and UDP false flags are high.\n\n"
        "**Shipped configuration, mean (min-max) over the app splits**\n\n"
        f"- False-flag rate on unseen-app normal traffic: TCP {shipped_tcp}, UDP {shipped_udp}\n"
        f"- Synthetic shapes detected (sanity only): SYN scan {shipped_syn}, UDP flood {shipped_flood}, big upload {shipped_upload}\n\n"
        f"## Held-out normal traffic (apps never seen in training), mean (min-max) over {len(runs)} different app splits\n\n"
        "False-flag rate; ideal is about 1 minus the percentile.\n\n"
        "| calibration percentile | all | TCP | UDP |\n|---|---|---|---|\n" + ff_rows + "\n\n"
        "Per split at the shipped percentiles (calibration UDP rows shows how thin the UDP calibration is):\n\n"
        "| seed | held-out apps | calibration UDP rows | TCP flagged | UDP flagged |\n|---|---|---|---|---|\n" + split_rows + "\n\n"
        "## Synthetic anomaly shapes, sanity only, NOT validation\n\n"
        "Shapes I defined in feature space, detected fraction (mean (min-max) over splits):\n\n"
        "| calibration percentile | SYN scan (tcp) | UDP flood (udp) | big upload (tcp) |\n|---|---|---|---|\n" + det_rows + "\n\n"
        "## What this does NOT show\n\n"
        "- **F1 is not measurable**: MIRAGE has no attack traffic. Detection ability awaits Warden-logged normal traffic plus deliberately generated test traffic.\n"
        "- The UNSW-NB15 proxy was dropped: it measured the gap between networks, not attacks.\n"
        f"- {records['app'].nunique()} apps in total; another phone, OS version or app mix may shift the distribution.\n"
        "- Thresholds change on every retrain; always read them from the model file. Superseded once a model trained on Warden-captured traffic exists.\n"
    )
    (TRAINING_DIR / config["paths"]["metrics_out"]).write_text(text)


if __name__ == "__main__":
    main()
