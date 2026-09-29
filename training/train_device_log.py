import json
import sys

import numpy as np
import pandas as pd
import torch

from config import TRAINING_DIR, load_config
from dataset import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from encoding import encode, fit_categories, numeric_stats
from export_autoencoder_model import export
from train_autoencoder import fit, score

ALLOWED_KEYS = {
    "v", "def", "day", "scored_after_s", "proto", "dst_port", "state", "src_byte_count", "src_packet_count",
    "dst_byte_count", "dst_chunk_count", "duration_millis", "handshake_latency_millis", "later_blocked",
}
RAW_NUMERIC = [
    "dst_port", "src_byte_count", "src_packet_count", "dst_byte_count", "dst_chunk_count",
    "duration_millis", "handshake_latency_millis", "scored_after_s",
]


def load_records(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with open(path) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                extra = set(record) - ALLOWED_KEYS
                if extra:
                    raise ValueError(f"{path}:{n}: unexpected field(s) {sorted(extra)}; refusing to read a log that may contain identifiers")
                missing = ALLOWED_KEYS - set(record)
                if missing:
                    raise ValueError(f"{path}:{n}: missing field(s) {sorted(missing)}")
                rows.append(record)
    df = pd.DataFrame(rows)
    for col in RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["later_blocked"] = df["later_blocked"].astype(bool)
    return df


def prepare(df: pd.DataFrame, expected_def: int) -> pd.DataFrame:
    counts = {"records": len(df)}
    df = df[df["def"] == expected_def]
    counts["current_def"] = len(df)
    df = df[~df["later_blocked"]]
    counts["not_later_blocked"] = len(df)
    df = df[df["proto"].isin(["tcp", "udp"])].copy()
    counts["tcp_udp"] = len(df)
    print(f"records kept at each filter: {counts}", file=sys.stderr)

    packets = df["src_packet_count"].to_numpy(dtype=float)
    df["smean"] = np.where(packets > 0, df["src_byte_count"] / np.where(packets > 0, packets, 1), 0.0)
    unknown = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if unknown:
        raise ValueError(f"config [features] asks for {unknown}, which the device log does not carry")
    return df.reset_index(drop=True)


def split_three(df: pd.DataFrame, holdout_fraction: float, seed: int):
    days = sorted(df["day"].unique())
    k = max(1, round(len(days) * holdout_fraction / 2))
    if len(days) >= 5:
        test_days, calib_days = set(days[-k:]), set(days[-2 * k:-k])
        is_test = df["day"].isin(test_days)
        is_calib = df["day"].isin(calib_days)
        note = f"time split: calibration days {sorted(calib_days)}, test days {sorted(test_days)}"
    else:
        draw = np.random.default_rng(seed).random(len(df))
        is_test = pd.Series(draw >= 1 - holdout_fraction / 2, index=df.index)
        is_calib = pd.Series((draw >= 1 - holdout_fraction) & (draw < 1 - holdout_fraction / 2), index=df.index)
        note = "WARNING: fewer than 5 distinct days, used a random row split (near-duplicate records may leak across it)"
    return df[~is_test & ~is_calib], df[is_calib], df[is_test], note


def train_from_records(df: pd.DataFrame, config: dict, out_prefix: str | None = None) -> dict:
    cfg = config["training"]
    dcfg = config["device_log"]
    df = prepare(df, dcfg["def"])
    if len(df) < dcfg["min_records"]:
        print(f"WARNING: only {len(df)} usable records, below the recommended {dcfg['min_records']}", file=sys.stderr)

    train_df, calib_df, test_df, split_note = split_three(df, dcfg["holdout_fraction"], cfg["random_state"])
    print(f"{split_note}; train {len(train_df)} / calibration {len(calib_df)} / test {len(test_df)}", file=sys.stderr)

    categories = fit_categories(train_df, cfg["categorical_top_n"])
    categorical_dims = [len(categories[c]) for c in CATEGORICAL_COLUMNS]
    mean, std = numeric_stats(train_df)

    model = fit(torch.from_numpy(encode(train_df, mean, std, categories)), categorical_dims, cfg)

    calib_errors = score(model, torch.from_numpy(encode(calib_df, mean, std, categories)))
    threshold = float(np.percentile(calib_errors, dcfg["threshold_percentile"]))
    test_errors = score(model, torch.from_numpy(encode(test_df, mean, std, categories)))
    flagged = test_errors > threshold

    per_proto = {}
    for proto in ("tcp", "udp"):
        mask = (test_df["proto"] == proto).to_numpy()
        if mask.any():
            per_proto[proto] = {"rows": int(mask.sum()), "flagged_rate": float(flagged[mask].mean())}

    result = {
        "usable_records": len(df),
        "train_records": len(train_df),
        "calibration_records": len(calib_df),
        "test_records": len(test_df),
        "split": split_note,
        "threshold_percentile": dcfg["threshold_percentile"],
        "threshold": threshold,
        "test_normal_flagged_rate": float(flagged.mean()),
        "per_proto_flagged_rate": per_proto,
    }

    if dcfg.get("anomaly_log"):
        anomalies = prepare(load_records([dcfg["anomaly_log"]]), dcfg["def"])
        anomaly_errors = score(model, torch.from_numpy(encode(anomalies, mean, std, categories)))
        result["anomaly_records"] = len(anomalies)
        result["anomaly_detection_rate"] = float((anomaly_errors > threshold).mean())

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
    if out_prefix is not None:
        checkpoint_path = TRAINING_DIR / f"{out_prefix}.pt"
        torch.save(checkpoint, checkpoint_path)
        export(checkpoint_path, TRAINING_DIR / f"{out_prefix}_ondevice.json")
    return result


def main():
    config = load_config()
    dcfg = config["device_log"]
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: uv run train_device_log.py <feature-log.jsonl> [more.jsonl ...]")

    result = train_from_records(load_records(paths), config, out_prefix=dcfg["model_prefix"])
    metrics = (
        "# Autoencoder trained on Warden-captured normal traffic\n\n"
        f"- Usable records: {result['usable_records']} (train {result['train_records']}, calibration {result['calibration_records']}, test {result['test_records']})\n"
        f"- Split: {result['split']}\n"
        f"- Threshold: {result['threshold_percentile']}th percentile of calibration-set normal error = {result['threshold']:.5f}\n"
        f"- Test-set normal records flagged at that threshold: {result['test_normal_flagged_rate']:.4f} "
        f"(expected about {1 - result['threshold_percentile'] / 100:.4f}; a big gap means the test days differ from the calibration days)\n"
        + "".join(f"- {p}: {v['rows']} test records, {v['flagged_rate']:.4f} flagged\n" for p, v in result["per_proto_flagged_rate"].items())
        + (f"- Deliberate-anomaly records: {result['anomaly_records']}, detected {result['anomaly_detection_rate']:.4f}\n" if "anomaly_records" in result else
           "- Detection ability is NOT measured: no attack labels exist in this data. Supply `anomaly_log` (deliberate lab anomalies captured in the same format) to measure it.\n")
    )
    (TRAINING_DIR / dcfg["metrics_out"]).write_text(metrics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
