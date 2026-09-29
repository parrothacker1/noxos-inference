import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config

_features = load_config()["features"]
NUMERIC_COLUMNS = _features["numeric"]
CATEGORICAL_COLUMNS = _features["categorical"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
LOG_COLUMNS = _features.get("log", ["dst_port"])

CANDIDATE_COLUMNS = {
    "src_bits_per_sec": "Sload",
    "dst_bits_per_sec": "Dload",
    "src_loss": "sloss",
    "dst_loss": "dloss",
    "src_win": "swin",
    "dst_win": "dwin",
    "src_jitter": "Sjit",
    "dst_jitter": "Djit",
    "src_interpkt_millis": "Sintpkt",
    "dst_interpkt_millis": "Dintpkt",
    "synack_millis": "synack",
    "ackdat_millis": "ackdat",
    "ct_srv_src": "ct_srv_src",
    "ct_srv_dst": "ct_srv_dst",
    "ct_dst_ltm": "ct_dst_ltm",
    "ct_src_ltm": "ct_src_ltm",
    "ct_src_dport_ltm": "ct_src_dport_ltm",
    "ct_dst_sport_ltm": "ct_dst_sport_ltm",
    "ct_dst_src_ltm": "ct_dst_src_ltm",
}
SECONDS_TO_MILLIS = {"synack", "ackdat"}


def raw_file_path(raw_dir: Path, url: str) -> Path:
    return raw_dir / url.rsplit("/", 1)[-1]


def load_raw(raw_dir: Path, urls: list[str]) -> pd.DataFrame:
    paths = [raw_file_path(raw_dir, url) for url in urls]
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = [f'curl -sL "{u}" -o {p}' for u, p in zip(urls, paths) if p in missing]
        sys.exit("missing raw dataset file(s) — fetch first:\n" + "\n".join(lines))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def parse_port(value) -> float:
    text = str(value).strip()
    try:
        return float(int(text))
    except ValueError:
        pass
    try:
        return float(int(text, 16) & 0xFFFF)
    except ValueError:
        return np.nan


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["label"] = df["label"].astype(int)
    x["proto"] = df["proto"].astype(str)
    x["state"] = df["state"].astype(str)
    x["dttl"] = df["dttl"].astype(float)
    x["dst_port"] = df["dsport"].map(parse_port)
    x["src_byte_count"] = df["sbytes"].astype(float)
    x["src_packet_count"] = df["Spkts"].astype(float)
    x["dst_byte_count"] = df["dbytes"].astype(float)
    x["dst_packet_count"] = df["Dpkts"].astype(float)
    x["duration_millis"] = df["dur"].astype(float) * 1000.0
    x["handshake_latency_millis"] = df["tcprtt"].astype(float) * 1000.0
    x["smean"] = (x["src_byte_count"] / x["src_packet_count"].replace(0, np.nan)).fillna(0.0)
    x["dmean"] = (x["dst_byte_count"] / x["dst_packet_count"].replace(0, np.nan)).fillna(0.0)

    for name, raw in CANDIDATE_COLUMNS.items():
        scale = 1000.0 if raw in SECONDS_TO_MILLIS else 1.0
        x[name] = df[raw].astype(float) * scale

    before = len(x)
    x = x.dropna(subset=["dst_port"])
    dropped = before - len(x)
    print(f"dropped {dropped} rows with unparseable dst_port (out of {before})", file=sys.stderr)

    return x
