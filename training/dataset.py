import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "dst_port",
    "src_byte_count",
    "src_packet_count",
    "dst_byte_count",
    "dst_packet_count",
    "duration_millis",
    "handshake_latency_millis",
    "smean",
    "dmean",
]
CATEGORICAL_COLUMNS = ["proto"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


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
    x["dst_port"] = df["dsport"].map(parse_port)
    x["src_byte_count"] = df["sbytes"].astype(float)
    x["src_packet_count"] = df["Spkts"].astype(float)
    x["dst_byte_count"] = df["dbytes"].astype(float)
    x["dst_packet_count"] = df["Dpkts"].astype(float)
    x["duration_millis"] = df["dur"].astype(float) * 1000.0
    x["handshake_latency_millis"] = df["tcprtt"].astype(float) * 1000.0
    x["smean"] = (x["src_byte_count"] / x["src_packet_count"].replace(0, np.nan)).fillna(0.0)
    x["dmean"] = (x["dst_byte_count"] / x["dst_packet_count"].replace(0, np.nan)).fillna(0.0)

    before = len(x)
    x = x.dropna(subset=["dst_port"])
    dropped = before - len(x)
    print(f"dropped {dropped} rows with unparseable dst_port (out of {before})", file=sys.stderr)

    return x
