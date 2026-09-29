import glob
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import REPO_ROOT, load_config

SNAPSHOT_MAX_SECONDS = 30.0
SNAPSHOTS_PER_DESTINATION = 3


def flow_state(key: str, record: dict) -> dict:
    src_ip, src_port, dst_ip, dst_port, proto_number = key.split(",")
    packets = record["packet_data"]
    meta = record["flow_metadata"]
    directions = np.asarray(packets["packet_dir"])
    payload = np.asarray(packets["L4_payload_bytes"], dtype=float)
    times = np.cumsum(np.asarray(packets["iat"], dtype=float))
    up = directions == 0
    down = directions == 1

    up_overhead = (meta["UF_IP_packet_bytes"] - meta["UF_L4_payload_bytes"]) / max(meta["UF_num_packets"], 1)
    handshake = None
    if proto_number == "6" and directions.size and directions[0] == 0 and down.any():
        handshake = float(times[np.argmax(down)])

    return {
        "dst_ip": dst_ip,
        "dst_port": int(dst_port),
        "proto": "tcp" if proto_number == "6" else "udp",
        "times": times,
        "up_pkt_cum": np.cumsum(up.astype(float)),
        "up_payload_cum": np.cumsum(np.where(up, payload, 0.0)),
        "down_payload_cum": np.cumsum(np.where(down, payload, 0.0)),
        "recorded": int(directions.size),
        "total_packets": int(meta["BF_num_packets"]),
        "duration": float(meta["BF_duration"]),
        "up_packets_total": float(meta["UF_num_packets"]),
        "up_payload_total": float(meta["UF_L4_payload_bytes"]),
        "down_payload_total": float(meta["DF_L4_payload_bytes"]),
        "up_overhead": up_overhead,
        "handshake": handshake,
    }


def cumulative_at(flow: dict, seconds: float, prefix_key: str, total_key: str) -> float:
    times = flow["times"]
    if times.size == 0:
        return 0.0
    idx = np.searchsorted(times, seconds, side="right")
    prefix = float(flow[prefix_key][idx - 1]) if idx > 0 else 0.0
    if seconds <= times[-1] or flow["recorded"] >= flow["total_packets"]:
        return prefix
    tail = flow["duration"] - times[-1]
    fraction = 1.0 if tail <= 0 else min(max((seconds - times[-1]) / tail, 0.0), 1.0)
    prefix_at_end = float(flow[prefix_key][-1])
    return prefix_at_end + (flow[total_key] - prefix_at_end) * fraction


def destination_record(flows: list[dict], seconds: float) -> dict:
    up_packets = sum(cumulative_at(f, seconds, "up_pkt_cum", "up_packets_total") for f in flows)
    up_payload = sum(cumulative_at(f, seconds, "up_payload_cum", "up_payload_total") for f in flows)
    up_ip_bytes = sum(
        cumulative_at(f, seconds, "up_payload_cum", "up_payload_total")
        + cumulative_at(f, seconds, "up_pkt_cum", "up_packets_total") * f["up_overhead"]
        for f in flows
    )
    down_payload = sum(cumulative_at(f, seconds, "down_payload_cum", "down_payload_total") for f in flows)
    first_tcp = next((f for f in flows if f["proto"] == "tcp"), None)
    handshake_ms = 0.0
    if first_tcp is not None and first_tcp["handshake"] is not None and first_tcp["handshake"] <= seconds:
        handshake_ms = first_tcp["handshake"] * 1000.0
    first = flows[0]
    return {
        "proto": first["proto"],
        "dst_port": first["dst_port"],
        "src_byte_count": up_ip_bytes,
        "src_packet_count": up_packets,
        "dst_byte_count": down_payload,
        "duration_millis": seconds * 1000.0,
        "handshake_latency_millis": handshake_ms,
        "smean": up_ip_bytes / up_packets if up_packets > 0 else 0.0,
        "n_flows": len(flows),
    }


def process_file(path: str) -> list[dict]:
    device = os.path.basename(os.path.dirname(path))
    app = os.path.basename(path).split("_", 1)[1].split("_MIRAGE")[0]
    with open(path) as f:
        biflows = json.load(f)

    by_destination: dict[str, list[dict]] = {}
    for key, record in biflows.items():
        flow = flow_state(key, record)
        by_destination.setdefault(flow["dst_ip"], []).append(flow)

    rows = []
    for dst_ip, flows in by_destination.items():
        rng = np.random.default_rng(zlib.crc32(f"{path}|{dst_ip}".encode()))
        for _ in range(SNAPSHOTS_PER_DESTINATION):
            seconds = float(rng.uniform(0.0, SNAPSHOT_MAX_SECONDS))
            rows.append({**destination_record(flows, seconds), "device": device, "app": app, "capture": os.path.basename(path)})
    return rows


def build(directory: str, out_path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(directory, "*", "*.json")))
    print(f"{len(files)} capture files", file=sys.stderr)
    with Pool() as pool:
        chunks = pool.map(process_file, files, chunksize=8)
    df = pd.DataFrame([row for chunk in chunks for row in chunk])
    df.to_parquet(out_path, index=False)
    print(f"{len(df)} destination snapshots from {df['capture'].nunique()} captures, {df['app'].nunique()} apps -> {out_path}", file=sys.stderr)
    return df


if __name__ == "__main__":
    config = load_config()
    mirage = config["mirage"]
    build(str(REPO_ROOT / mirage["dir"]), str(REPO_ROOT / mirage["records"]))
