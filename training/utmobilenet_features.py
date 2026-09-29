import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import REPO_ROOT, load_config

SNAPSHOT_MAX_SECONDS = 30.0
SNAPSHOTS_PER_DESTINATION = 3
SESSION_GAP_SECONDS = 300.0
TCP_HEADER_OVERHEAD = 52.0
UDP_HEADER_OVERHEAD = 28.0

APP_ALIASES = {
    "youtube": "com.google.android.youtube",
    "spotify": "com.spotify.music",
    "pinterest": "com.pinterest",
    "facebook": "com.facebook.katana",
    "twitter": "com.twitter.android",
    "dropbox": "com.dropbox.android",
    "messenger": "com.facebook.orca",
}


def prepare_flow(row) -> dict:
    sizes = np.asarray(row.pkts_size, dtype=float)
    directions = np.asarray(row.pkts_dir)
    times = np.asarray(row.timetofirst, dtype=float)
    order = np.argsort(times, kind="stable")
    sizes, directions, times = sizes[order], directions[order], times[order]
    handshake = None
    if row.ip_proto == 6 and directions.size and directions[0] == 1:
        down = np.flatnonzero(directions == 0)
        if down.size:
            handshake = float(times[down[0]])
    return {
        "start": float(row.first), "proto": "tcp" if row.ip_proto == 6 else "udp", "dst_port": int(row.dst_port),
        "sizes": sizes, "up": directions == 1, "times": times, "handshake": handshake,
    }


def snapshot(flows: list[dict], seconds: float) -> dict:
    origin = flows[0]["start"]
    up_packets = up_payload = down_payload = 0.0
    up_ip_bytes = 0.0
    for f in flows:
        offset = f["start"] - origin
        if offset > seconds:
            continue
        inside = (f["times"] + offset) <= seconds
        up = inside & f["up"]
        down = inside & ~f["up"]
        n_up = float(up.sum())
        overhead = TCP_HEADER_OVERHEAD if f["proto"] == "tcp" else UDP_HEADER_OVERHEAD
        up_packets += n_up
        up_payload += float(f["sizes"][up].sum())
        up_ip_bytes += float(f["sizes"][up].sum()) + n_up * overhead
        down_payload += float(f["sizes"][down].sum())
    first_tcp = next((f for f in flows if f["proto"] == "tcp"), None)
    handshake_ms = 0.0
    if first_tcp is not None and first_tcp["handshake"] is not None:
        if (first_tcp["start"] - origin) + first_tcp["handshake"] <= seconds:
            handshake_ms = first_tcp["handshake"] * 1000.0
    first = flows[0]
    return {
        "proto": first["proto"], "dst_port": first["dst_port"], "src_byte_count": up_ip_bytes,
        "src_packet_count": up_packets, "dst_byte_count": down_payload, "duration_millis": seconds * 1000.0,
        "handshake_latency_millis": handshake_ms,
        "smean": up_ip_bytes / up_packets if up_packets > 0 else 0.0, "n_flows": len(flows),
    }


def destination_sessions(group: pd.DataFrame) -> list[list[dict]]:
    group = group.sort_values("first")
    sessions: list[list[dict]] = []
    last_activity = -np.inf
    for row in group.itertuples():
        if not sessions or row.first - last_activity > SESSION_GAP_SECONDS:
            sessions.append([])
        sessions[-1].append(prepare_flow(row))
        last_activity = max(last_activity, row.last)
    return sessions


def process_group(args) -> list[dict]:
    (app, src_ip, dst_ip), group = args
    rows = []
    for index, flows in enumerate(destination_sessions(group)):
        rng = np.random.default_rng(zlib.crc32(f"{app}|{src_ip}|{dst_ip}|{index}".encode()))
        for _ in range(SNAPSHOTS_PER_DESTINATION):
            record = snapshot(flows, float(rng.uniform(0.0, SNAPSHOT_MAX_SECONDS)))
            rows.append({**record, "device": src_ip, "app": APP_ALIASES.get(app, app), "capture": f"utmobilenet21|{app}|{src_ip}|{dst_ip}|{index}"})
    return rows


def build(source: str, out_path: str) -> pd.DataFrame:
    flows = pd.read_parquet(source)
    flows = flows[flows["ip_proto"].isin([6, 17])]
    flows["app"] = flows["app"].astype(str)
    groups = list(flows.groupby(["app", "src_ip", "dst_ip"], sort=False))
    print(f"{len(flows)} flows in {len(groups)} (app, device, destination) groups", file=sys.stderr)
    with Pool() as pool:
        chunks = pool.map(process_group, groups, chunksize=32)
    df = pd.DataFrame([row for chunk in chunks for row in chunk])
    df.to_parquet(out_path, index=False)
    print(f"{len(df)} snapshots, {df['proto'].value_counts().to_dict()}, {df['app'].nunique()} apps -> {out_path}", file=sys.stderr)
    return df


if __name__ == "__main__":
    ut = load_config()["utmobilenet"]
    build(str(REPO_ROOT / ut["parquet"]), str(REPO_ROOT / ut["records"]))
