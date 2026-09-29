import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from config import load_config
from train_device_log import load_records, prepare, train_from_records


def synthetic_records(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        tcp = rng.random() < 0.7
        src_packets = int(rng.integers(3, 60))
        records.append({
            "v": 1,
            "def": 1,
            "day": f"2026-09-{20 + i % 8:02d}",
            "scored_after_s": int(rng.integers(0, 30)),
            "proto": "tcp" if tcp else "udp",
            "dst_port": 443 if tcp else 53,
            "state": str(rng.choice(["CON", "FIN"], p=[0.8, 0.2])) if tcp else "CON",
            "src_byte_count": int(src_packets * rng.normal(90, 10)),
            "src_packet_count": src_packets,
            "dst_byte_count": int(rng.lognormal(9, 1)),
            "dst_chunk_count": int(rng.integers(1, 40)),
            "duration_millis": int(rng.integers(50, 30000)),
            "handshake_latency_millis": float(rng.normal(30, 5)) if tcp else 0.0,
            "later_blocked": False,
        })
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def main():
    config = copy.deepcopy(load_config())
    config["training"]["epochs"] = 3
    config["device_log"]["min_records"] = 1

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.jsonl"
        write_jsonl(good, synthetic_records(6000))

        leaky = Path(tmp) / "leaky.jsonl"
        write_jsonl(leaky, [{**synthetic_records(1)[0], "dst_ip": "203.0.113.9"}])
        try:
            load_records([str(leaky)])
        except ValueError as e:
            assert "unexpected field" in str(e) and "dst_ip" in str(e), e
        else:
            raise AssertionError("a log carrying an identifier field must be rejected")

        records = synthetic_records(200, seed=1)
        records[0]["def"] = 2
        records[1]["later_blocked"] = True
        records[2]["proto"] = "icmp"
        stale = Path(tmp) / "mixed.jsonl"
        write_jsonl(stale, records)
        kept = prepare(load_records([str(stale)]), expected_def=1)
        assert len(kept) == 197, f"filters should drop exactly the stale-def, blocked and non-TCP/UDP rows, kept {len(kept)}"
        assert (kept["smean"] >= 0).all()

        result = train_from_records(load_records([str(good)]), config)

    assert result["split"].startswith("time split"), result["split"]
    assert result["calibration_records"] > 0 and result["test_records"] > 0
    flagged = result["test_normal_flagged_rate"]
    assert flagged < 0.10, f"same-distribution test days flagged {flagged:.3f}, expected near 1%"
    print(json.dumps(result, indent=2))
    print("OK (synthetic records only: proves the plumbing and the privacy guard, says nothing about real traffic)")


if __name__ == "__main__":
    main()
