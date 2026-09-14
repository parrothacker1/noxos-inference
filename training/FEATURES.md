# Network classifier — feature schema

**Retrained 2026-09-14 against the locked spec in `knowledge-graph/noxos-inference/TASKS.md`'s "Real training spec, LOCKED 2026-09-14"** — supersedes the original 39-feature version trained on the reduced `UNSW_NB15_training-set.csv`. Everything below describes the current, real model.

Trained on the full, unreduced [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset (2,280,090 rows, ~11% attack rate — closer to Warden's real deployment than the reduced/rebalanced 175K-row file the original model used), fetched as two parquet files from the [Mouwiya/UNSW-NB15 HuggingFace mirror](https://huggingface.co/datasets/Mouwiya/UNSW-NB15):

```bash
mkdir -p data/raw
curl -sL "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/data/train-00000-of-00002.parquet" -o data/raw/train-00000-of-00002.parquet
curl -sL "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/data/train-00001-of-00002.parquet" -o data/raw/train-00001-of-00002.parquet
```

Gitignored (`data/raw/`), URLs live in `config.toml`'s `[dataset].urls` — the training script and CI workflow both read from there, nothing hardcoded twice. Held-out split is our own 80/20 stratified split (`config.toml`'s `[training]` section), not the dataset's original one.

## Feature columns the model expects — 10 fields, locked, F1 = 0.9724

Deliberately not the full 39-column schema — this set was chosen specifically to match what `noxos-app` can realistically supply per flagged destination (see the locked spec for the empirical progression that justified it: 0.9679 → 0.9715 with real `dst_port` → 0.9724 with derived `smean`/`dmean`). Feature names below are what `noxos-app` will send — the raw dataset columns get renamed/derived to match during training, not the other way around, so the exported tree JSON's `feature` strings are directly usable without a translation table on the app side.

| Feature name (what the model/JSON export use) | Raw dataset column | Notes |
|---|---|---|
| `proto` | `proto` | categorical, label-encoded (100+ real values in the full dataset — arp/ospf/sctp/icmp/etc., not just tcp/udp) |
| `dst_port` | `dsport` | **string in the raw data, not numeric** — 7 of 2,280,090 rows have hex (`0xc0a8`) or non-numeric (`-`) values; parsed as decimal first, then `int(x, 16) & 0xFFFF` as fallback, dropped if neither parses |
| `src_byte_count` | `sbytes` | |
| `src_packet_count` | `Spkts` | capital P in the full dataset — don't assume it matches the reduced set's lowercase `spkts` |
| `dst_byte_count` | `dbytes` | |
| `dst_packet_count` | `Dpkts` | capital D, same note |
| `duration_millis` | `dur` | **unit conversion**: dataset stores seconds, multiplied by 1000 during training so it matches what `noxos-app` sends in milliseconds |
| `handshake_latency_millis` | `tcprtt` | same seconds→milliseconds conversion; 0 for non-TCP flows in the raw data (no handshake to time), matching what `noxos-app` sends for UDP |
| `smean` | derived: `sbytes / Spkts` | mean source-side packet size; 0 when packet count is 0 rather than a division error |
| `dmean` | derived: `dbytes / Dpkts` | mean destination-side packet size, same zero-guard |

## Known gap: what `noxos-app` actually sends today

This 10-feature set was chosen to match what `noxos-app` *could* send, not what it *does* send yet — `AnalysisDispatcher` still only has `ip`/`priority`/`reason`/`first_flagged_epoch_millis` per flagged `AclEntry` as of this writing. `service/app.py`'s `/analyze/network` endpoint fills every missing field with a default stored in the model artifact, so a request with none of the real flow fields still gets a real (if low-information) prediction rather than erroring. Closing this gap means `noxos-app` persisting and sending the fields in the table above (some of it — `srcPacketCount`/`srcByteCount`/`dstPacketCount`/`dstByteCount`/`durationMillis`/`handshakeLatencyMillis` — already exists in `AclEntity` schema v8, just not wired into the dispatcher's request body yet; see `../noxos-app/TASKS.md`).

**Defaults are computed from `label == 0` (normal) rows only, not the whole dataset.** An almost-empty request (today's real case) would otherwise read as attack-like, since this dataset's overall column medians skew toward its ~11%-but-still-overrepresented-in-aggregate attack traffic more than a truly random normal flow does. A security feature defaulting to "block" when it has no real signal is the wrong failure mode. Conditioning defaults on normal-only rows fixes this: an empty request reads as a typical *normal* flow.

## Verdict thresholds and the safety_score polarity fix

Internally, `model.predict_proba()[1]` (probability of `label=1`, i.e. attack) is bucketed:
- `< 0.4` → `"allow"`
- `0.4`–`0.6` → `"uncertain"`
- `> 0.6` → `"block"`

Picked as a reasonable starting split, not tuned against any specific cost function — revisit once there's real traffic to check false-positive/false-negative rates against.

**Polarity note (resolved 2026-09-13, see `knowledge-graph/noxos-inference/TASKS.md`):** `noxos-app`'s `AnalysisDispatcher` convention is *higher `safety_score` = safer* (its own tests pair `0.02` with a `block` verdict). The model's natural output above is an attack probability — higher = more dangerous, opposite polarity. Fixed at the API boundary, not internally: `service/app.py` exposes `safety_score = 1.0 - attack_probability` in the JSON response, so the model/training code keeps its natural attack-probability representation and only the wire format inverts.
