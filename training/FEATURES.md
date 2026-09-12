# Network classifier — feature schema

Trained on [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (`UNSW_NB15_training-set.csv`, fetched from the [Mouwiya/UNSW-NB15 HuggingFace mirror](https://huggingface.co/datasets/Mouwiya/UNSW-NB15) — the official UNSW CloudStor link works too, HuggingFace was just the easiest to script against without an interactive download). Only the training-set file exists in that mirror (no matching testing-set.csv), so the held-out split is our own 80/20 stratified split, not the dataset's original one — see `training/METRICS.md` for the real numbers this produces.

## Fetching the raw data

```bash
mkdir -p data/raw
curl -sL "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv" -o data/raw/UNSW_NB15_training-set.csv
```

Gitignored (`data/raw/`) — 32MB, re-fetchable, no reason to bloat the repo.

## Feature columns the model expects

All UNSW-NB15 columns except `id` (row index), `attack_cat` (would leak the label — it's the specific attack name), and `label` (the target itself). 39 features total, 3 categorical (`proto`, `service`, `state` — label-encoded using categories learned at training time, saved in the model artifact) and the rest numeric flow statistics (`dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `ct_*` connection-count features, etc. — see `data/raw/NUSW-NB15_features.csv` for the dataset's own column descriptions).

## Known gap: what `noxos-app` actually sends today

`AnalysisDispatcher` in `noxos-app` only has `ip`/`priority`/`reason`/`first_flagged_epoch_millis` per flagged `AclEntry` — none of the 39 columns above. `service/app.py`'s `/analyze/network` endpoint fills every missing field with a default stored in the model artifact, so a request with none of the real flow fields still gets a real (if low-information) prediction rather than erroring. This means today's actual verdicts are close to whatever the model predicts for a "typical" flow — not meaningfully informed by the specific destination. Closing this gap means either extending `AclEntity`/`AnalysisDispatcher` in `noxos-app` to persist and send real per-flow stats, or accepting that this model is only useful once richer request data exists. See `noxos-inference/AGENTS.md` for the fuller discussion.

**Defaults are computed from `label == 0` (normal) rows only, not the whole dataset.** First version used the whole dataset's median/mode — an almost-empty request (today's real case) came back `"block"` with a low safety score, because UNSW-NB15's overall column medians happen to resemble attack traffic more than normal traffic (this dataset skews attack-heavy). A security feature defaulting to "block" when it has no real signal is the wrong failure mode — it should lean toward "I don't actually know" rather than confidently wrong. Conditioning defaults on normal-only rows fixes this: an empty request now reads as an unremarkable, typical *normal* flow instead of an unremarkable typical *dataset* row.

## Verdict thresholds and the safety_score polarity fix

Internally, `model.predict_proba()[1]` (probability of `label=1`, i.e. attack) is bucketed:
- `< 0.4` → `"allow"`
- `0.4`–`0.6` → `"uncertain"`
- `> 0.6` → `"block"`

Picked as a reasonable starting split, not tuned against any specific cost function — revisit once there's real traffic to check false-positive/false-negative rates against.

**Polarity note (resolved 2026-09-13, see `knowledge-graph/noxos-inference/TASKS.md`):** `noxos-app`'s `AnalysisDispatcher` convention is *higher `safety_score` = safer* (its own tests pair `0.02` with a `block` verdict). The model's natural output above is an attack probability — higher = more dangerous, opposite polarity. Fixed at the API boundary, not internally: `service/app.py` exposes `safety_score = 1.0 - attack_probability` in the JSON response, so the model/training code keeps its natural attack-probability representation and only the wire format inverts.
