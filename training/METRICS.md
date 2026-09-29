# Autoencoder — unsupervised anomaly-detection tier (Tier 1)

Trained on normal traffic only (1616920 rows, `label == 0`), never sees an attack label. One shared encoder/bottleneck; decoder heads: linear/MSE over the 7 z-score-standardized numeric features, plus one softmax/cross-entropy head per categorical feature (proto, state). Architecture: input(17) -> hidden(32) -> bottleneck(4) -> hidden(32) -> [numeric(7), proto(3), state(7)], LeakyReLU, 60 epochs, Adam lr=0.001.

Reconstruction error per row = **mean** over numeric features of squared error + sum over categorical heads of per-row cross-entropy (unweighted).

## Encoding

- `dst_port`: `log1p` before standardization.
- Categorical features bucketed to the top 6 values by frequency in the normal-only training split plus `other`:
  - `proto`: tcp, udp, other
  - `state`: FIN, CON, INT, REQ, RST, CLO, other
- `dttl` (destination TTL) and `state` (connection state) were added after a feature sweep — the original 10 features capped F1 at ~0.75 (see `knowledge-graph/noxos-inference/TASKS.md`).

## Held-out evaluation (`StratifiedGroupKFold`, groups = exact feature pattern over all model inputs)

- Held-out rows: 435593 (395919 normal, 39674 attack)
- Mean reconstruction error: normal 0.0034 (median 0.0009), attack 0.0322 (median 0.0072)
- Chosen threshold: 92th percentile of normal held-out error = 0.00626 (F1-maximizing point of the sweep below; note it is selected on the same held-out set it is reported on, so treat the F1 as slightly optimistic)

## At the chosen threshold

- Accuracy: 0.9220
- Precision: 0.5412
- Recall: 0.9417
- F1: 0.6874
- Flagged rate: 0.1585

## Per-protocol breakdown at the chosen threshold (overall F1 hides this)

| protocol | rows | attack rate | precision | recall | F1 | normal rows flagged |
|---|---|---|---|---|---|---|
| tcp | 269264 | 0.0364 | 0.2172 | 0.8911 | 0.3493 | 0.1214 |
| udp | 166329 | 0.1796 | 0.9938 | 0.9584 | 0.9757 | 0.0013 |

## Threshold sweep (even percentiles + the chosen one)

| percentile | threshold | precision | recall | F1 | flagged rate |
|---|---|---|---|---|---|
| 70 | 0.00164 | 0.2457 | 0.9750 | 0.3925 | 0.3615 |
| 72 | 0.00176 | 0.2585 | 0.9741 | 0.4086 | 0.3432 |
| 74 | 0.00191 | 0.2727 | 0.9729 | 0.4260 | 0.3249 |
| 76 | 0.00208 | 0.2886 | 0.9718 | 0.4451 | 0.3067 |
| 78 | 0.00227 | 0.3064 | 0.9698 | 0.4656 | 0.2883 |
| 80 | 0.00248 | 0.3266 | 0.9681 | 0.4885 | 0.2700 |
| 82 | 0.00275 | 0.3499 | 0.9668 | 0.5138 | 0.2517 |
| 84 | 0.00304 | 0.3766 | 0.9645 | 0.5417 | 0.2333 |
| 86 | 0.00349 | 0.4077 | 0.9616 | 0.5726 | 0.2148 |
| 88 | 0.00419 | 0.4443 | 0.9575 | 0.6069 | 0.1963 |
| 90 | 0.00497 | 0.4881 | 0.9516 | 0.6453 | 0.1776 |
| 92 | 0.00626 | 0.5412 | 0.9417 | 0.6874 | 0.1585 |
| 94 | 0.00866 | 0.2695 | 0.2210 | 0.2428 | 0.0747 |
| 96 | 0.01515 | 0.2976 | 0.1691 | 0.2157 | 0.0518 |
| 98 | 0.02757 | 0.3552 | 0.1099 | 0.1679 | 0.0282 |

Not a like-for-like comparison with the teacher/student models: this is an unsupervised anomaly detector scored against labels it never trained on.
