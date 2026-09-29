# Autoencoder — unsupervised anomaly-detection tier (Tier 1)

Trained on normal traffic only (1621865 rows, `label == 0`), never sees an attack label. One shared encoder/bottleneck; decoder heads: linear/MSE over the 10 z-score-standardized numeric features, plus one softmax/cross-entropy head per categorical feature (proto, state). Architecture: input(24) -> hidden(32) -> bottleneck(4) -> hidden(32) -> [numeric(10), proto(7), state(7)], LeakyReLU, 60 epochs, Adam lr=0.001.

Reconstruction error per row = **mean** over numeric features of squared error + sum over categorical heads of per-row cross-entropy (unweighted).

## Encoding

- `dst_port`: `log1p` before standardization.
- Categorical features bucketed to the top 6 values by frequency in the normal-only training split plus `other`:
  - `proto`: tcp, udp, arp, ospf, icmp, igmp, other
  - `state`: FIN, CON, INT, REQ, RST, ECO, other
- `dttl` (destination TTL) and `state` (connection state) were added after a feature sweep — the original 10 features capped F1 at ~0.75 (see `knowledge-graph/noxos-inference/TASKS.md`).

## Held-out evaluation (`StratifiedGroupKFold`, groups = exact feature pattern over all model inputs)

- Held-out rows: 456016 (405466 normal, 50550 attack)
- Mean reconstruction error: normal 0.0020 (median 0.0001), attack 0.1242 (median 0.0028)
- Chosen threshold: 94th percentile of normal held-out error = 0.00235 (F1-maximizing point of the sweep below; note it is selected on the same held-out set it is reported on, so treat the F1 as slightly optimistic)

## At the chosen threshold

- Accuracy: 0.9464
- Precision: 0.6746
- Recall: 0.9978
- F1: 0.8050
- Flagged rate: 0.1640

## Threshold sweep (even percentiles + the chosen one)

| percentile | threshold | precision | recall | F1 | flagged rate |
|---|---|---|---|---|---|
| 70 | 0.00022 | 0.2936 | 1.0000 | 0.4539 | 0.3776 |
| 72 | 0.00023 | 0.3081 | 1.0000 | 0.4710 | 0.3598 |
| 74 | 0.00027 | 0.3241 | 1.0000 | 0.4895 | 0.3420 |
| 76 | 0.00033 | 0.3419 | 0.9999 | 0.5095 | 0.3242 |
| 78 | 0.00039 | 0.3617 | 0.9999 | 0.5312 | 0.3065 |
| 80 | 0.00044 | 0.3840 | 0.9999 | 0.5549 | 0.2887 |
| 82 | 0.00052 | 0.4092 | 0.9999 | 0.5807 | 0.2709 |
| 84 | 0.00063 | 0.4379 | 0.9999 | 0.6091 | 0.2531 |
| 86 | 0.00072 | 0.4710 | 0.9999 | 0.6404 | 0.2353 |
| 88 | 0.00084 | 0.5095 | 0.9999 | 0.6751 | 0.2175 |
| 90 | 0.00096 | 0.5549 | 0.9999 | 0.7137 | 0.1998 |
| 92 | 0.00134 | 0.6091 | 0.9999 | 0.7570 | 0.1820 |
| 94 | 0.00235 | 0.6746 | 0.9978 | 0.8050 | 0.1640 |
| 96 | 0.00481 | 0.1718 | 0.0665 | 0.0959 | 0.0429 |
| 98 | 0.01333 | 0.2304 | 0.0480 | 0.0795 | 0.0231 |

Not a like-for-like comparison with the teacher/student models: this is an unsupervised anomaly detector scored against labels it never trained on.
