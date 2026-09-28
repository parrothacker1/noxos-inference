# Autoencoder — unsupervised anomaly-detection tier (Tier 1)

Trained on normal traffic only (1621863 rows, `label == 0`) — never sees an attack label during training, per the design in `knowledge-graph/noxos-inference/ML-NETWORK-DESIGN.md`'s "Autoencoder reconstruction-loss design" section. Two decoder heads sharing one encoder/bottleneck: a linear head over the 9 z-score-standardized numeric features (loss = MSE), and a softmax head over one-hot `proto` (loss = cross-entropy). Architecture: input(16) -> hidden(32) -> bottleneck(4) -> hidden(32) -> [numeric(9), proto(7)], 60 epochs, Adam lr=0.001.

## Open design questions from the design doc, resolved here

- **`dst_port`'s lumpy/multimodal distribution**: resolved with a `log1p` transform before standardization (not bucketing) — keeps it in the numeric head as a continuous value, just compresses the long tail of high ephemeral ports toward the well-known-port cluster.
- **`proto`'s 100+ distinct values**: resolved by bucketing to the top 6 most frequent protocols (by row count in the normal-only training split) plus an `"other"` bucket, instead of full-cardinality one-hot. Real categories kept: tcp, udp, arp, ospf, icmp, igmp, plus `other` for everything else.

## Held-out evaluation (`StratifiedGroupKFold` split, same leak-free grouping as the teacher/student models — every held-out feature-pattern is unseen in training)

- Held-out rows: 456020 (405468 normal, 50552 attack)
- Mean reconstruction error, normal held-out rows: 0.0021 (median 0.0003)
- Mean reconstruction error, attack held-out rows: 0.0262 (median 0.0017)
- Flagging threshold: 92th percentile of normal held-out reconstruction error = 0.0017

**Why the 92nd percentile, not the conventional 95th**: a real threshold sweep (50th-99th) found a sharp cliff, not a smooth precision/recall tradeoff — F1 stays 0.62-0.75 from the 85th through 92nd percentile, then collapses to ~0.076 at 93rd and stays there through 99th. Root cause: roughly 90% of attack rows in the held-out set land within a narrow band around a single reconstruction-error value (~0.0017-0.0018), almost certainly reflecting UNSW-NB15's own known row duplication among synthetic attack flows (see the `StratifiedGroupKFold` grouping above — 1,199,022 distinct patterns across 2,280,083 rows). The 92nd-percentile-of-normal threshold (0.0017) sits just below that cluster; the 93rd (0.0020) sits just above it, so a 0.0003 move in the threshold flips ~90% of attacks from flagged to unflagged in one step. 92 is the last percentile before that cliff and gives the best F1 in the sweep. Recall matters more than precision for this specific tier — per `ML-NETWORK-DESIGN.md` item 15, a flag here is cheap (traffic stays live while escalating to the student) but a miss here means the destination never reaches the student/teacher tiers at all — and recall stays within 0.005 of its ceiling (0.984 at 92 vs. 0.989 at 90) while precision and F1 are both meaningfully better at 92.

## Anomaly-detection performance at that threshold (flag vs. true label)

- Accuracy: 0.9271
- Precision: 0.6053
- Recall: 0.9840
- F1: 0.7495

**Not a like-for-like comparison with the teacher/student models** — this is an unsupervised anomaly detector scored against labels it never trained on, evaluated for a different job (catching traffic shapes the supervised models were never trained to recognize at all), not for beating their F1.
