# Autoencoder trained on MIRAGE-2019 (real Android app traffic, normal only)

Normal-traffic anomaly detector, no attack label used. Shared encoder/bottleneck, MSE head over the 7 transformed, z-scored numeric features and a softmax head for `proto`; per-row error = mean numeric squared error + cross-entropy. Input -> 32 -> 4 -> 32 -> outputs, LeakyReLU, 60 epochs, Adam lr 0.001.

## Data and features

- Source 1: MIRAGE-2019, the downloadable release: 20 Android apps on 2 devices (the paper describes 40 apps; this public release is a subset). Giuseppe Aceto, Domenico Ciuonzo, Antonio Montieri, Valerio Persico and Antonio Pescapè, "MIRAGE: Mobile-app Traffic Capture and Ground-truth Creation", 4th IEEE International Conference on Computing, Communications and Security (ICCCS 2019). Dataset licence: CC BY-NC-ND 4.0.
- **The licence is non-commercial and no-derivatives. A commercial product must retrain on Warden's own logged traffic.**
- 124950 destination snapshots ({'tcp': 118890, 'udp': 6060}) from 1638 captures.
- One record per destination IP per capture, at a random snapshot time in 0-30 s, aggregating that destination's flows. Per-packet detail exists only for each flow's first 32 packets; later counts are interpolated linearly from flow totals. Flows have no absolute start time, so a destination's flows are assumed to start together. Upstream IP bytes are payload plus the flow's average header overhead.
- No TCP flags in the data, so `state` is not a feature; `dttl` is not a feature. Mostly HTTPS over TCP; UDP is thin.
- Input transforms (also written into the export as `numeric_transforms`): dst_port: log1p, src_byte_count: sqrt, src_packet_count: sqrt, dst_byte_count: sqrt, duration_millis: sqrt, handshake_latency_millis: sqrt, smean: sqrt.

## Thresholds: one per protocol, from calibration apps of that protocol

Shipped thresholds: TCP = 97th percentile of TCP calibration error = 0.32267; UDP = 99th percentile of UDP calibration error = 0.12507 (a protocol with fewer than 200 calibration rows falls back to the pooled percentile). UDP uses a looser percentile because its calibration data is thin and UDP false flags are high.

**Shipped configuration, mean (min-max) over the app splits**

- False-flag rate on unseen-app normal traffic: TCP 0.0212 (0.0064-0.0347), UDP 0.0909 (0.0077-0.1944)
- Synthetic shapes detected (sanity only): SYN scan 0.4358 (0.0005-0.7535), UDP flood 0.9924 (0.9815-1.0000), big upload 0.9949 (0.9745-1.0000)

## Held-out normal traffic (apps never seen in training), mean (min-max) over 5 different app splits

False-flag rate; ideal is about 1 minus the percentile.

| calibration percentile | all | TCP | UDP |
|---|---|---|---|
| 95 | 0.0615 (0.0278-0.1257) | 0.0355 (0.0134-0.0568) | 0.2548 (0.0903-0.4562) |
| 97 | 0.0430 (0.0145-0.0918) | 0.0212 (0.0064-0.0347) | 0.1836 (0.0625-0.3553) |
| 99 | 0.0196 (0.0049-0.0408) | 0.0073 (0.0021-0.0129) | 0.0909 (0.0077-0.1944) |

Per split at the shipped percentiles (calibration UDP rows shows how thin the UDP calibration is):

| seed | held-out apps | calibration UDP rows | TCP flagged | UDP flagged |
|---|---|---|---|---|
| 42 | com.google.android.youtube, com.iconology.comics, com.tripadvisor.tripadvisor, com.twitter.android | 807 | 0.0064 | 0.1944 |
| 43 | com.google.android.youtube, com.groupon, com.trello, com.waze | 189 | 0.0242 | 0.1649 |
| 44 | com.duolingo, com.groupon, com.iconology.comics, com.trello | 132 | 0.0274 | 0.0104 |
| 45 | com.duolingo, com.joelapenna.foursquared, com.pinterest, com.twitter.android | 135 | 0.0347 | 0.0077 |
| 46 | com.contextlogic.wish, com.trello, de.motain.iliga, it.subito | 135 | 0.0133 | 0.0769 |

## Synthetic anomaly shapes, sanity only, NOT validation

Shapes I defined in feature space, detected fraction (mean (min-max) over splits):

| calibration percentile | SYN scan (tcp) | UDP flood (udp) | big upload (tcp) |
|---|---|---|---|
| 95 | 0.5771 (0.0070-0.7975) | 0.9998 (0.9990-1.0000) | 0.9987 (0.9935-1.0000) |
| 97 | 0.4358 (0.0005-0.7535) | 0.9998 (0.9990-1.0000) | 0.9949 (0.9745-1.0000) |
| 99 | 0.1643 (0.0000-0.4440) | 0.9924 (0.9815-1.0000) | 0.9877 (0.9495-1.0000) |

## What this does NOT show

- **F1 is not measurable**: MIRAGE has no attack traffic. Detection ability awaits Warden-logged normal traffic plus deliberately generated test traffic.
- The UNSW-NB15 proxy was dropped: it measured the gap between networks, not attacks.
- 20 apps in total; another phone, OS version or app mix may shift the distribution.
- Thresholds change on every retrain; always read them from the model file. Superseded once a model trained on Warden-captured traffic exists.

## UDP threshold recalibration

Weights, features, transforms, categories, error formula and the TCP threshold are unchanged. Only `reconstruction_thresholds.udp` changed: 0.125071 -> 0.429368.

**Rule.** The UDP threshold is the 99th percentile of normal UDP reconstruction error pooled over out-of-fold rows from all 5 app splits (calibration and test rows; every row scored by a model that never trained on its app), with every app weighted equally. Why: the old rule used one split's calibration apps (132-807 UDP rows, threshold ranging 0.125-0.499 across splits), and one app (YouTube) holds 69% of the pooled UDP rows, so a plain percentile mostly describes YouTube. It was chosen by comparing rules on held-out apps, not by fitting to any device's test flows.

**Evaluation**, leave-split-out: for each split the threshold is computed from the other splits' rows excluding the apps in this split's test set, then applied to this split's test apps. Old rule = calibration-only percentile as shipped before.

Old rule (99th of that split's calibration UDP): UDP false-flag 0.0585 (0.0026-0.1944), per-app mean 0.0334 (0.0023-0.0951), UDP flood detected 0.9812 (0.9585-0.9990), DNS-like flows flagged 0.1433 (0.0000-0.4000)

| seed | threshold | UDP false-flag | per-app mean false-flag | UDP flood detected | DNS-like flagged (rows) |
|---|---|---|---|---|---|
| 42 | 0.1251 | 0.1944 | 0.0951 | 0.9990 | 0.250 (20) |
| 43 | 0.4988 | 0.0355 | 0.0093 | 0.9585 | 0.000 (24) |
| 44 | 0.1610 | 0.0417 | 0.0415 | 0.9990 | 0.400 (15) |
| 45 | 0.2305 | 0.0026 | 0.0023 | 0.9705 | 0.067 (15) |
| 46 | 0.4476 | 0.0183 | 0.0189 | 0.9790 | 0.000 (34) |

New rule: UDP false-flag 0.0217 (0.0000-0.0513), per-app mean 0.0163 (0.0000-0.0616), UDP flood detected 0.9463 (0.8625-0.9925), DNS-like flows flagged 0.0613 (0.0000-0.2647)

| seed | threshold | UDP false-flag | per-app mean false-flag | UDP flood detected | DNS-like flagged (rows) |
|---|---|---|---|---|---|
| 42 | 0.4463 | 0.0083 | 0.0022 | 0.9460 | 0.000 (20) |
| 43 | 0.3873 | 0.0490 | 0.0180 | 0.9825 | 0.042 (24) |
| 44 | 0.4892 | 0.0000 | 0.0000 | 0.9480 | 0.000 (15) |
| 45 | 0.4562 | 0.0000 | 0.0000 | 0.8625 | 0.000 (15) |
| 46 | 0.3193 | 0.0513 | 0.0616 | 0.9925 | 0.265 (34) |

DNS-like = UDP snapshots with at most 3 source packets, under 400 source bytes and under 2000 destination bytes. MIRAGE has no UDP traffic on port 53, so this is a shape proxy (mostly Spotify discovery on 57621, STUN on 3478, DHCP), not real DNS.

Shipped model on its own split-42 UDP rows (calibration and test apps), old vs new threshold: flagged 0.1581 -> 0.0069; DNS-like flows flagged 0.030 -> 0.000 (370 rows).

**Remaining limits.** F1 is still unmeasured (MIRAGE has no attacks). A rate of a few percent on unseen apps remains and is not zero. The pooled errors come from the 5 retrained split models, not the shipped weights themselves. Real DNS (port 53) is absent from the training data. The cost of the higher threshold is UDP flood sensitivity: 94.6% (86.3-99.3%) detected against 98.1% (95.9-99.9%) before; smaller UDP anomalies sitting between the old and new threshold are no longer flagged.
