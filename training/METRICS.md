# Autoencoder trained on MIRAGE-2019 (real Android app traffic, normal only)

Normal-traffic anomaly detector: no attack label is used. One shared encoder/bottleneck, MSE head over the 7 log-transformed, z-scored numeric features and a softmax head for `proto`; per-row error = mean numeric squared error + cross-entropy. Architecture input -> 32 -> 4 -> 32 -> outputs, LeakyReLU, 60 epochs, Adam lr 0.001.

## Data and features

- Source: MIRAGE-2019, the downloadable release: 20 Android apps on 2 devices (the paper describes 40 apps; this public release is a subset). Giuseppe Aceto, Domenico Ciuonzo, Antonio Montieri, Valerio Persico and Antonio Pescapè, "MIRAGE: Mobile-app Traffic Capture and Ground-truth Creation", 4th IEEE International Conference on Computing, Communications and Security (ICCCS 2019). Dataset licence: CC BY-NC-ND 4.0.
- **The licence is non-commercial and no-derivatives. A commercial product must retrain on Warden's own logged traffic.**
- 124950 destination snapshots ({'tcp': 118890, 'udp': 6060}), from 1638 captures and 20 apps.
- One record per destination IP per capture, at a random snapshot time in 0-30 s, aggregating that destination's flows. Per-packet detail exists only for each flow's first 32 packets; beyond that, counts are interpolated linearly from flow totals.
- Flows have no absolute start time, so a destination's flows are assumed to start together (this overcounts if they start later).
- Upstream IP bytes are estimated as payload plus the flow's average header overhead (per-packet IP length is not stored).
- No TCP flags in the data, so **`state` is not a feature** here. `dttl` is not a feature (not capturable on Android).
- Mostly HTTPS over TCP; UDP is thin. Captures are single-app sessions, so 'normal' means these 40 apps.

## Split (by app, so the test measures unseen apps)

- Train 88785, calibration 14400 (com.joelapenna.foursquared, com.spotify.music), test 21765 (com.google.android.youtube, com.iconology.comics, com.tripadvisor.tripadvisor, com.twitter.android).
- Threshold: 99th percentile of calibration-app reconstruction error = 1.90863.

## False-flag rate on held-out normal traffic (unseen apps)

| calibration percentile | all | TCP | UDP |
|---|---|---|---|
| 95 | 0.0637 | 0.0144 | 0.3435 |
| 97 | 0.0123 | 0.0022 | 0.0696 |
| 99 | 0.0001 | 0.0000 | 0.0006 |
| 99.5 | 0.0000 | 0.0000 | 0.0000 |

If the model generalises across apps, the rate should be close to 1 minus the percentile. A larger number means unseen apps look unusual.

### Spread across 5 different app splits (each retrains from scratch; the shipped model is the first)

At the 99th percentile the false-flag rate on unseen apps ranged from 0.0001 to 0.0086 (mean 0.0050); with only 20 apps, which apps are held out matters a lot.

| seed | held-out apps | all | TCP | UDP |
|---|---|---|---|---|
| 42 | com.google.android.youtube, com.iconology.comics, com.tripadvisor.tripadvisor, com.twitter.android | 0.0001 | 0.0000 | 0.0006 |
| 43 | com.google.android.youtube, com.groupon, com.trello, com.waze | 0.0086 | 0.0057 | 0.0198 |
| 44 | com.duolingo, com.groupon, com.iconology.comics, com.trello | 0.0051 | 0.0050 | 0.0139 |
| 45 | com.duolingo, com.joelapenna.foursquared, com.pinterest, com.twitter.android | 0.0037 | 0.0035 | 0.0154 |
| 46 | com.contextlogic.wish, com.trello, de.motain.iliga, it.subito | 0.0077 | 0.0072 | 0.0659 |

## Sanity checks, NOT validation

Synthetic anomaly shapes I defined in feature space (they show the model is not blind to extreme shapes, nothing more):

- syn_scan (tcp, random port, 1-2 packets, no reply): 0.0005 flagged
- udp_flood (udp, thousands of packets, no reply): 0.0425 flagged
- big_upload (tcp 443, tens of thousands of packets up): 0.0000 flagged

UNSW-NB15 rows scored with this model at the same threshold (different network, whole-flow records, so units differ; treat as a rough shift indicator only):

- UNSW-NB15 normal rows: all 0.3707 of 2012839, tcp 0.1697 of 1330535, udp 0.7627 of 682304
- UNSW-NB15 attack rows: all 0.7564 of 214990, tcp 0.1656 of 49030, udp 0.9310 of 165960

## Threshold sensitivity and proxy F1 (added after review; primary model, seed 42)

Error scale on calibration apps: median 0.003, 95th percentile 0.16, 99th 1.91 (heavy tail); held-out apps: median 0.005, 99th 0.57. The 99th-percentile threshold from only 2 calibration apps is therefore far too loose (0.01% of held-out normal flagged, almost no detection).

| calib. pct | threshold | held-out normal flagged all / TCP / UDP | synthetic SYN scan / UDP flood / big upload detected | UNSW proxy F1 all / TCP / UDP | UNSW normal flagged TCP / UDP | UNSW attack flagged TCP / UDP |
|---|---|---|---|---|---|---|
| 90 | 0.065 | 0.142 / 0.055 / 0.635 | 0.949 / 1.000 / 0.538 | 0.177 / 0.053 / 0.327 | 0.862 / 1.000 | 0.666 / 1.000 |
| 93 | 0.101 | 0.096 / 0.028 / 0.478 | 0.878 / 1.000 / 0.275 | 0.178 / 0.048 / 0.329 | 0.818 / 0.989 | 0.567 / 0.998 |
| 95 | 0.160 | 0.064 / 0.014 / 0.344 | 0.779 / 1.000 / 0.112 | 0.188 / 0.050 / 0.331 | 0.715 / 0.973 | 0.526 / 0.992 |
| 97 | 0.497 | 0.012 / 0.002 / 0.070 | 0.126 / 0.983 / 0.000 | 0.213 / 0.051 / 0.350 | 0.532 / 0.873 | 0.403 / 0.974 |
| 99 | 1.909 | 0.0001 / 0.000 / 0.0006 | 0.001 / 0.043 / 0.000 | 0.289 / 0.057 / 0.367 | 0.170 / 0.763 | 0.166 / 0.931 |

How to read this:
- **The UNSW proxy F1 (0.18-0.29 overall, TCP about 0.05, UDP 0.33-0.37 at every threshold) is not a usable measure.** This model finds UNSW *normal* TCP more anomalous (median error 0.572) than UNSW *attack* TCP (0.217), so the proxy is dominated by the difference between two networks and measurement units, not by normal-versus-attack. It does not show the model works and does not show it fails.
- **No F1 above 0.80 can be measured or honestly claimed from these data.** In-domain there are no attacks.
- At the 95th percentile, held-out normal TCP is flagged 1.4% (under 5%), and synthetic SYN-scan-like and UDP-flood-like shapes are flagged 78% and 100%. But held-out normal UDP is flagged 34%, because UDP is only 5% of the data and the threshold is set mostly by TCP; per-protocol thresholds would be needed.
- The synthetic "big upload" shape is essentially not detected at any useful threshold (log-scaling compresses large counts). The synthetic shapes are my own definitions, so this shows blind spots, not real-attack performance.

## Limits

- There is no attack traffic in MIRAGE, so detection ability in this domain is unmeasured.
- Trained on 3 devices and 40 apps; another phone, OS version or app mix may shift the distribution.
- The threshold changes on every retrain; read `reconstruction_threshold` from the model file.
- Superseded once a model trained on Warden-captured traffic exists.
