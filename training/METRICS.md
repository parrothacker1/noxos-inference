# File classifier (Drebin-215, APK permission features)

XGBoost (300 trees, depth 6), trained on the public Drebin-215 feature CSV: 15031 apps (5555 malware / 9476 benign), 7256 distinct feature patterns. Evaluated with `StratifiedGroupKFold` (5-fold, out-of-fold over every row) grouped by exact feature pattern within each feature set, so no pattern is in both train and held-out.

| feature set | columns | precision | recall | F1 | AUC |
|---|---|---|---|---|---|
| permissions | 113 | 0.9496 | 0.8927 | 0.9203 | 0.9801 |
| permissions+intents | 136 | 0.9563 | 0.8940 | 0.9241 | 0.9827 |
| all_215 (not deployable: API calls need DEX analysis) | 215 | 0.9852 | 0.9685 | 0.9768 | 0.9977 |
| baseline: payload-style rule (>=4 high-risk permissions) | 12 | 0.5882 | 0.1627 | 0.2549 | n/a |

Deployed feature set: `permissions` (the only one `noxos-payload` can extract today).

## Most important features in the deployed model

- `SEND_SMS`: 0.1455
- `READ_HISTORY_BOOKMARKS`: 0.0536
- `GET_ACCOUNTS`: 0.0530
- `MANAGE_ACCOUNTS`: 0.0502
- `READ_PHONE_STATE`: 0.0404
- `WRITE_HISTORY_BOOKMARKS`: 0.0347
- `ACCESS_LOCATION_EXTRA_COMMANDS`: 0.0311
- `AUTHENTICATE_ACCOUNTS`: 0.0257
- `RECEIVE_SMS`: 0.0246
- `WRITE_CALL_LOG`: 0.0236
- `USE_CREDENTIALS`: 0.0223
- `READ_SMS`: 0.0212

Caveats: Drebin is a 2010-2012 era dataset, so a modern benign app's permission set may sit outside its distribution; the public CSV has 9,476 benign apps, not the 123K in the original paper. F1 here is on this dataset only, not on captured Warden traffic.
