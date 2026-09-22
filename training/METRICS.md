# Teacher ensemble — out-of-fold metrics

5 XGBoost models (500 trees each, `subsample=0.8`/`colsample_bytree=0.8` — each tree also bags a random row/feature subset on top of the fold split, for extra ensemble diversity), one per `StratifiedGroupKFold(n_splits=5)` fold, grouped by the exact 10-feature pattern per row (same leak-free methodology locked in for the single-split model, see `knowledge-graph/noxos-inference/TASKS.md`). Every row's reported prediction comes from the one fold-model that never saw its feature-pattern during training — this is a real held-out evaluation over the entire 2280083-row dataset, not a single 80/20 split.

- Accuracy: 0.9937
- Precision: 0.9773
- Recall: 0.9654
- F1 (binary, attack=1): 0.9713

`oof_predictions.parquet` holds every row's out-of-fold attack probability — this is the distillation target for the on-device student model (`student-xgboost` branch), not the raw teacher predict_proba on data a fold-model has already seen.
