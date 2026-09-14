# Network classifier — held-out metrics

Trained on UNSW-NB15 (`UNSW_NB15_training-set.csv`, 175341 rows), 80/20 stratified split, `random_state=42`.

- Accuracy: 0.9575
- F1 (binary, attack=1): 0.9690

Regenerate with `uv run training/train_network_model.py` after fetching the dataset per `config.toml`'s `[dataset]` section (see `training/FEATURES.md`).
