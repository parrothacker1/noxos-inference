# Network classifier — held-out metrics

Trained on the full raw UNSW-NB15 dataset (2280083 rows after cleaning), 10-feature set locked in `knowledge-graph/noxos-inference/TASKS.md` ("Real training spec, LOCKED 2026-09-14"), 80/20 stratified split, `random_state=42`.

- Accuracy: 0.9939
- F1 (binary, attack=1): 0.9724

Regenerate with `uv run training/train_network_model.py` after fetching the dataset per `config.toml`'s `[dataset]` section (see `training/FEATURES.md`).
