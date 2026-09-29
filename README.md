# noxos-inference

The Warden threat-analysis inference backend for [NoxOS](https://github.com/parrothacker1/noxos).

**This branch (`main`) holds only CI workflows and this README.** The actual code lives on topic branches — GitHub's `workflow_dispatch` requires a workflow's `.yml` file to exist on the default branch to be manually triggerable at all, but the workflow itself can (and does) check out a different branch to actually run against. `main` is that trigger point, not the codebase.

## Where the code actually is

- **`teacher-xgboost`** — the server-side teacher: a 5-fold `StratifiedGroupKFold` XGBoost ensemble on the full UNSW-NB15 dataset, producing leak-free out-of-fold attack-probability predictions used as the distillation target for the student model.
- **`student-xgboost`** — the lightweight on-device student, distilled from the teacher's out-of-fold predictions via regression, exported to the same on-device tree-JSON contract.
- **`teacher-server`** — a Go (gin) serving layer for the tree-JSON model export contract.
- **`filter-autoencoder`** — the on-device Tier 1 filter: an unsupervised autoencoder trained on normal traffic only, flagging destinations whose reconstruction error is anomalous before they ever reach the student/teacher classifiers.

- **`file-classifier`** — the APK file classifier: XGBoost on the public Drebin permission features, served by `teacher-server`'s `/analyze/file`. **Advisory only** (2012-era vocabulary, unvalidated on modern APKs); it must not gate quarantine or any verdict.

## Workflows on this branch

- **`train-teacher.yml`** (manual `workflow_dispatch`) — checks out `teacher-xgboost`, trains the 5-fold teacher ensemble on the full UNSW-NB15 dataset (out-of-fold predictions, the student's distillation target), *and* trains a separate full-data production teacher exported to the on-device JSON contract — this is the real servable model `teacher-server` dynamically loads. Publishes both as a rolling `teacher-latest` release (plus a timestamped `teacher-<version>` release for history).
- **`train-student.yml`** (manual `workflow_dispatch`, or automatically via `workflow_run` whenever `train-teacher.yml` completes successfully) — checks out `student-xgboost`, downloads `teacher-latest`'s out-of-fold predictions, distills the student model, exports it to the on-device JSON format, verifies the export against the real model, and publishes a rolling `student-latest` release (plus a timestamped `student-<version>` release) — this is what a model consumer should actually poll.
- **`train-file.yml`** (manual `workflow_dispatch`) — checks out `file-classifier`, fetches the SHA256-pinned Drebin feature files, trains and evaluates with grouped cross-validation, verifies the export against the real model, and publishes a rolling `file-latest` release (plus `file-<version>`), marked advisory-only in its notes.
- **`train-autoencoder.yml`** (manual `workflow_dispatch`) — checks out `filter-autoencoder`, trains the Tier 1 autoencoder on the full UNSW-NB15 dataset's normal-traffic rows only, exports it to a dense-layer on-device JSON contract (distinct from the tree-JSON contract above — this is a small MLP, not a tree ensemble), verifies the export against the real model, and publishes a rolling `autoencoder-latest` release (plus a timestamped `autoencoder-<version>` release).

See each topic branch's own `README.md`/`CLAUDE.md` for local dev setup, retraining instructions, and deployment.
