# noxos-inference

The Warden threat-analysis inference backend for [NoxOS](https://github.com/parrothacker1/noxos).

**This branch (`main`) holds only CI workflows and this README.** The actual code lives on topic branches — GitHub's `workflow_dispatch` requires a workflow's `.yml` file to exist on the default branch to be manually triggerable at all, but the workflow itself can (and does) check out a different branch to actually run against. `main` is that trigger point, not the codebase.

## Where the code actually is

- **`teacher-xgboost`** — the server-side teacher: a 5-fold `StratifiedGroupKFold` XGBoost ensemble on the full UNSW-NB15 dataset, producing leak-free out-of-fold attack-probability predictions used as the distillation target for the student model.
- **`student-xgboost`** — the lightweight on-device student, distilled from the teacher's out-of-fold predictions via regression, exported to the same on-device tree-JSON contract.
- **`teacher-server`** — a Go (gin) serving layer for the tree-JSON model export contract.
- **`filter-autoencoder`** — reserved for a future unsupervised anomaly-detection tier ahead of the classifier. Branch exists, no code yet — deliberately deferred.

## Workflows on this branch

- **`train-teacher.yml`** (manual `workflow_dispatch`) — checks out `teacher-xgboost`, trains the 5-fold teacher ensemble on the full UNSW-NB15 dataset, and publishes the out-of-fold predictions as a rolling `teacher-latest` release (plus a timestamped `teacher-<version>` release for history).
- **`train-student.yml`** (manual `workflow_dispatch`, or automatically via `workflow_run` whenever `train-teacher.yml` completes successfully) — checks out `student-xgboost`, downloads `teacher-latest`'s out-of-fold predictions, distills the student model, exports it to the on-device JSON format, verifies the export against the real model, and publishes a rolling `student-latest` release (plus a timestamped `student-<version>` release) — this is what a model consumer should actually poll.

See each topic branch's own `README.md`/`CLAUDE.md` for local dev setup, retraining instructions, and deployment.
