# noxos-inference

The Warden threat-analysis inference backend for [NoxOS](https://github.com/parrothacker1/noxos).

**This branch (`main`) holds only CI workflows and this README.** The actual code lives on topic branches — GitHub's `workflow_dispatch` requires a workflow's `.yml` file to exist on the default branch to be manually triggerable at all, but the workflow itself can (and does) check out a different branch to actually run against. `main` is that trigger point, not the codebase.

## Where the code actually is

- **`xgboost-classifier`** — the on-device network-traffic classifier: dataset prep, training (`training/`), the on-device JSON export, the FastAPI service (`service/`), its tests, EC2 deploy scripts (`infra/`). `.github/workflows/train-ondevice-model.yml` (on this branch, `main`) checks this branch out and runs its training pipeline, publishing results as GitHub Releases tagged `ondevice-model-*`.
- **`feature/ml-server-architecture`** — the heavier server-side ML workload, architecture still being figured out. Will get its own workflow on `main` once there's something real to run, with its own distinct release-tag prefix so it never collides with `ondevice-model-*`.

## Workflows on this branch

- **`train-ondevice-model.yml`** (manual `workflow_dispatch` only) — checks out `xgboost-classifier`, trains the network classifier on the full UNSW-NB15 dataset, exports it to the on-device JSON tree format, verifies the export matches the trained model, and publishes both a rolling `ondevice-model-latest` release (what `noxos-app`'s `ModelUpdateManager` actually polls) and a permanent timestamped release per run for history.

See `xgboost-classifier`'s own `README.md` and `CLAUDE.md` for everything else — local dev setup, retraining instructions, the FastAPI service, deployment.
