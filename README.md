# noxos-inference

The Warden threat-analysis inference backend for [NoxOS](https://github.com/parrothacker1/noxos). Warden (`noxos-app`) routes newly-seen network destinations and arriving files through a local ACL cost-gate first — only a genuinely new, ambiguous destination or file ever reaches this service, once, ever (see `AclRepository.nextAnalysisBatch` in `noxos-app`).

Two independent problems, two different tools, not one model wearing two hats:

- **Network traffic** (`/analyze/network`) — a gradient-boosted classifier (XGBoost/LightGBM) trained on flow metadata (destination IP/port, protocol, volume, frequency), not an LLM. See `training/`.
- **Files** (`/analyze/file`) — ClamAV, signature-based. Most ML malware-classifier research targets Windows PE executables; Warden receives APKs/images/docs, a poor fit for that research. See `service/clamav_client.py`.

Never sees raw packet payload or file contents beyond what ClamAV needs — everything on-device is TLS today, so payload inspection buys nothing without MITM, which is explicitly out of scope for this OS's trust model.

## Layout

- `config.toml` — dataset URL, training hyperparameters, verdict thresholds, output paths. Single source of truth for both the training script and the CI workflow.
- `training/` — dataset prep + model training, produces the server-side artifact `service/` serves and (`export_ondevice_model.py`) the on-device JSON tree export `noxos-app`'s `OnDeviceNetworkClassifier` consumes.
- `service/` — FastAPI app: `POST /analyze/network`, `POST /analyze/file`.
- `infra/` — EC2 deploy scripts for the self-hosted inference box (`us-east-1`, no IAM role needed — see `infra/README.md`).
- `tests/` — real tests against the trained model and the live service, not decorative ones.
- `.github/workflows/train-ondevice-model.yml` — trains and publishes the on-device model as a GitHub Release (manual trigger). See "On-device model releases" below.

Package management: [`uv`](https://docs.astral.sh/uv/), not pip/poetry. `pyproject.toml` + `uv.lock` are the source of truth for dependencies.

## Run locally

```bash
uv sync
uv run uvicorn service.app:app --reload
```

To retrain and re-export the on-device model:

```bash
mkdir -p data/raw
for url in $(uv run python -c "import tomllib; print('\n'.join(tomllib.load(open('config.toml','rb'))['dataset']['urls']))"); do
  curl -sL "$url" -o "data/raw/$(basename "$url")"
done
uv run training/train_network_model.py
uv run training/export_ondevice_model.py
uv run tests/test_export_matches_model.py
```

## On-device model releases

`.github/workflows/train-ondevice-model.yml` (manual `workflow_dispatch` trigger) trains, exports, and publishes the on-device model as GitHub Releases — a rolling `ondevice-model-latest` release (assets always overwritten, the one `noxos-app` actually polls) plus a permanent `ondevice-model-<unix-timestamp>` release per run for history. Assets: `model.json` (the tree JSON `OnDeviceNetworkClassifier` loads), `manifest.json` (`{version, sha256, modelUrl}` — the schema `noxos-app`'s `ModelUpdateManager` expects), `model.json.sha256`, `METRICS.md`.

**Deliberately separate from any future `ml-server-*` release scheme** — see `feature/ml-server-architecture` branch and this repo's `TASKS.md` — so the two don't collide or get confused later.

The raw dataset is never committed or pushed anywhere (`data/raw/` gitignored) — every training run, local or CI, fetches it fresh from the URL in `config.toml`.

See `training/` for how to regenerate the server-side model artifact, and `infra/README.md` for deploying the FastAPI service for real.
