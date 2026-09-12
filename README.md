# noxos-inference

The Warden threat-analysis inference backend for [NoxOS](https://github.com/parrothacker1/noxos). Warden (`noxos-app`) routes newly-seen network destinations and arriving files through a local ACL cost-gate first — only a genuinely new, ambiguous destination or file ever reaches this service, once, ever (see `AclRepository.nextAnalysisBatch` in `noxos-app`).

Two independent problems, two different tools, not one model wearing two hats:

- **Network traffic** (`/analyze/network`) — a gradient-boosted classifier (XGBoost/LightGBM) trained on flow metadata (destination IP/port, protocol, volume, frequency), not an LLM. See `training/`.
- **Files** (`/analyze/file`) — ClamAV, signature-based. Most ML malware-classifier research targets Windows PE executables; Warden receives APKs/images/docs, a poor fit for that research. See `service/clamav_client.py`.

Never sees raw packet payload or file contents beyond what ClamAV needs — everything on-device is TLS today, so payload inspection buys nothing without MITM, which is explicitly out of scope for this OS's trust model.

## Layout

- `training/` — dataset prep + model training, produces the artifact `service/` serves.
- `service/` — FastAPI app: `POST /analyze/network`, `POST /analyze/file`.
- `infra/` — EC2 deploy scripts for the self-hosted inference box (`us-east-1`, no IAM role needed — see `infra/README.md`).
- `tests/` — real tests against the trained model and the live service, not decorative ones.

## Run locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn service.app:app --reload
```

See `training/` for how to regenerate the model artifact, and `infra/README.md` for deploying this for real.
