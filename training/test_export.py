import json

import numpy as np
import torch

from config import REPO_ROOT, TRAINING_DIR, load_config
from dataset import build_dataset, load_raw
from encoding import onehot_proto, standardize_numeric
from export_autoencoder_model import export
from model import Autoencoder, reconstruction_error


def linear(x: np.ndarray, layer: dict) -> np.ndarray:
    weight = np.asarray(layer["weight"], dtype=np.float64)
    bias = np.asarray(layer["bias"], dtype=np.float64)
    y = x @ weight.T + bias
    if layer["activation"] == "relu":
        y = np.maximum(y, 0.0)
    return y


def predict_from_export(payload: dict, x: np.ndarray) -> np.ndarray:
    numeric_dim = len(payload["numeric_features"])
    z = x
    for layer in payload["encoder"]:
        z = linear(z, layer)
    h = z
    for layer in payload["decoder_trunk"]:
        h = linear(h, layer)
    numeric_recon = linear(h, payload["decoder_numeric_head"])
    proto_logits = linear(h, payload["decoder_proto_head"])

    numeric_true = x[:, :numeric_dim]
    proto_true_idx = x[:, numeric_dim:].argmax(axis=1)

    numeric_term = ((numeric_recon - numeric_true) ** 2).mean(axis=1)
    log_probs = proto_logits - np.log(np.exp(proto_logits).sum(axis=1, keepdims=True))
    proto_term = -log_probs[np.arange(len(x)), proto_true_idx]
    return numeric_term + proto_term


def main():
    config = load_config()
    checkpoint_path = TRAINING_DIR / config["paths"]["model_out"]
    out_path = TRAINING_DIR / config["paths"]["ondevice_out"]
    payload = export(checkpoint_path, out_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = Autoencoder(
        numeric_dim=checkpoint["numeric_dim"],
        proto_dim=checkpoint["proto_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        bottleneck_dim=checkpoint["bottleneck_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    raw_dir = REPO_ROOT / config["dataset"]["raw_dir"]
    raw_df = load_raw(raw_dir, config["dataset"]["urls"])
    dataset = build_dataset(raw_df).reset_index(drop=True)
    sample = dataset.sample(n=30, random_state=7)

    mean = np.asarray(checkpoint["numeric_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["numeric_std"], dtype=np.float32)
    numeric = standardize_numeric(sample, mean, std)
    proto = onehot_proto(sample["proto"], checkpoint["proto_categories"])
    x = np.concatenate([numeric, proto], axis=1)

    with torch.no_grad():
        numeric_recon, proto_logits = model(torch.from_numpy(x))
        real_errors = reconstruction_error(
            numeric_recon,
            torch.from_numpy(x[:, : model.numeric_dim]),
            proto_logits,
            torch.from_numpy(x[:, model.numeric_dim :].argmax(axis=1).astype(np.int64)),
        ).numpy()

    exported_errors = predict_from_export(payload, x.astype(np.float64))

    max_diff = float(np.max(np.abs(real_errors - exported_errors)))
    print(json.dumps({"max_diff": max_diff, "real": real_errors.tolist(), "exported": exported_errors.tolist()}, indent=2))

    assert max_diff < 1e-3, f"exported forward pass diverges from the real model by {max_diff}"
    print(f"OK — max reconstruction-error diff across 30 real sampled rows: {max_diff:.8f}")


if __name__ == "__main__":
    main()
