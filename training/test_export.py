import json

import numpy as np
import pandas as pd
import torch

from config import REPO_ROOT, TRAINING_DIR, load_config
from encoding import encode
from export_autoencoder_model import export, load_model
from model import reconstruction_error, split_targets


def linear(x: np.ndarray, layer: dict, negative_slope: float) -> np.ndarray:
    y = x @ np.asarray(layer["weight"], dtype=np.float64).T + np.asarray(layer["bias"], dtype=np.float64)
    if layer["activation"] == "leaky_relu":
        y = np.where(y >= 0, y, y * negative_slope)
    return y


def encode_from_export(payload: dict, df: pd.DataFrame) -> np.ndarray:
    columns = payload["numeric_features"]
    numeric = df[columns].to_numpy(dtype=np.float64).copy()
    for i, column in enumerate(columns):
        kind = payload["numeric_transforms"][column]
        if kind == "log1p":
            numeric[:, i] = np.log1p(np.clip(numeric[:, i], 0, None))
        elif kind == "sqrt":
            numeric[:, i] = np.sqrt(np.clip(numeric[:, i], 0, None))
    numeric = (numeric - np.asarray(payload["numeric_mean"])) / np.asarray(payload["numeric_std"])
    parts = [numeric]
    for feature in payload["categorical_features"]:
        categories = feature["categories"]
        codes = [categories.index(v) if v in categories else categories.index("other") for v in df[feature["name"]]]
        onehot = np.zeros((len(df), len(categories)))
        onehot[np.arange(len(df)), codes] = 1.0
        parts.append(onehot)
    return np.concatenate(parts, axis=1)


def predict_from_export(payload: dict, x: np.ndarray) -> np.ndarray:
    slope = payload["leaky_relu_negative_slope"]
    numeric_dim = len(payload["numeric_features"])
    z = x
    for layer in payload["encoder"]:
        z = linear(z, layer, slope)
    h = z
    for layer in payload["decoder_trunk"]:
        h = linear(h, layer, slope)

    numeric_recon = linear(h, payload["decoder_numeric_head"], slope)
    error = ((numeric_recon - x[:, :numeric_dim]) ** 2).mean(axis=1)

    offset = numeric_dim
    for feature, head in zip(payload["categorical_features"], payload["decoder_categorical_heads"]):
        width = len(feature["categories"])
        true_idx = x[:, offset : offset + width].argmax(axis=1)
        logits = linear(h, head, slope)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        error = error - log_probs[np.arange(len(x)), true_idx]
        offset += width
    return error


def main():
    config = load_config()
    checkpoint_path = TRAINING_DIR / config["paths"]["model_out"]
    out_path = TRAINING_DIR / config["paths"]["ondevice_out"]
    payload = export(checkpoint_path, out_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = load_model(checkpoint)

    records = pd.read_parquet(REPO_ROOT / config["mirage"]["records"])
    sample = records.sample(n=200, random_state=7)

    mean = np.asarray(checkpoint["numeric_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["numeric_std"], dtype=np.float32)
    x = encode(sample, mean, std, checkpoint["categories"])

    with torch.no_grad():
        x_t = torch.from_numpy(x)
        numeric_recon, logits = model(x_t)
        numeric_true, idx = split_targets(x_t, model.numeric_dim, model.categorical_dims)
        real_errors = reconstruction_error(numeric_recon, numeric_true, logits, idx).numpy()

    x_from_json = encode_from_export(payload, sample)
    encode_diff = float(np.max(np.abs(x_from_json - x)))
    assert encode_diff < 1e-3, f"encoding rebuilt from the exported JSON alone differs from training by {encode_diff}"

    exported_errors = predict_from_export(payload, x_from_json)
    max_diff = float(np.max(np.abs(real_errors - exported_errors)))
    print(json.dumps({"max_diff": max_diff, "encode_diff_from_json_only": encode_diff}))

    assert max_diff < 1e-3, f"exported forward pass diverges from the real model by {max_diff}"
    print(f"OK — max reconstruction-error diff across 200 real sampled rows: {max_diff:.8f}")


if __name__ == "__main__":
    main()
