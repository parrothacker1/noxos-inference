import json

import numpy as np
import torch

from config import TRAINING_DIR, load_config
from model import Autoencoder


def linear_layer(layer: torch.nn.Linear) -> dict:
    return {
        "weight": layer.weight.detach().numpy().astype(np.float64).tolist(),
        "bias": layer.bias.detach().numpy().astype(np.float64).tolist(),
    }


def load_model(checkpoint: dict) -> Autoencoder:
    model = Autoencoder(
        numeric_dim=checkpoint["numeric_dim"],
        categorical_dims=checkpoint["categorical_dims"],
        hidden_dim=checkpoint["hidden_dim"],
        bottleneck_dim=checkpoint["bottleneck_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def thresholds_from(checkpoint: dict) -> dict:
    if "reconstruction_thresholds" in checkpoint:
        return {proto: float(t) for proto, t in checkpoint["reconstruction_thresholds"].items()}
    single = float(checkpoint["reconstruction_threshold"])
    return {"tcp": single, "udp": single}


def export(checkpoint_path, out_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = load_model(checkpoint)

    payload = {
        "numeric_features": checkpoint["numeric_columns"],
        "numeric_mean": np.asarray(checkpoint["numeric_mean"], dtype=np.float64).tolist(),
        "numeric_std": np.asarray(checkpoint["numeric_std"], dtype=np.float64).tolist(),
        "numeric_reduction": "mean",
        "categorical_features": [
            {"name": name, "categories": checkpoint["categories"][name]}
            for name in checkpoint["categorical_columns"]
        ],
        "numeric_transforms": checkpoint["numeric_transforms"],
        "reconstruction_thresholds": thresholds_from(checkpoint),
        "encoder": [
            {**linear_layer(model.encoder[0]), "activation": "leaky_relu"},
            {**linear_layer(model.encoder[2]), "activation": "linear"},
        ],
        "decoder_trunk": [
            {**linear_layer(model.decoder_trunk[0]), "activation": "leaky_relu"},
        ],
        "decoder_numeric_head": {**linear_layer(model.decoder_numeric), "activation": "linear"},
        "decoder_categorical_heads": [
            {"name": name, **linear_layer(head), "activation": "linear"}
            for name, head in zip(checkpoint["categorical_columns"], model.decoder_categorical)
        ],
        "leaky_relu_negative_slope": 0.01,
    }

    out_path.write_text(json.dumps(payload))
    return payload


def main():
    config = load_config()
    checkpoint_path = TRAINING_DIR / config["paths"]["model_out"]
    out_path = TRAINING_DIR / config["paths"]["ondevice_out"]
    export(checkpoint_path, out_path)
    print(f"exported {out_path}")


if __name__ == "__main__":
    main()
