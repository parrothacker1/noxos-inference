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


def export(checkpoint_path, out_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = Autoencoder(
        numeric_dim=checkpoint["numeric_dim"],
        proto_dim=checkpoint["proto_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        bottleneck_dim=checkpoint["bottleneck_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    payload = {
        "numeric_features": checkpoint["numeric_columns"],
        "proto_categories": checkpoint["proto_categories"],
        "numeric_mean": np.asarray(checkpoint["numeric_mean"], dtype=np.float64).tolist(),
        "numeric_std": np.asarray(checkpoint["numeric_std"], dtype=np.float64).tolist(),
        "reconstruction_threshold": float(checkpoint["reconstruction_threshold"]),
        "encoder": [
            {**linear_layer(model.encoder[0]), "activation": "relu"},
            {**linear_layer(model.encoder[2]), "activation": "linear"},
        ],
        "decoder_trunk": [
            {**linear_layer(model.decoder_trunk[0]), "activation": "relu"},
        ],
        "decoder_numeric_head": {**linear_layer(model.decoder_numeric), "activation": "linear"},
        "decoder_proto_head": {**linear_layer(model.decoder_proto), "activation": "linear"},
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
