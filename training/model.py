import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, numeric_dim: int, proto_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        input_dim = numeric_dim + proto_dim
        self.numeric_dim = numeric_dim
        self.proto_dim = proto_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder_trunk = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_numeric = nn.Linear(hidden_dim, numeric_dim)
        self.decoder_proto = nn.Linear(hidden_dim, proto_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        h = self.decoder_trunk(z)
        return self.decoder_numeric(h), self.decoder_proto(h)


def reconstruction_error(
    numeric_recon: torch.Tensor,
    numeric_true: torch.Tensor,
    proto_logits: torch.Tensor,
    proto_true_idx: torch.Tensor,
) -> torch.Tensor:
    numeric_term = ((numeric_recon - numeric_true) ** 2).mean(dim=1)
    proto_term = nn.functional.cross_entropy(proto_logits, proto_true_idx, reduction="none")
    return numeric_term + proto_term
