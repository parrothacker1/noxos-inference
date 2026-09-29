import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, numeric_dim: int, categorical_dims: list[int], hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        input_dim = numeric_dim + sum(categorical_dims)
        self.numeric_dim = numeric_dim
        self.categorical_dims = list(categorical_dims)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder_trunk = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.decoder_numeric = nn.Linear(hidden_dim, numeric_dim)
        self.decoder_categorical = nn.ModuleList([nn.Linear(hidden_dim, d) for d in categorical_dims])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z = self.encoder(x)
        h = self.decoder_trunk(z)
        return self.decoder_numeric(h), [head(h) for head in self.decoder_categorical]


def split_targets(x: torch.Tensor, numeric_dim: int, categorical_dims: list[int]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    numeric_true = x[:, :numeric_dim]
    indices = []
    offset = numeric_dim
    for d in categorical_dims:
        indices.append(x[:, offset : offset + d].argmax(dim=1))
        offset += d
    return numeric_true, indices


def reconstruction_error(
    numeric_recon: torch.Tensor,
    numeric_true: torch.Tensor,
    categorical_logits: list[torch.Tensor],
    categorical_true_idx: list[torch.Tensor],
) -> torch.Tensor:
    error = ((numeric_recon - numeric_true) ** 2).mean(dim=1)
    for logits, idx in zip(categorical_logits, categorical_true_idx):
        error = error + nn.functional.cross_entropy(logits, idx, reduction="none")
    return error
