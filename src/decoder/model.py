import torch
from torch import nn


class Net(nn.Module):
    """Kleines MLP, das y = sin(x) approximiert."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
