import math
from pathlib import Path

import torch
from torch import nn

from .model import Net

DEFAULT_PATH = Path("model.pt")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epochs: int = 2000, lr: float = 1e-2, path: Path = DEFAULT_PATH) -> None:
    device = get_device()
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.linspace(-math.pi, math.pi, 512, device=device).unsqueeze(1)
    y = torch.sin(x)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoche {epoch:5d}/{epochs}  Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), path)
    print(f"Modell gespeichert: {path}")
