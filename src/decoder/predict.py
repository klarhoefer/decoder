from pathlib import Path

import torch

from .model import Net
from .train import DEFAULT_PATH, get_device


def predict(values: list[float], path: Path = DEFAULT_PATH) -> None:
    device = get_device()
    model = Net().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    x = torch.tensor(values, device=device).unsqueeze(1)
    with torch.no_grad():
        y = model(x).squeeze(1)

    for xi, yi in zip(values, y.tolist()):
        print(f"f({xi:g}) = {yi:.4f}")
