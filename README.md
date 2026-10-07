# decoder

Einfaches PyTorch-Projekt: Ein kleines neuronales Netz (MLP) lernt die Funktion `sin(x)`.

## Verwendung

```sh
uv sync
uv run decoder train --epochs 2000     # trainiert und speichert model.pt
uv run decoder run 0 1.5708 3.14159    # führt das Modell aus
```

## Struktur

- `src/decoder/model.py` – Modelldefinition
- `src/decoder/train.py` – Trainingsschleife
- `src/decoder/predict.py` – Modell laden und ausführen
