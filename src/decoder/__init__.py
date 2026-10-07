import argparse
from pathlib import Path

from .train import DEFAULT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decoder", description="Einfaches PyTorch-Modell: lernt sin(x)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Modell trainieren und speichern")
    p_train.add_argument("--epochs", type=int, default=2000)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--model", type=Path, default=DEFAULT_PATH)

    p_run = sub.add_parser("run", help="Trainiertes Modell ausführen")
    p_run.add_argument("values", type=float, nargs="+", help="Eingabewerte x")
    p_run.add_argument("--model", type=Path, default=DEFAULT_PATH)

    args = parser.parse_args()
    if args.command == "train":
        from .train import train

        train(args.epochs, args.lr, args.model)
    else:
        from .predict import predict

        predict(args.values, args.model)
