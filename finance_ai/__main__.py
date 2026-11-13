"""Command line entry point: train, compare-horizons, or predict."""
import argparse
import json
from pathlib import Path

from .experiment import Settings, predict, train
from .horizons import compare_horizons
from .architectures import compare_architectures


def main():
    parser = argparse.ArgumentParser(description="AI Finance: CPU-only neural network v0.4")
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train", help="Train and evaluate on chronological splits")
    training.add_argument("--data", "--csv", "--excel", dest="data", required=True, help="MetaTrader M15 .csv/.xlsx/.xls export")
    training.add_argument("--out", default="models/finance-v0.2")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--patience", type=int, default=12)
    training.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    comparison = actions.add_parser("compare-horizons", help="Compare targets at 1, 4 and 8 candles using validation only")
    comparison.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    comparison.add_argument("--out", default="models/finance-v0.3")
    comparison.add_argument("--epochs", type=int, default=100)
    comparison.add_argument("--batch-size", type=int, default=64)
    comparison.add_argument("--lr", type=float, default=0.001)
    comparison.add_argument("--patience", type=int, default=12)
    comparison.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    architectures = actions.add_parser("compare-architectures", help="Compare MLP sizes for one M15 candle using validation only")
    architectures.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    architectures.add_argument("--out", default="models/finance-v0.4")
    architectures.add_argument("--epochs", type=int, default=100)
    architectures.add_argument("--batch-size", type=int, default=64)
    architectures.add_argument("--lr", type=float, default=0.001)
    architectures.add_argument("--patience", type=int, default=12)
    architectures.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    prediction = actions.add_parser("predict", help="Predict after the latest completed candle")
    prediction.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    prediction.add_argument("--model", default="models/finance-v0.2")
    args = parser.parse_args()
    if args.action == "train":
        output = train(args.data, args.out, Settings(epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, patience=args.patience, seeds=tuple(args.seeds)))
    elif args.action == "compare-horizons":
        output = compare_horizons(args.data, args.out,
                Settings(epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr, patience=args.patience, seeds=tuple(args.seeds)))
    elif args.action == "compare-architectures":
        output = compare_architectures(args.data, args.out,
                Settings(epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr, patience=args.patience, seeds=tuple(args.seeds)))
    else:
        probability = predict(args.data, args.model)
        output = {"up_probability_uncalibrated": probability, "predicted_direction": "up" if probability >= .5 else "down"}
        config = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
        output["horizon_candles"] = int(config.get("horizon_candles", 1))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
