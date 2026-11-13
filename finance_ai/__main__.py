"""Command line entry point: python -m finance_ai train|predict."""
import argparse
import json

from .experiment import Settings, predict, train


def main():
    parser = argparse.ArgumentParser(description="AI Finance: CPU-only neural network v0.1")
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train", help="Train and evaluate on chronological splits")
    training.add_argument("--excel", required=True, help="MetaTrader 15-minute .xlsx/.xls export")
    training.add_argument("--out", default="models/finance-v0.1")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--patience", type=int, default=12)
    training.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    prediction = actions.add_parser("predict", help="Predict after the latest completed candle")
    prediction.add_argument("--excel", required=True)
    prediction.add_argument("--model", default="models/finance-v0.1")
    args = parser.parse_args()
    if args.action == "train":
        output = train(args.excel, args.out, Settings(epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, patience=args.patience, seeds=tuple(args.seeds)))
    else:
        probability = predict(args.excel, args.model)
        output = {"up_probability_uncalibrated": probability, "predicted_direction": "up" if probability >= .5 else "down"}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
