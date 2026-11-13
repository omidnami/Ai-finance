"""Command line entry point: binary and ATR three-class research."""
import argparse
import json
from pathlib import Path

from .experiment import Settings, predict, train
from .horizons import compare_horizons
from .architectures import compare_architectures
from .atr import train_atr, predict_atr, DEFAULT_HIDDEN, DEFAULT_ATR_MULTIPLIER
from .trend_detection import train_trend, detect_trend, DEFAULT_HIDDEN as TREND_HIDDEN
from .trend_chart import plot_validation_trend
from .reversal_detection import train_reversal, detect_reversal, DEFAULT_HIDDEN as REVERSAL_HIDDEN
from .reversal_chart import plot_validation_reversals


def main():
    parser = argparse.ArgumentParser(description="AI Finance: CPU-only neural network v0.8")
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
    architectures.add_argument("--out", default="models/finance-v0.5")
    architectures.add_argument("--epochs", type=int, default=100)
    architectures.add_argument("--batch-size", type=int, default=64)
    architectures.add_argument("--lr", type=float, default=0.001)
    architectures.add_argument("--patience", type=int, default=12)
    architectures.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    atr_train = actions.add_parser("train-atr", help="Train a larger 3-class MLP using ATR-neutral labels")
    atr_train.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    atr_train.add_argument("--out", default="models/finance-v0.6")
    atr_train.add_argument("--epochs", type=int, default=100)
    atr_train.add_argument("--batch-size", type=int, default=64)
    atr_train.add_argument("--lr", type=float, default=0.001)
    atr_train.add_argument("--patience", type=int, default=12)
    atr_train.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    atr_train.add_argument("--hidden", type=int, nargs="+", default=list(DEFAULT_HIDDEN),
                           help="Hidden widths, e.g. 512 256 128 64 (larger needs more CPU/RAM)")
    atr_train.add_argument("--atr-multiplier", type=float, default=DEFAULT_ATR_MULTIPLIER,
                           help="Neutral band as fraction of ATR14 at candle t")
    trend_training = actions.add_parser("train-trend", help="Learn current up/down trend from past-only price reference")
    trend_training.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    trend_training.add_argument("--out", default="models/finance-v0.7")
    trend_training.add_argument("--epochs", type=int, default=100)
    trend_training.add_argument("--batch-size", type=int, default=64)
    trend_training.add_argument("--lr", type=float, default=0.001)
    trend_training.add_argument("--patience", type=int, default=12)
    trend_training.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    trend_training.add_argument("--hidden", type=int, nargs="+", default=list(TREND_HIDDEN))
    trend_detection = actions.add_parser("detect-trend", help="Recognize current trend using completed candles")
    trend_detection.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    trend_detection.add_argument("--model", default="models/finance-v0.7")
    trend_chart = actions.add_parser("plot-trend", help="Plot last N validation candles with v0.7 up/down trend bands")
    trend_chart.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    trend_chart.add_argument("--model", default="models/finance-v0.7")
    trend_chart.add_argument("--out", default="charts/finance-v0.7-validation-500.png")
    trend_chart.add_argument("--candles", type=int, default=500)
    rev_training = actions.add_parser("train-reversal", help="Learn past-only confirmed swing reversal candidates")
    rev_training.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    rev_training.add_argument("--out", default="models/finance-v0.8")
    rev_training.add_argument("--epochs", type=int, default=100)
    rev_training.add_argument("--batch-size", type=int, default=64)
    rev_training.add_argument("--lr", type=float, default=0.001)
    rev_training.add_argument("--patience", type=int, default=12)
    rev_training.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 97])
    rev_training.add_argument("--hidden", type=int, nargs="+", default=list(REVERSAL_HIDDEN))
    rev_detect = actions.add_parser("detect-reversal", help="Detect confirmed reversal candidate on last closed M15 candle")
    rev_detect.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    rev_detect.add_argument("--model", default="models/finance-v0.8")
    rev_plot = actions.add_parser("plot-reversals", help="Plot v0.8 model reversal markers on validation candles")
    rev_plot.add_argument("--data", "--csv", "--excel", dest="data", required=True)
    rev_plot.add_argument("--model", default="models/finance-v0.8")
    rev_plot.add_argument("--out", default="charts/finance-v0.8-validation-500.png")
    rev_plot.add_argument("--candles", type=int, default=500)
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
    elif args.action == "train-atr":
        output = train_atr(args.data, args.out,
                           Settings(epochs=args.epochs, batch_size=args.batch_size,
                                    lr=args.lr, patience=args.patience,
                                    seeds=tuple(args.seeds)),
                           hidden=tuple(args.hidden), atr_multiplier=args.atr_multiplier,
                           include_macd=False)
    elif args.action == "train-trend":
        output = train_trend(args.data, args.out,
                           Settings(epochs=args.epochs, batch_size=args.batch_size,
                                    lr=args.lr, patience=args.patience,
                                    seeds=tuple(args.seeds)),
                           hidden=tuple(args.hidden))
    elif args.action == "detect-trend":
        output = detect_trend(args.data, args.model)
    elif args.action == "plot-trend":
        output = plot_validation_trend(args.data, args.model, args.out, args.candles)
    elif args.action == "train-reversal":
        output = train_reversal(args.data, args.out,
                                Settings(epochs=args.epochs, batch_size=args.batch_size,
                                         lr=args.lr, patience=args.patience,
                                         seeds=tuple(args.seeds)), hidden=tuple(args.hidden))
    elif args.action == "detect-reversal":
        output = detect_reversal(args.data, args.model)
    elif args.action == "plot-reversals":
        output = plot_validation_reversals(args.data, args.model, args.out, args.candles)
    else:
        config = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
        if config.get("model_version") in {"0.5.0", "0.6.0"}:
            print(json.dumps(predict_atr(args.data, args.model), ensure_ascii=False, indent=2))
            return
        probability = predict(args.data, args.model)
        output = {"up_probability_uncalibrated": probability, "predicted_direction": "up" if probability >= .5 else "down"}
        config = json.loads((Path(args.model) / "config.json").read_text(encoding="utf-8"))
        output["horizon_candles"] = int(config.get("horizon_candles", 1))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
