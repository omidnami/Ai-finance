# AI Finance

A lightweight **financial neural network built from scratch with NumPy**, trained on CPU using real 15-minute MetaTrader CSV/Excel candles. The first version is an educational experiment in learning and evaluating next-candle direction; **it does not place trades**.

## v0.1: what works

- Reads `.csv`/`.xlsx`/`.xls` with `time, open, high, low, close, volume, timeframe` (Unix seconds, timeframe **15**).
- Computes MACD (12/26/9), RSI (14), SMA20, SMA50 using [`ta`](https://technical-analysis-library-in-python.readthedocs.io/en/latest/).
- Adds UTC time, weekday, early/middle/late month, approximate Tokyo/London/New York session indicators (DST-aware), and actual candle gaps: **24 features/candle**.
- Builds 20-candle rolling windows, predicts whether the **next available completed candle** closes above the last observed close.
- Implements MLP **480 → 128 → 64 → 32 → 1**, ReLU, sigmoid, binary cross-entropy, backpropagation, and AdamW in NumPy. No pretrained weights, PyTorch, or TensorFlow.
- Chronological **70% training / 15% validation / 15% holdout test**. Normalization is fitted only on the training portion. Multiple random seeds and early stopping are selected by validation loss; the test set is evaluated **only on the selected checkpoint**.
- Saves the best weights and run metadata for CPU inference; reports accuracy, error rate, loss, and class-baseline accuracy.

## Quick start (Ubuntu / Linux)

Requires **Python 3.10+** (3.12 recommended), a CPU, and a CSV or Excel file exported from MetaTrader.

```bash
git clone https://github.com/omidnami/Ai-finance.git
cd Ai-finance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Copy your **15-minute** candle file somewhere on your computer (e.g. `data/history.csv`); raw Excel and trained model outputs are gitignored.

```bash
python -m finance_ai train --csv data/history.csv --out models/finance-v0.1
```

To run a quick smoke test on fewer epochs:

```bash
python -m finance_ai train --csv data/history.csv --out models/finance-v0.1 --epochs 5 --seeds 42
```

Model artifacts:

```text
models/finance-v0.1/
├── model.npz           # Best validation checkpoint (weights and biases)
├── config.json         # Model architecture and features
├── preprocessing.json  # Train-only normalization and indicator settings
└── metrics.json        # Chronological split, runs, train/val/test statistics
```

Inference uses **the same indicator and time-feature service**:

```bash
python -m finance_ai predict --csv data/recent_candles.csv --model models/finance-v0.1
```

Use enough historical **closed** candles in the prediction CSV file for MA50/MACD warmup + 20-candle lookback (100+ recommended). Output is an **uncalibrated** upward probability; it is not a guaranteed real-world probability.

To run tests:

```bash
python -m pytest -q
```

**Note:** Version 0.1 requires timeframe `15` in every row. A CSV exported as M5 (`timeframe=5`) will be rejected. A 100-candle file can test the importer and indicators, but is too short for the current train/validation/test split; use at least 200 closed M15 candles for the minimal training smoke test, and preferably much more.\n\n## Input definition

| Field | Definition |
| --- | --- |
| `time` | Unix timestamp in seconds, in UTC, aligned to 15-minute bars |
| `open`, `high`, `low`, `close` | Valid positive OHLC prices |
| `volume` | Nonnegative reported volume (could be tick volume depending on source) |
| `timeframe` | `15` for every row |

Data must be in ascending time order, without duplicated timestamps. Valid market closures, weekends, and holidays can cause **gaps**; no synthetic candles are inserted. Session flags are simple time-window approximations (Tokyo 09–18, London 08–17, New York 08–17 in local time, weekdays), **not official exchange opening calendars**. Verify MetaTrader timestamps and broker time-zone conventions before training; if the input `time` values are broker-local rather than UTC epoch seconds, convert them first.

The 24-feature specification is in `finance_ai/features.py` and can be versioned for future models. If features or network dimensions change, old checkpoints are **not** automatically compatible.

## How evaluation works

Each sample contains 20 completed candles and a binary label defined by `close(t+1) > close(t)`. Ties count as down (label 0). The model never receives the next close as an input. Splits are **time-ordered**, and only training samples determine normalization statistics. The validation portion chooses the best seed/epoch; the holdout test is reserved for a single final assessment. Overlapping past context at a split boundary is permitted because it is historically available; labels remain strictly inside their own split.

The metrics are **classification metrics**, not a backtest or expected trading profit. Market predictions are uncertain and an accuracy above 50% is not proof of a profitable strategy. Future versions can add walk-forward evaluation and verified exchange-session calendars.

## Support websites

- [Omid Nami](https://omidnami.com)
- [TurnX](https://turnx.ir)

## License

Apache License 2.0 (see `LICENSE`).
