# AI Finance

A small **CPU-only financial neural network built from scratch in NumPy**. It learns to predict the next available 15-minute candle's direction from real MetaTrader OHLCV exports. This is a research/learning project—not a trading system or financial advice.

## v0.2: trend-aware experiment

- Imports comma-, tab-, or semicolon-separated CSV, or Excel (`.xlsx`/`.xls`). Required columns: `time,open,high,low,close,volume,timeframe`; timestamps are Unix UTC seconds; timeframe must be `15`.
- Calculates MACD (12/26/9), RSI14, SMA20, SMA50 and **ATR14** with [`ta`](https://technical-analysis-library-in-python.readthedocs.io/en/latest/).
- Derives the existing **24 features per candle** (OHLCV relative values, indicators, weekday, time of day, early/middle/late month, approximate Asian/European/US market sessions, and trading gaps).
- Adds **five trend summaries once per 20-candle sample**: returns over 5, 10, and 20 observed candles, trend strength (unsigned efficiency ratio), and relative ATR14. **Total inputs: 20 × 24 + 5 = 485.**
- Trains a custom NumPy MLP **485 → 128 → 64 → 32 → 1**, using backpropagation, AdamW and BCE loss; random initialization, no pretrained weights or deep-learning framework.
- Splits data **chronologically (70% training / 15% validation / 15% internal test)**; fits normalization **only on training data**. Selects the best seed and epoch by validation loss and saves the selected checkpoint.
- Keeps existing **v0.1 (`480 → 128 → 64 → 32 → 1`) model checkpoints loadable for inference**; v0.2 is retrained from scratch and written to a different folder.

### What the five features mean

The sample ends at a **closed** candle `t`; no future values are used:

| Feature | Definition |
| --- | --- |
| `trend_5` | `close[t] / close[t-5] - 1` |
| `trend_10` | `close[t] / close[t-10] - 1` |
| `trend_20` | `close[t] / close[t-19] - 1` (first-to-last close within the 20-candle window) |
| `trend_strength` | `abs(close[t] - close[t-19]) / sum(abs(close changes over the window))`; zero when flat |
| `atr14_ratio` | `ta`'s ATR14 (Wilder smoothing), divided by `close[t]` |

Trend strength ranges from 0 to 1: 1 indicates a perfectly directional sequence; values near 0 indicate choppiness. It does not itself imply an upward or downward trend.

## Quick start (Ubuntu / Linux)

Python 3.10+ (3.12 recommended), CPU, and a MetaTrader M15 CSV are sufficient.

```bash
# If you already cloned the project, use: git pull origin main
git clone https://github.com/omidnami/Ai-finance.git
cd Ai-finance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Copy your **15-minute CSV** into `data/development.csv`. Keep the separately reserved `data/final_test.csv` **untouched** while developing and tuning this model.

```bash
python -m finance_ai train --csv data/development.csv --out models/finance-v0.2 \
  --epochs 100 --seeds 13 42 97
```

For a quick smoke test (still trains on the development data):

```bash
python -m finance_ai train --csv data/development.csv --out models/finance-v0.2 \
  --epochs 5 --seeds 42
```

Saved artifacts:

```text
models/finance-v0.2/
├── model.npz           # selected best weights and biases
├── config.json         # architecture and feature names
├── preprocessing.json  # train-only normalization, ATR/indicator definitions
└── metrics.json        # selection history and split results
```

Inference uses the **same feature and normalization pipeline**:

```bash
python -m finance_ai predict --csv data/recent_candles.csv --model models/finance-v0.2
```

For an existing v0.1 checkpoint, explicitly pass `--model models/finance-v0.1`. Do not overwrite v0.1 when experimenting with v0.2.

Run tests:

```bash
python -m pytest -q
```

## Data and evaluation notes

- Files must contain strictly increasing unique timestamps with valid OHLC prices and at least 100 closed M15 candles; **a few hundred or more** are needed for the current train/validation/test split, and thousands are preferable.
- Trading hours and holiday/weekend **gaps are genuine**; no synthetic candles are created. Session windows are approximate local Tokyo/London/New York hours using DST-aware IANA zones, not official exchange calendars.
- Technical indicators warm up from previous candles; incomplete initial rows are excluded. All 5 trend features use only historical closes and the ATR value available at the end of the window.
- Training labels represent `close[t+1] > close[t]` and ties count as down. Output probabilities are **uncalibrated**; classification accuracy is not a profitability measure.
- Development comparisons should be selected **only by validation**, not by repeated peeking at the internal test or the reserved `final_test.csv`. The final reserved holdout is for one evaluation after selecting the approach. The CLI still reports the internal test diagnostically after training.

## Support websites

- [Omid Nami](https://omidnami.com)
- [TurnX](https://turnx.ir)

## License

Apache License 2.0 (see `LICENSE`).

## v0.3: compare forecast horizons (1, 4, 8 closed M15 candles)

Version 0.3 keeps the exact **485 → 128 → 64 → 32 → 1** architecture,
24 per-candle features, and five trend/ATR features from v0.2. It changes
**only the training target**, not the model input. For the last observed
closed candle `t`, each horizon `h` uses the label:

`close[t+h] > close[t]` (tie = down). Horizon 4 is four **traded** candles
later, nominally one hour; market closures can make elapsed wall time longer.

To perform a fair comparison all three models use the **same input windows**,
with identical train/validation boundaries and training-only normalization.
An eight-candle gap is purged before each split boundary so future target
prices cannot cross into validation from training or into the withheld portion
from validation. Each horizon has its own trained weights and validation loss.
The best is selected **solely by validation loss**, not by the internal test.

```bash
git pull origin main
python -m pip install -e '.[dev]'
python -m finance_ai compare-horizons --csv data/development.csv \
  --out models/finance-v0.3 --epochs 100 --seeds 13 42 97
```

Results are saved to `models/finance-v0.3/comparison.json` and to separate
`h1/`, `h4/`, `h8/` folders containing `model.npz`, `config.json`,
`preprocessing.json`, and `metrics.json`. The JSON identifies the selected
model path. To predict with the winner, use for example:

```bash
python -m finance_ai predict --csv data/recent_candles.csv \
  --model models/finance-v0.3/h4
```

The prediction JSON includes `horizon_candles` so a four-candle forecast is
not mistaken for a next-candle forecast. Existing v0.1/v0.2 inference remains
supported. **Do not use `data/final_test.csv` for comparison or tuning.**
The v0.3 comparison command does not evaluate even the internal reserved
15% development holdout. A final unbiased test requires data not used during
tuning. This is a research experiment; outputs are uncalibrated and are not
investment recommendations.
