# AI Finance

**Current development version: v0.7.0 — detection of the current XAUUSD M15 trend.**
It labels the present state as **uptrend / downtrend**, using only closed historical candles.
There is **no future forecast or reversal/entry signal** in v0.7; older v0.1–v0.6 experiments remain available.
The v0.6.0 release remains documented below.

A small **CPU-only financial neural network built from scratch in NumPy** using MetaTrader OHLCV candles. Earlier versions experimented with next-candle forecasts; v0.7 focuses only on recognizing the **current** market trend. This is a research/learning project—not a trading system or financial advice.

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

## v0.4: compare network sizes (fixed 15-minute target)

Version 0.4 keeps the **same 485 inputs**, derived from the same 20-candle
windows and indicators, and holds the target constant at the direction of
the *next completed M15 candle*. Four fresh networks are trained separately:

| Name | Architecture |
| --- | --- |
| large (v0.2/v0.3 control) | `485 → 128 → 64 → 32 → 1` |
| medium | `485 → 64 → 32 → 1` |
| small | `485 → 32 → 16 → 1` |
| linear baseline | `485 → 1` |

These use the **exact same chronological training/validation windows as the
v0.3 h1 run**, including the 8-candle boundary purge. The scaler is fitted
exclusively on training windows, and the chosen seed/epoch and architecture
minimize **validation loss**. Neither the internal holdout nor
`data/final_test.csv` is evaluated by this command.

```bash
git pull origin main
python -m finance_ai compare-architectures \
  --csv data/development.csv \
  --out models/finance-v0.4 \
  --epochs 100 --seeds 13 42 97
```

The summary is `models/finance-v0.4/comparison.json`. Individual checkpoints
are in `large/`, `medium/`, `small/` and `linear/`, each with its own
`model.npz`, `config.json`, `preprocessing.json` and `metrics.json`.
Existing v0.1/v0.2/v0.3 checkpoints are still readable for inference.

For example, to predict with the selected checkpoint (`selected_model` in
the report):

```bash
python -m finance_ai predict --csv data/recent_candles.csv \
  --model models/finance-v0.4/small
```

The names are examples: use the **actual selected checkpoint**, not necessarily
`small`. All models are research experiments; accuracy does not imply profit.

## v0.5: wider network, three-way ATR classification

The existing v0.1-v0.4 commands and model directories are unchanged. The new
experiment predicts whether the **next closed M15 candle** rises materially,
falls materially, or remains within an ATR-scaled neutral band.

- Features: same 20 × 24 + 5 = **485** inputs as v0.2-v0.4.
- Default MLP: **485 → 512 → 256 → 128 → 64 → 3** (**421,507** learnable
  parameters), versus 72,577 in the v0.4 large binary network. Extra neurons
  are **not evidence of better out-of-sample performance**; this is an
  intentionally larger experiment. Hidden widths are configurable to fit
  CPU/RAM limits; larger models train more slowly.
- Classes: `down`, `neutral`, `up`. Let `d = close[t+1] - close[t]`, and
  `threshold = 0.25 * ATR14[t]` (from **the most recent completed candle**).
  `d < -threshold` → down; `d > threshold` → up; otherwise neutral.
  The default threshold can be changed with `--atr-multiplier`.
- Training: 3-class **softmax** and **categorical cross-entropy**, trained with
  custom NumPy backprop + AdamW and early stopping.
- Measures not only accuracy and validation cross-entropy but also 3-class
  confusion matrix (rows=true, columns=predicted), per-class recall/precision,
  macro F1, balanced accuracy, majority accuracy, and validation cross-entropy
  of the training-class-prior baseline. High neutral-class accuracy alone is
  not sufficient evidence of predictive skill.
- Uses the **same historical window endpoints and eight-candle purge** as
  v0.3/v0.4, training-only normalization, and **validation-loss-only** seed
  selection. The internal 15% holdout and external `data/final_test.csv` are
  **not evaluated** by this command. Old binary and new ternary losses are
  not directly comparable; the label problem is different.

```bash
git pull origin main
python -m finance_ai train-atr --csv data/development.csv \
  --out models/finance-v0.5 --epochs 100 --seeds 13 42 97
```

For an optional smaller CPU trial, add `--hidden 256 128 64` to the above
command and choose another output directory so checkpoints are not replaced.
For an optional larger trial, use `--hidden 1024 512 256 128 64`, **only if your
hardware can handle the increased compute** and the validation result justifies
it. These are not automatically searched.

Inference returns all three **uncalibrated** probabilities and the top class:

```bash
python -m finance_ai predict --csv data/recent_candles.csv \
  --model models/finance-v0.5
```

Model files: `model.npz`, `config.json`, `preprocessing.json`, `metrics.json`.
These are experimental signals, not trading recommendations or validated
probabilities of price movement.

## v0.6.0 — MACD-free ATR three-class classifier

`train-atr` now trains a fresh **MACD-free** 3-class classifier. No MACD
indicators are calculated or provided to this model. RSI14, SMA20/SMA50,
ATR14, time/session features, 20-candle history and five trend summaries
are unchanged. Features drop from **24 to 21 per candle**; network input
changes from **485 to 425**, giving architecture
**425 → 512 → 256 → 128 → 64 → 3** (**390,787 parameters**).
Softmax outputs remain `down`, `neutral`, and `up`. The train/validation
splits, 25% ATR label threshold, and validation-only selection are unchanged.

```bash
git pull origin main
python -m finance_ai train-atr --csv data/development.csv \
  --out models/finance-v0.6 --epochs 100 --seeds 13 42 97
```

The `config.json` and `preprocessing.json` record the 21-feature schema and
version `0.6.0`. Existing v0.1–v0.5 checkpoints remain readable for inference
with their exact old schemas; do **not** overwrite a previous model directory.
The final test set is still reserved and is never used for tuning.

### v0.6.0 validation snapshot (M15 XAUUSD development data)

For the same chronological validation window (**4,171 samples**) with
ATR multiplier **0.25**, the user-run experiment returned:

| Validation metric | v0.5 (MACD) | v0.6.0 (no MACD) |
| --- | ---: | ---: |
| Accuracy | 38.50% | 37.54% |
| Balanced accuracy | 36.47% | 36.39% |
| Macro F1 | 30.11% | 34.86% |
| Cross-entropy loss | 1.08892 | 1.09323 |
| Down precision | 27.78% | 31.07% |
| Down recall | 0.41% | 15.16% |

The **majority-class accuracy baseline is 37.09%**. Removing MACD makes
DOWN predictions less rare and improves macro F1, but **does not establish
better overall predictive performance**: validation loss and accuracy are
slightly worse. The selected v0.6.0 run used seed **13**, epoch **3**.
Neither the internal reserved holdout nor `data/final_test.csv` was used
for selection/evaluation. These figures are **not** live trading results.

## v0.7.0 — Current-trend recognition (no prediction)

**Scope:** classify the current XAUUSD M15 regime as exactly **uptrend** or
**downtrend** using only the candles that have already closed. No reversal
detection, future-price targets, entry signals, or trading decisions.
This is a **separate model and command**: v0.1–v0.6 checkpoints are untouched.

### Indicators and input features

| Indicator | Configuration | Purpose |
| --- | --- | --- |
| Bollinger Bands | **20 periods, 2 standard deviations** | %B, relative width, SMA20 middle-line slope |
| Single moving average | **EMA10** | EMA slope, its distance from the Bollinger SMA20 middle line; report bullish/bearish line crossings |
| Aroon Up / Down | **25 periods** (26 observed bars in the `ta` convention) | Recency of highs and lows; up/down spread |
| ATR | **14 periods** (Wilder smoothing) | Volatility-adjusted strength and relative ATR input |

The model also receives normalized OHLC returns, log tick volume, and real
candle gaps. **15 features × 40 completed M15 candles = 600 inputs**.
This new pipeline does **not** use MACD, RSI, or SMA50. Indicator calculations
use pandas/NumPy, not an additional dependency; they are historical-only.
The MLP defaults to **600 → 512 → 256 → 128 → 64 → 1** with a binary sigmoid
output (custom NumPy backpropagation and AdamW). Hidden sizes can be adjusted.

### Training-reference definition and important limitations

There is no unique objective ground-truth label for 'current trend'.
For this first experiment we assign an **explicit causal reference label**:
run ordinary least squares on **log(close) for the trailing 32 completed bars**,
ending at time `t`. Positive or zero slope → `uptrend`; negative slope →
`downtrend`. No `t+1` candle, future extrema, or future ATR is used.

**Crucial:** This tests whether the neural network can **reproduce a
historical price-slope definition**, not whether its market interpretation
is correct or whether future prices will rise/fall. The label is not derived
from Aroon or EMA crossings, so the model is not merely copying those specific
indicators; it is still learning a **constructed label from the same past data**.
When the slope's net 32-bar move is smaller than **0.5 ATR14**, the report flags
`weak_price_trend`, but outputs remain binary per specification. The sigmoid
score is **uncalibrated**, not a reliable 80%-confidence promise.

Use chronological **70% train / 15% validation / 15% reserved internal
holdout** within `data/development.csv`. Scaling is fitted on training only;
seed/epoch selection uses validation loss only. Neither the internal reserved
holdout nor external `data/final_test.csv` is evaluated in training.
Validation includes majority-class, previous-trend **persistence**, and EMA10
vs Bollinger-middle baselines, so a high accuracy cannot be mistaken for an
independent forecast result.

### Commands

```bash
git pull origin main
python -m finance_ai train-trend --csv data/development.csv \
  --out models/finance-v0.7 --epochs 100 --seeds 13 42 97
```

To classify the **current last completed candle** after training (supply enough
recent history to warm up Aroon25/BB20 and feed all 40 feature candles):

```bash
python -m finance_ai detect-trend --csv data/development.csv \
  --model models/finance-v0.7
```

For a smaller CPU smoke test use `--epochs 2 --seeds 42 --hidden 64 32`
with **a different output directory**; that checkpoint has a different
architecture. `train-trend` writes `model.npz`, `config.json`,
`preprocessing.json`, and `metrics.json`. The `detect-trend` JSON provides
the trend, both uncalibrated scores, weak-trend flag, timestamp, indicator
snapshot, and EMA10/SMA20 crossing direction. It assumes the supplied export
contains **only closed candles**.

**v0.8 reversal-point detection is intentionally out of scope.**

### v0.7 visual review — 500 validation candles

Render the trained v0.7 checkpoint **without retraining**. The chart shows
XAUUSD M15 candlesticks, light green/red shading for the model's *current*
uptrend/downtrend classification, EMA10, Bollinger Bands (20,2), and two
colored ribbons: **AI v0.7** versus the **32-bar historical OLS-slope proxy**.
The two ribbons may disagree; this is useful for inspecting transition periods.

Matplotlib is an **optional plotting-only dependency**:

```bash
python -m pip install -e '.[plot]'
python -m finance_ai plot-trend \
  --csv data/development.csv \
  --model models/finance-v0.7 \
  --candles 500 \
  --out charts/finance-v0.7-validation-500.png
```

Open the generated PNG at `charts/finance-v0.7-validation-500.png`.
`--candles` controls how many consecutive validation candles to display.
The command intentionally selects the **last 500 completed candles from the
chronological validation segment of `development.csv`**, **not** the latest 500
candles in the file, which would expose the reserved internal holdout. The
separately reserved `final_test.csv` is never inspected. No forecasts,
reversal/entry signals, training, checkpoint modifications, or automatic
parameter selection are performed. The printed agreement rate measures
matching a constructed historical reference; **not** real-market accuracy.
