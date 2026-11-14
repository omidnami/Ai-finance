# AI Finance

**Current development version: v0.9.0 — market context, setup quality and stop-entry paper simulation.**
This version **combines the unchanged v0.7 current-trend and v0.8 confirmed-reversal checkpoints** with a new, transparent, Brooks-inspired decision/replay engine. No new neural weights are trained, and no live orders are sent. All earlier commands remain available.

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

## v0.8.0 — confirmed price-reversal candidates (no future forecast)

**Goal:** identify **bullish reversal / bearish reversal / none** on XAUUSD M15.
This model is **independent of the v0.7 trend checkpoint**. The earlier
trend-recognition network and all v0.1–v0.7 commands/checkpoints remain
unchanged. The third category (`none`) is essential because genuine
turning-point candidates are sparse.

### Features (not fixed indicator triggers)

The new model sees **40 completed candles × 20 channels = 800 inputs**:

- **RSI14**: continuous value, including the user's areas of interest below
  **20** (potential oversold reversal), above **70** (overbought), and the
  **60–70** zone in possible pullbacks. These are **observations**, not
  automatic buy/sell signals.
- **Bollinger Bands (20, 2)**: position (%B), bandwidth, current upper/lower
  touches and distance from the SMA20 middle.
- **MACD (12, 26, 9)**: MACD, signal, histogram and histogram change.
  MACD provides momentum context; it is **not** required to cross to classify.
- **Candlestick/volatility**: OHLC relative returns, body, upper/lower wick
  normalized by **ATR14**, volume and observed candle gaps. No artificial
  candles are added across market closures.

The custom NumPy CPU MLP is
**800 → 512 → 256 → 128 → 64 → 3** by default, trained with class-weighted
softmax cross entropy; the indicator pipeline uses only available past bars.
You can supply smaller hidden widths for a quick smoke test.

### What counts as a training reversal?

There is **no unique objective ground truth for turning points**. v0.8 uses a
precisely documented **price-only, past-only confirmation proxy** so RSI,
Bollinger and MACD are not themselves the correct answers.

For each latest **completed confirmation candle `t`**, a *candidate pivot*
may have formed at **`t−1`**:

1. A **bullish** reference requires a descending price context from the
   20-bar historical OLS log-close slope **ending at `t−2`**, a 9-bar
   local low at `t−1`, and a higher close at `t` by at least
   **0.20 × ATR14[t−1]**, without a new lower low on `t`.
2. A **bearish** reference is symmetric: ascending prior context, 9-bar
   local high at `t−1`, a lower close by at least 0.20 ATR, and no new
   higher high. The pre-move direction must exceed **0.35 ATR** in the
   corresponding direction. Otherwise the reference is `none`.

Every value in this rule is known **at or before the end of `t`**. The
**earliest possible detection time is the close of `t`**; this program
does **not** claim the turn was detected at `t−1`. No future close or
future pivot is used in features, training labels, inference or evaluation.
These constructed historical patterns are **not validated price turns**,
reversal forecasts or guaranteed trade signals.

### Training and evaluation

Chronological **70/15/15 train/validation/reserved holdout** within
`data/development.csv`, with a **40-candle purge** at each boundary to
prevent overlapping training and validation input windows. Normalization and
class weights are fitted on training only; the best seed/epoch is selected
by **weighted validation loss**. No internal holdout or separately reserved
`data/final_test.csv` is evaluated or used to select the model.

Because `none` is likely the dominant label, inspect the **per-class
precision/recall, confusion matrix and macro F1**, not just accuracy.
Scores are **uncalibrated**; even high accuracy against a constructed
historical label is **not evidence of real reversal forecasting skill**.

### Commands

```bash
git pull origin main
python -m finance_ai train-reversal --csv data/development.csv \
  --out models/finance-v0.8 --epochs 100 --seeds 13 42 97
```

Progress (seed, epoch, weighted validation loss) is printed as training
runs. For a quick CPU smoke test, use
`--epochs 2 --seeds 42 --hidden 64 32 --out models/finance-v0.8-smoke`;
keep that separate from the full checkpoint.

To classify **the last completed candle**:

```bash
python -m finance_ai detect-reversal --csv data/development.csv \
  --model models/finance-v0.8
```

The JSON gives one of `none`, `bullish_reversal`, `bearish_reversal`,
uncalibrated class scores, an independent price-only reference,
pre-move context, RSI thresholds (20/60/70), Bollinger touch/re-entry
flags, and MACD histogram/cross. The pivot time shown is **historical**;
the detection timestamp is **one closed candle later**.

To visualize **the last 500 validation candles** with markers at the
**confirmation candle** (not an earlier hindsight pivot), plus Bollinger
bands and RSI thresholds:

```bash
python -m pip install -e '.[plot]'
python -m finance_ai plot-reversals --csv data/development.csv \
  --model models/finance-v0.8 --candles 500 \
  --out charts/finance-v0.8-validation-500.png
```

Filled green/up and red/down triangles are **network outputs**; empty
circles are price-only reference confirmations. Only the validation
range is displayed; the reserved holdout and final test remain untouched.

**Scope boundary:** this is *price-reversal candidate recognition*, not a
change to v0.7 current-trend classification. Future turning-point
prediction, trend/reversal combination and actual trade-entry logic are
not included.

### v0.8 sliding-window walk-forward (40 history → 1-step → 20 future)

This command uses the **existing trained v0.8 checkpoint**; weights are
unchanged. It is a *causal, fixed-horizon directional experiment*, not a
live trading system or verified profitability backtest.

**There are two different numbers:**
- The **neural network requires 40 past closed M15 candles** per decision.
  After the initial indicator warmup (at least 110 historical raw candles),
  the 40-candle **input window slides forward by ONE candle each step**:
  `[0..39] → [1..40] → [2..41] → …` (illustrative window offsets,
  after warmup). The last completed candle of **every** window gets a
  fresh reversal classification; intermediate bars are no longer skipped.
- **20 future *observed* candles** are used only to **evaluate each previously
  issued paper signal**. They are **never included in that signal's input**.

A BUY/SELL candidate is logged at candle `t` only if v0.8's class
(triangle) and the price-only training reference (circle) **match in the
same non-none direction**, as in the chart. The price-only reference is a
*training proxy*, **not independent confirmation**.

The hypothetical entry is **OPEN[t+1]**; the fixed exit is **CLOSE[t+20]**.
Only once the 20th subsequent candle closes is that signal labeled `WIN`
/ `LOSS` / `FLAT` from its directional price move (after an optional flat
quote-price-unit cost). **Each signal is tracked separately**: if two
candidates occur a candle apart, they can be *pending simultaneously*
and resolve 20 candles after their respective confirmation times.
Unresolved signals at the end stay `pending`; they are never assigned
outcomes using unavailable future bars.

```bash
git pull origin main
python -m finance_ai walkforward-reversals \
  --csv data/final_test.csv \
  --model models/finance-v0.8 \
  --horizon-candles 20 \
  --out reports/v0.8-test-20.csv
```

For a quick sample, add `--max-steps 50` to review the first 50
**consecutive one-candle steps** (10 steps are not enough for a 20-candle
outcome). `--batch-candles` and `--max-batches` are still accepted as
**backward-compatible aliases**, but they now mean the evaluation horizon
and maximum number of **steps**, respectively—not the scanning stride.

The command prints each step and creates:

- `reports/v0.8-test-20.csv`: one row per **observed candle**, 40-bar window
  endpoints, fresh AI/reference labels and signal, any newly resolved
  *previous* signal, entry and exit prices, directional move and excursion.
- `reports/v0.8-test-20.summary.json`: total signals, resolved WIN/LOSS/FLAT,
  unresolved pending signals, separate BUY/SELL statistics and assumptions.

Optional `--cost-per-unit 0.5` subtracts 0.5 quote-price units per ounce
for each completed paper signal. Otherwise **cost is zero**.
No stop loss, take profit, variable spread, slippage, position sizing,
leverage or actual executed P&L is simulated. Even a WIN on the 20-candle
directional test need not be a profitable or executable trade.
**Repeated strategy tuning against `final_test.csv` uses up its status
as an untouched final test**; reserve a newer, unseen period for a final
unbiased evaluation.

## v0.9.0 — Market Context → Setup Quality → Entry & Risk

**The new v0.9 is a separate, deterministic and auditable research layer**,
not a new neural network or claim of an Al Brooks proprietary strategy.
It loads the existing v0.7 trend checkpoint and v0.8 confirmed-reversal
checkpoint. Their architecture, preprocessing and trained weights are
unchanged, and **no retraining is required**.

### Research basis: price action concepts vs our implementation

| Concept | Brooks' published approach | v0.9 experiment |
| --- | --- | --- |
| **Market context** | Distinguish breakout/spike, trend channel, and trading range; entries need to suit the market cycle, and the middle of a range is typically a poor location. | Inspect trailing **32 completed candles**, regression slope (in ATR units), directional efficiency, relative position within the 32-bar range, plus the **v0.7 direction**. Tag `breakout`, `trend_channel` or `trading_range`. |
| **Setup and signal bar** | A multi-bar *setup* differs from the final *signal bar*. Strong closing location, surrounding bars and context matter more than an isolated candlestick. | Require v0.8 **AI triangle and constructed price-reference circle** to agree (same bullish/bearish direction), then grade the signal candle's body/close position/wick, range edge, prior slope and v0.8 score. A **0–100 heuristic** controls whether an order can be queued. |
| **Entry and risk** | A stop entry can trigger beyond the signal-bar high/low on a *subsequent* candle, with an appropriately placed protective stop. Risk, reward and probability interact; reward/risk alone is not a profitability guarantee. | Plan a **BUY STOP** above signal high or **SELL STOP** below signal low by a configurable tick; protective stop beyond the signal/pivot extreme + ATR buffer, a configurable target (default **2R**). Paper orders expire after 3 future bars; after filling, process stop/target or close after 20 observed bars. |

The three-source research references are [Market Cycle and trend channels](https://www.brookstradingcourse.com/futures-market/trade-trend-channels/),
[Setup vs Signal Bar](https://www.brookstradingcourse.com/price-action/professional-price-action-trader/),
[Trading ranges](https://www.brookstradingcourse.com/how-to-trade-manual/trading-ranges/),
[Stop-entry signal/entry bars](https://www.brookstradingcourse.com/analysis/emini-high-1-buy-signal-context-follow-through/),
and [Risk/reward tradeoffs](https://www.brookstradingcourse.com/how-to-trade-price-action-manual/).
**The numeric thresholds, 0–100 score and simulator are our engineering
hypotheses, NOT rules or statistical probabilities supplied by Brooks.**

### Important causal / execution rules

- Read **only completed M15 candles**; each classification is at candle
  close `t`. Model features and context use `<= t`. The earliest
  stop-entry fill is **candle `t+1`**, only if its high/low hits the trigger.
  No automatic fill at the next open unless the price actually gaps through
  the stop-entry trigger.
- A filled paper position has a **fixed stop** and **fixed target**. If
  stop and target are both touched inside the same OHLC candle, assume
  the **stop was first** (conservative without tick-level data).
  Adverse open gaps through the stop exit at the worse opening price.
  Favorable target gaps are capped at the target level.
- Keep at most **one pending order or open position** at a time; log
  overlapping opportunities as skipped. Expired stop-entry orders were never
  positions. Closed trades have a fixed maximum holding period.
- An optional fixed **round-trip cost in XAUUSD quote-price units** is
  deducted from outcomes (default **zero**). Hypothetical position size uses
  a constant user-supplied initial equity with default **0.5% risk budget**
  per paper trade; **not** a real broker position sizing convention,
  compounding engine or evidence of executable returns.
- Context and quality are **filters**. Model softmax and sigmoid outputs
  are uncalibrated. A favorable quality score is **not** a win probability.
  The v0.8 price reference **is its training target**, not an independent
  confirmation source. The simulator does not forecast reversals in advance.

### CLI commands

**Inspect one current setup on the latest completed candle:**

```bash
git pull origin main
python -m finance_ai analyze-entry \
  --csv data/recent_candles.csv \
  --trend-model models/finance-v0.7 \
  --reversal-model models/finance-v0.8
```

Returns: current cycle and v0.7 trend, AI reversal, matching historical
reference, candidate side, reason if rejected, quality score and planned
**entry trigger, protective stop and 2R target**.

**Replay every candle chronologically with paper entry execution:**

```bash
python -m finance_ai backtest-entries \
  --csv data/new_unseen_m15.csv \
  --trend-model models/finance-v0.7 \
  --reversal-model models/finance-v0.8 \
  --out reports/v0.9-entries.csv
```

Test the first 200 sequential steps with `--max-steps 200`.
Parameters (research settings, **not optimized defaults**):
`--min-quality 65`, `--order-valid-bars 3`,
`--max-holding-bars 20`, `--target-rr 2.0`,
`--tick-size 0.01`, `--roundtrip-cost-units 0.0`,
`--risk-budget-fraction 0.005`, `--initial-equity 10000`.

Files written:
- `reports/v0.9-entries.csv`: **one row per closed candle** with regime,
  both models' labels, setup score/filters, order status and trade event.
- `reports/v0.9-entries.trades.csv`: **one row per completed paper trade**
  including entry/stop/target/exit, reason, return in `R` and hypothetical
  P&L with the configured simplified fixed cost.
- `reports/v0.9-entries.summary.json`: event counts, accepted/rejected
  setups, filled/expired orders and BUY/SELL/overall completed trade stats.

**Data discipline:** `data/final_test.csv` was already inspected and
used for v0.8 development discussion. It must **not** be represented as
an untouched out-of-sample evaluation of v0.9. Use a new subsequent
time period for an unbiased evaluation after freezing these rules.
This project provides research tools, **not investment advice or real orders**.

### Full offline interactive history chart (all ~7,000 test candles)

The separate **visual-only** command `plot-history` combines the **existing
reports**, not a new backtest. It displays *all original OHLC candles* and
overlays v0.8 AI predictions (triangles), the v0.8 price-only reference
(open circles), agreement between the two (gold stars), and the v0.9
**queued/proposed setups** (diamonds). **No TP, SL, position execution,
win/loss, R-multiples or hypothetical P&L are plotted or recalculated.**
The v0.9 queued setup is **not necessarily a filled trade**.

```bash
git pull origin main
python -m pip install -e '.[interactive]'

python -m finance_ai plot-history \
  --csv data/final_test.csv \
  --reversals reports/v0.8-test-20.csv \
  --setups reports/v0.9-entries.csv \
  --out charts/finance-full-history.html

xdg-open charts/finance-full-history.html
```

The `--setups` overlay is optional if you only want the v0.8 markers.
The browser HTML is **fully offline** (Plotly bundled locally, **no CDN**);
all candles are retained, but the first view zooms to the last **280**
candles for readability. Use **mouse wheel zoom, drag-pan, legend toggles,
date buttons and the bottom range slider** to inspect the entire history.
Choose a different initial view with `--initial-candles 600`.
The output size is several MB because Plotly is embedded for offline use.

The command cross-checks timestamps from each report against the raw candle
file, refusing to show mismatched datasets. Use the reports generated from
**that exact same OHLC CSV**. The first indicator-warmup candles naturally
have no markers; this does **not** mean they were tested.
The figure is purely for human inspection and **does not set trading exits
or test profitability**. Existing v0.9 paper risk settings were provisional
research defaults, **not user-approved TP/SL rules**.

### Fixed $2.50 TP / $2.00 SL: signal-close entry and wick-based first-hit review

This is a **separate v0.9 research command**. It **does not use** the older
`backtest-entries` stop-order triggers, ATR structural stops, 2R targets, or
portfolio sizing. The existing neural checkpoints and prior commands remain
unchanged.

**Exact rules agreed for this experiment** (XAUUSD quoted in USD per troy ounce):

- At the close of a confirmed signal candle `t`, record an **illustrative
  paper entry at `CLOSE[t]`**. This is *not* a guaranteed real-world closing
  price fill after the signal becomes known.
- BUY: `TP = CLOSE[t] + 2.50`, `SL = CLOSE[t] - 2.00`.
  SELL: `TP = CLOSE[t] - 2.50`, `SL = CLOSE[t] + 2.00`.
- Search **only subsequent candles** `t+1, t+2, …`, and detect the **first
  touch of the level with High/Low (shadows)**. Never use Close/Open to
  decide whether the target or stop was touched; the signal candle's own
  high/low does **not** count.
- If the next observed candle hits **both TP and SL**, the actual sequence
  is unknowable from OHLC. Record `AMBIGUOUS_SL_FIRST`, separately counted
  as a loss in the **conservative** win rate. Do not call it an observed loss.
- By default, leave each paper trade running until either TP/SL is hit or
  the **file ends**. End-of-file unresolved trades are `PENDING_END`, not
  forced losses or wins. To limit the scan to 20 subsequent candles, pass
  `--max-holding-bars 20`; otherwise unhit outcomes within the horizon are
  `NO_HIT` and excluded from resolved win rate.
- Each candidate signal is evaluated **independently**, even if another
  signal's hypothetical trade remains open. These are signal statistics,
  **not a single-position account/portfolio simulation**.

Run on **the exact same original OHLC dataset** as the saved reports:

```bash
git pull origin main
python -m finance_ai backtest-fixed-tpsl \
  --csv data/final_test.csv \
  --signals reports/v0.9-entries.csv \
  --tp 2.5 --sl 2.0 \
  --out reports/v0.9-fixed-tpsl.csv
```

Alternatively, for all v0.8 agree-on-reversal signals without v0.9 context:

```bash
python -m finance_ai backtest-fixed-tpsl \
  --csv data/final_test.csv \
  --signals reports/v0.8-test-20.csv \
  --out reports/v0.8-fixed-tpsl.csv
```

The **Excel-ready CSV** has one row per matched BUY/SELL signal: candle
UTC, direction, context/quality/old-v0.9 decision, entry at signal close,
absolute TP/SL price levels, time of first future wick hit, bars until
resolution, `TP`/`SL`/`AMBIGUOUS_SL_FIRST`/`NO_HIT`/`PENDING_END`,
per-oz gross signed move and MFE/MAE **up to first touch**. The accompanying
`.summary.json` splits **all 270 matched signals**, v0.9 quality-approved
setups, and previously queued v0.9 setups, further split into BUY/SELL.
It reports all counts, number resolved and a **conservative resolved win
rate** (`TP / (TP + SL + ambiguous)`). Existing v0.9 report
`quality_reason == accepted` is used **only for the optional comparison**;
it does not filter the full-signal total.

No real fills, bid/ask spread, commissions, gap slips, swaptime fees,
executed trade P&L, or single-position cash exposure are modeled. A
historical signal-close fill can be unavailable in real execution. Before
claiming a strategy has edge, use genuinely new data: the original
`final_test.csv` has already been inspected repeatedly.

### Optional M1 intrabar disambiguation (no model retraining)

**Keep M15 signals and the existing neural models exactly as they are.**
The confirmation candle's timestamp in our files is the **M15 opening
timestamp**, so its actual close/entry is **15 minutes later**.
For **M1 review**, the first eligible minute starts at the timestamp of
the **next M15 candle** (never the minutes inside the signal candle).

Only a candle in which the **first M15 TP/SL touch is ambiguous** needs
M1 inspection. M15 outcomes in which only TP or only SL was touched are
already decidable without finer data. Within each ambiguous M15 candle,
the M1 tester checks one-minute **HIGH/LOW shadows in chronological
order** for the first touch; if **both** levels are hit in one minute,
it is still ambiguous and counted as SL for the conservative rate.

```bash
git pull origin main
python -m finance_ai backtest-fixed-tpsl \
  --csv data/final_test.csv \
  --signals reports/v0.9-entries.csv \
  --m1-csv data/final_test_m1.csv \
  --tp 2.5 --sl 2 \
  --out reports/v0.9-fixed-tpsl-m1.csv
```

`--m1-csv` must use **M1 candles from the same broker and historical
period**, with the same Unix-UTC timestamp convention and exact prices
as the M15 file. Required columns: `time,open,high,low,close,timeframe`
(`timeframe=1`; `volume` optional). CSV or Excel inputs supported.
Timestamps must be ascending whole-minute Unix seconds; we **do not**
auto-shift timezones or guess whether timestamps mean candle start/end.
The M15 source still requires its existing `timeframe=15` schema.

An ambiguous M15 candle is resolved using M1 **only if all 15 M1
minute candles are present** and their aggregated OHLC matches the M15
candle within a configurable tolerance (default **$0.02/oz**;
`--m1-price-tolerance 0.05` to raise if justified by known export
precision). If missing or mismatched, result becomes
`M1_DATA_INCOMPLETE` or `M1_PRICE_MISMATCH`: **not counted** as win
or loss. Report fields `m15_result_before_m1`, `m1_resolution`,
`m1_first_touch_utc`, and `m1_bars_in_ambiguous_candle` show what
happened. The summary includes `m15_originally_ambiguous`,
`m1_resolved`, and `m1_unresolved_missing_or_mismatch`. This
strict completeness check may leave legitimate low-activity minutes
unresolved; it is conservative about dataset integrity.

All trades remain **hypothetical fills at the M15 signal CLOSE**,
not guaranteed broker fills. No spread, slippage, commissions, or tick
path inside a single M1 minute; this is a **wick-first-touch signal
audit**, not demonstrated executable profitability. No v0.7/v0.8
weights, original signals, or old M15-only reports are changed.
