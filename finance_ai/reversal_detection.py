"""v0.8: causal *confirmed* swing-reversal candidates, not future turning-point forecasts.

Reference labels are an explicit, price-only past-pattern proxy. The network
learns that reference using RSI14 / Bollinger20,2 / MACD12,26,9 / candle
features. A confirmed event is available only AFTER the confirmation candle
closes; it is never projected backward as a tradable signal at the pivot.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .atr import SoftmaxMLP
from .experiment import Settings, fit_scaler, split_positions, transform
from .features import read_candles
from .trend_detection import wilder_atr

VERSION = "0.8.0"
TIMEFRAME = 15
LOOKBACK = 40
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
MACD = (12, 26, 9)
ATR_PERIOD = 14
PRETREND_PERIOD = 20
SWING_PERIOD = 9
PRETREND_ATR = 0.35
CONFIRM_ATR = 0.20
PURGE = LOOKBACK  # prevents any shared input candles across chronological splits
DEFAULT_HIDDEN = (512, 256, 128, 64)
CLASSES = ("none", "bullish_reversal", "bearish_reversal")
FEATURES = (
    "open_return", "high_return", "low_return", "close_return", "log_volume",
    "rsi14", "bb_percent_b", "bb_bandwidth", "bb_upper_touch", "bb_lower_touch",
    "bb_middle_distance", "macd_ratio", "macd_signal_ratio", "macd_hist_ratio",
    "macd_hist_change", "body_atr", "upper_wick_atr", "lower_wick_atr",
    "atr14_ratio", "log_gap",
)
assert len(FEATURES) == 20


@dataclass(frozen=True)
class ReversalData:
    timestamps: np.ndarray
    features: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    atr: np.ndarray


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI with causal exponential smoothing, gracefully handles flats."""
    change = close.diff()
    ups = change.clip(lower=0)
    downs = (-change).clip(lower=0)
    # Seed the Wilder average using the first period deltas.
    avg_up = np.full(len(close), np.nan)
    avg_down = np.full(len(close), np.nan)
    if len(close) > period:
        avg_up[period] = ups.iloc[1:period+1].mean()
        avg_down[period] = downs.iloc[1:period+1].mean()
        for i in range(period+1, len(close)):
            avg_up[i] = (avg_up[i-1]*(period-1)+float(ups.iloc[i]))/period
            avg_down[i] = (avg_down[i-1]*(period-1)+float(downs.iloc[i]))/period
    result = np.full(len(close), np.nan)
    idx = np.arange(len(close)) >= period
    positive = avg_up[idx]
    negative = avg_down[idx]
    result[idx] = np.where(negative > 0, 100.0*positive/np.maximum(positive+negative, 1e-15),
                           np.where(positive > 0, 100.0, 50.0))
    return pd.Series(result, index=close.index)


def prepare_reversal(path: str | Path) -> ReversalData:
    """Only completed candles; no centered rolling statistics or future pivots."""
    df = read_candles(path)
    if len(df) < 110:
        raise ValueError("At least 110 closed M15 candles are required")
    close = df["close"].astype(float)
    prev = close.shift(1)
    mid = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    stdev = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    upper, lower = mid+BB_STD*stdev, mid-BB_STD*stdev
    band = upper-lower
    pctb = ((close-lower)/band.where(band != 0)).where(band != 0, 0.5)
    fast = close.ewm(span=MACD[0], adjust=False, min_periods=MACD[0]).mean()
    slow = close.ewm(span=MACD[1], adjust=False, min_periods=MACD[1]).mean()
    macd = fast-slow
    signal = macd.ewm(span=MACD[2], adjust=False, min_periods=MACD[2]).mean()
    hist = macd-signal
    atr = wilder_atr(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
                     close.to_numpy(dtype=float), ATR_PERIOD)
    safe_atr = np.maximum(atr, 1e-12)
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    f = pd.DataFrame(index=df.index)
    for name in ("open", "high", "low", "close"):
        f[name+"_return"] = df[name]/prev-1.0
    f["log_volume"] = np.log1p(df["volume"])
    f["rsi14"] = _rsi(close)/100.0
    f["bb_percent_b"] = pctb
    f["bb_bandwidth"] = band/mid
    f["bb_upper_touch"] = (df["high"] >= upper).astype(float)
    f["bb_lower_touch"] = (df["low"] <= lower).astype(float)
    f["bb_middle_distance"] = close/mid-1.0
    f["macd_ratio"] = macd/close
    f["macd_signal_ratio"] = signal/close
    f["macd_hist_ratio"] = hist/close
    f["macd_hist_change"] = hist.diff()/close
    f["body_atr"] = (close-df["open"])/safe_atr
    f["upper_wick_atr"] = (df["high"]-body_high)/safe_atr
    f["lower_wick_atr"] = (body_low-df["low"])/safe_atr
    f["atr14_ratio"] = atr/close
    f["log_gap"] = np.log1p(df["time"].diff()/(TIMEFRAME*60))
    f = f[list(FEATURES)]
    values = f.to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        raise ValueError("Indicator warmup failed")
    first = int(valid.argmax())
    if not valid[first:].all():
        raise ValueError("Nonfinite indicator values after warmup")
    return ReversalData(
        timestamps=df["time"].to_numpy(dtype=np.int64)[first:],
        features=values[first:],
        close=close.to_numpy(dtype=float)[first:],
        high=df["high"].to_numpy(dtype=float)[first:],
        low=df["low"].to_numpy(dtype=float)[first:],
        atr=atr[first:],
    )


def reversal_reference(data: ReversalData, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label confirmation candle t, pivot candidate t-1; all bars <= t.

    A bullish candidate follows a descending prior 20-bar OLS price slope,
    a prior local 9-bar low, then a >=0.20 ATR close rebound without a new low.
    The bearish rule is symmetric. Neither RSI/Bollinger/MACD defines labels.
    """
    ends = np.asarray(ends, dtype=np.int64)
    if ends.ndim != 1 or (ends < max(PRETREND_PERIOD+1, SWING_PERIOD+1)).any() or (ends >= len(data.close)).any():
        raise ValueError("Invalid reversal-reference candle endpoints")
    t = np.arange(PRETREND_PERIOD, dtype=float)
    t -= t.mean()
    from numpy.lib.stride_tricks import sliding_window_view
    slopes = sliding_window_view(np.log(data.close), PRETREND_PERIOD)[ends-(PRETREND_PERIOD+1)] @ t / (t @ t)
    prior = ends-2
    strength = slopes*(PRETREND_PERIOD-1)*data.close[prior]/np.maximum(data.atr[prior], 1e-12)
    # The local swing candidate is candle end-1. Include it among the 9 lows/highs.
    lows = sliding_window_view(data.low, SWING_PERIOD)[ends-SWING_PERIOD]
    highs = sliding_window_view(data.high, SWING_PERIOD)[ends-SWING_PERIOD]
    pivot = ends-1
    rise = data.close[ends]-data.close[pivot]
    ref = np.zeros(len(ends), dtype=np.int64)
    bull = ((strength <= -PRETREND_ATR) & (data.low[pivot] <= lows.min(axis=1)) &
            (data.low[ends] >= data.low[pivot]) &
            (rise >= CONFIRM_ATR*np.maximum(data.atr[pivot], 1e-12)))
    bear = ((strength >= PRETREND_ATR) & (data.high[pivot] >= highs.max(axis=1)) &
            (data.high[ends] <= data.high[pivot]) &
            (-rise >= CONFIRM_ATR*np.maximum(data.atr[pivot], 1e-12)))
    ref[bull] = 1
    ref[bear] = 2
    return ref, strength


def reversal_samples(data: ReversalData):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(data.close) < LOOKBACK+60:
        raise ValueError("Not enough candles for reversal training")
    ends = np.arange(LOOKBACK-1, len(data.close))
    x = sliding_window_view(data.features, LOOKBACK, axis=0).transpose(0, 2, 1)
    y, pretrend = reversal_reference(data, ends)
    return x, y, pretrend, ends


class BalancedSoftmax(SoftmaxMLP):
    """Train with inverse-sqrt support weights; save weights for reproducibility."""

    def __init__(self, sizes: tuple[int, ...], class_weights: np.ndarray, seed: int = 42):
        super().__init__(sizes, seed=seed)
        self.class_weights = np.asarray(class_weights, dtype=float)
        if self.class_weights.shape != (3,) or not np.isfinite(self.class_weights).all() or np.any(self.class_weights <= 0):
            raise ValueError("Three positive class weights required")

    def gradients(self, x, y):
        z, probs, acts = self.forward(x, cache=True)
        labels = np.asarray(y, dtype=np.int64)
        if len(x) == 0 or labels.shape != (len(x),) or np.any((labels < 0) | (labels > 2)):
            raise ValueError("Expected a 0/1/2 reversal label for every sample")
        weight = self.class_weights[labels]
        centered = z-z.max(axis=1, keepdims=True)
        ce = np.log(np.exp(centered).sum(axis=1))-centered[np.arange(len(labels)), labels]
        loss = float(np.sum(weight*ce)/weight.sum())
        delta = probs.copy()
        delta[np.arange(len(labels)), labels] -= 1.0
        delta *= (weight/weight.sum())[:, None]
        gw, gb = [None]*len(self.weights), [None]*len(self.biases)
        for i in reversed(range(len(self.weights))):
            gw[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0, keepdims=True)
            if i:
                delta = (delta @ self.weights[i].T)*(acts[i] > 0)
        return loss, gw, gb


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(float)
    if np.any(counts < 5):
        raise ValueError("Not enough bullish/bearish reference events for training (min 5 per class)")
    weights = 1.0/np.sqrt(counts/counts.sum())
    weights = np.minimum(weights/weights[0], 8.0)
    return weights


def reversal_metrics(model: SoftmaxMLP, x: np.ndarray, labels: np.ndarray,
                     weights: np.ndarray) -> dict:
    p = model.forward(x)
    pred = p.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    precision, recall, f1 = {}, {}, {}
    for i, name in enumerate(CLASSES):
        tp = confusion[i, i]
        pden, rden = confusion[:, i].sum(), confusion[i].sum()
        pr = float(tp/pden) if pden else 0.0
        re = float(tp/rden) if rden else 0.0
        precision[name], recall[name] = pr, re
        f1[name] = 2*pr*re/(pr+re) if pr+re else 0.0
    losses = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-15, 1.0))
    w = weights[labels]
    return {
        "loss": float(losses.mean()),
        "weighted_loss": float(np.sum(w*losses)/w.sum()),
        "accuracy": float(np.mean(labels == pred)),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(list(f1.values()))),
        "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": confusion.tolist(),
        "samples": int(len(labels)),
        "reference_counts": dict(zip(CLASSES, map(int, np.bincount(labels, minlength=3)))),
        "predicted_counts": dict(zip(CLASSES, map(int, np.bincount(pred, minlength=3)))),
        "majority_baseline_accuracy": float(np.bincount(labels, minlength=3).max()/len(labels)),
    }


def _selections(n: int) -> tuple[slice, slice, slice]:
    a, b = split_positions(n)
    # Keep only ~70% of the full timeline for training; skip 40 between splits.
    if a < 50 or b-a-PURGE < 20 or n-b-PURGE < 20:
        raise ValueError("Not enough samples after chronological boundary purge")
    return slice(0, a), slice(a+PURGE, b), slice(b+PURGE, n)


def train_reversal(csv: str, out: str, settings: Settings | None = None,
                   hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> dict:
    cfg = settings or Settings()
    if (cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or not cfg.seeds or
            any(not isinstance(s, int) for s in cfg.seeds) or not hidden or
            any(not isinstance(n, int) or n < 1 for n in hidden)):
        raise ValueError("Invalid reversal training settings")
    data = prepare_reversal(csv)
    x, y, strength, ends = reversal_samples(data)
    train_sel, val_sel, hold_sel = _selections(len(x))
    mu, sigma = fit_scaler(x[train_sel])
    xt, xv = transform(x[train_sel], mu, sigma), transform(x[val_sel], mu, sigma)
    yt, yv = y[train_sel], y[val_sel]
    weights = class_weights(yt)
    sizes = (LOOKBACK*len(FEATURES), *hidden, 3)
    best_loss, best_state, best_seed, best_epoch = np.inf, None, None, None
    trials = []
    for seed in cfg.seeds:
        model = BalancedSoftmax(sizes, weights, seed=seed)
        rng = np.random.default_rng(seed)
        trial_loss, trial_epoch, trial_state, misses = np.inf, None, None, 0
        for epoch in range(1, cfg.epochs+1):
            order = rng.permutation(len(xt))
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start:start+cfg.batch_size]
                model.train_batch(xt[idx], yt[idx], cfg.lr, cfg.weight_decay)
            loss = reversal_metrics(model, xv, yv, weights)["weighted_loss"]
            print(f"[v0.8] seed={seed} epoch={epoch}/{cfg.epochs} "
                  f"validation_weighted_loss={loss:.6f}", file=sys.stderr, flush=True)
            if loss < trial_loss-1e-6:
                trial_loss, trial_epoch, trial_state, misses = loss, epoch, model.state_dict(), 0
            else:
                misses += 1
            if misses >= cfg.patience:
                break
        trials.append({"seed": seed, "best_epoch": trial_epoch, "validation_weighted_loss": float(trial_loss)})
        if trial_loss < best_loss:
            best_loss, best_state, best_seed, best_epoch = trial_loss, trial_state, seed, trial_epoch
    chosen = BalancedSoftmax(sizes, weights, seed=best_seed)
    chosen.load_state_dict(best_state)
    report = {
        "model_version": VERSION,
        "task": "past_only_confirmed_swing_reversal_reference_not_forecasting",
        "classes": list(CLASSES),
        "label_definition": ("20-bar pre-move price slope ending at t-2; preceding 9-bar extremum at t-1; "
                             ">=0.20 ATR counter-close and no new extreme at confirmation t; "
                             "directional prior strength >=0.35 ATR"),
        "reference_is_proxy_not_ground_truth": True,
        "reference_causal": True,
        "signal_time": "confirmation candle t close; candidate pivot candle t-1 (not tradable at t-1)",
        "architecture": list(sizes),
        "parameters": int(sum(a*b+b for a,b in zip(sizes[:-1], sizes[1:]))),
        "selected_seed": best_seed, "selected_epoch": best_epoch,
        "class_weights_train_only": dict(zip(CLASSES, map(float, weights))),
        "trials": trials,
        "train": reversal_metrics(chosen, xt, yt, weights),
        "validation": reversal_metrics(chosen, xv, yv, weights),
        "split_windows": {"train": len(y[train_sel]), "validation": len(y[val_sel]),
                          "internal_holdout_reserved": len(y[hold_sel]),
                          "purge_each_boundary": PURGE},
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
    }
    # Reference events are rare: always print class-specific metrics, never
    # interpret overall accuracy as proof of quality or trading profitability.
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root/"model.npz", **best_state)
    config = {
        "model_version": VERSION, "task": report["task"], "architecture": list(sizes),
        "timeframe": TIMEFRAME, "lookback": LOOKBACK,
        "features": list(FEATURES), "classes": list(CLASSES),
        "indicators": {"rsi": RSI_PERIOD, "bollinger": [BB_PERIOD, BB_STD],
                       "macd": list(MACD), "atr": ATR_PERIOD},
        "reference": {"pretrend_period": PRETREND_PERIOD, "swing_period": SWING_PERIOD,
                      "pretrend_atr": PRETREND_ATR, "confirm_atr": CONFIRM_ATR},
        "training": asdict(cfg),
    }
    preprocessing = {"version": 8, "mean": mu.tolist(), "std": sigma.tolist(),
                     "class_weights": weights.tolist(),
                     "note": "Fit on development training slice only; all features causal"}
    (root/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (root/"preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    (root/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def detect_reversal(csv: str, model_dir: str) -> dict:
    root = Path(model_dir)
    config = json.loads((root/"config.json").read_text(encoding="utf-8"))
    prep = json.loads((root/"preprocessing.json").read_text(encoding="utf-8"))
    sizes = config.get("architecture")
    if (config.get("model_version") != VERSION or
            config.get("task") != "past_only_confirmed_swing_reversal_reference_not_forecasting" or
            config.get("timeframe") != TIMEFRAME or config.get("lookback") != LOOKBACK or
            config.get("features") != list(FEATURES) or config.get("classes") != list(CLASSES) or
            config.get("indicators") != {"rsi": RSI_PERIOD, "bollinger": [BB_PERIOD, BB_STD],
                                         "macd": list(MACD), "atr": ATR_PERIOD} or
            config.get("reference") != {"pretrend_period": PRETREND_PERIOD,
                                        "swing_period": SWING_PERIOD,
                                        "pretrend_atr": PRETREND_ATR, "confirm_atr": CONFIRM_ATR} or
            prep.get("version") != 8 or not isinstance(sizes, list) or len(sizes) < 2 or
            sizes[0] != LOOKBACK*len(FEATURES) or sizes[-1] != 3 or
            any(not isinstance(n, int) or n < 1 for n in sizes) or
            len(prep.get("mean", [])) != len(FEATURES) or
            len(prep.get("std", [])) != len(FEATURES)):
        raise ValueError("Incompatible v0.8 reversal checkpoint")
    data = prepare_reversal(csv)
    if len(data.close) < LOOKBACK:
        raise ValueError("Not enough completed candles")
    mean = np.asarray(prep["mean"], dtype=float)
    std = np.asarray(prep["std"], dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid scaler values")
    model = BalancedSoftmax(tuple(sizes), np.ones(3))
    with np.load(root/"model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    x = transform(data.features[-LOOKBACK:][None, ...], mean, std)
    p = model.forward(x)[0]
    ref, signed_strength = reversal_reference(data, np.array([len(data.close)-1]))
    last = data.features[-1]
    vals = dict(zip(CLASSES, map(float, p)))
    rsi = float(last[FEATURES.index("rsi14")]*100)
    bb = float(last[FEATURES.index("bb_percent_b")])
    upper_touch = bool(last[FEATURES.index("bb_upper_touch")])
    lower_touch = bool(last[FEATURES.index("bb_lower_touch")])
    hist_idx = FEATURES.index("macd_hist_ratio")
    prev_hist = float(data.features[-2, hist_idx])
    curr_hist = float(last[hist_idx])
    macd_cross = ("bullish" if prev_hist <= 0 < curr_hist else
                  "bearish" if prev_hist >= 0 > curr_hist else "none")
    return {
        "model_version": VERSION,
        "task": "current_closed_candle_reversal_candidate_not_future_forecast",
        "last_closed_candle_utc": pd.Timestamp(int(data.timestamps[-1]), unit="s", tz="UTC").isoformat(),
        "candidate_pivot_candle_utc": pd.Timestamp(int(data.timestamps[-2]), unit="s", tz="UTC").isoformat(),
        "detected_class": CLASSES[int(p.argmax())],
        "scores_uncalibrated": vals,
        "price_only_confirmed_reference": CLASSES[int(ref[0])],
        "pre_move_direction": "uptrend" if signed_strength[0] > 0 else "downtrend" if signed_strength[0] < 0 else "flat",
        "pre_move_strength_atr": float(signed_strength[0]),
        "indicators": {
            "rsi14": rsi,
            "rsi_below_20": rsi < 20,
            "rsi_above_60": rsi > 60,
            "rsi_above_70": rsi > 70,
            "bollinger_percent_b": bb,
            "bollinger_upper_touch": upper_touch,
            "bollinger_lower_touch": lower_touch,
            "bollinger_upper_reentry": upper_touch and bb < 1.0,
            "bollinger_lower_reentry": lower_touch and bb > 0.0,
            "macd_hist_ratio": curr_hist,
            "macd_signal_cross": macd_cross,
            "macd_hist_change": float(last[FEATURES.index("macd_hist_change")]),
        },
        "note": "Past-only confirmation candidate, not a real-time advance warning or trading signal; scores are not calibrated.",
    }
