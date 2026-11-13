"""Finance AI v0.7: recognize *current* M15 trend from completed candles.

Two outputs only (uptrend/downtrend). A causal trailing-price regression defines
an explicit, reproducible training reference; it is NOT objective trend truth
and validation measures agreement with that reference, not forecast skill.
No future close, pivot, trade entry, or reversal target is used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import Settings, fit_scaler, split_positions, transform
from .features import read_candles
from .network import MLP

MODEL_VERSION = "0.7.0"
TIMEFRAME_MINUTES = 15
LOOKBACK = 40
LABEL_PERIOD = 32
BB_PERIOD = 20
BB_STD = 2.0
EMA_PERIOD = 10
AROON_PERIOD = 25
ATR_PERIOD = 14
DEFAULT_HIDDEN = (512, 256, 128, 64)
WEAK_SLOPE_ATR = 0.50
CLASS_NAMES = ("downtrend", "uptrend")
FEATURE_NAMES = (
    "open_return", "high_return", "low_return", "close_return", "log_volume",
    "bb_percent_b", "bb_bandwidth", "bb_middle_slope", "ema10_vs_bb_middle",
    "ema10_slope", "aroon_up", "aroon_down", "aroon_spread", "atr14_ratio", "log_gap",
)


@dataclass(frozen=True)
class TrendData:
    timestamps: np.ndarray
    features: np.ndarray
    closes: np.ndarray
    atr14: np.ndarray


def aroon_channels(high: np.ndarray, low: np.ndarray, period: int = AROON_PERIOD):
    """ta==0.11.0 convention: highest/lowest in period+1 past bars.

    np.argmax/argmin choose the earliest index for tied extreme values,
    consistent with ta's rolling implementation. Returns fractions in [0,1].
    """
    high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    if period < 1 or high.ndim != 1 or high.shape != low.shape:
        raise ValueError("Aroon requires aligned 1-D high and low arrays")
    up = np.full(len(high), np.nan)
    down = np.full(len(low), np.nan)
    if len(high) > period:
        from numpy.lib.stride_tricks import sliding_window_view
        h = sliding_window_view(high, period + 1)
        l = sliding_window_view(low, period + 1)
        up[period:] = h.argmax(axis=1) / period
        down[period:] = l.argmin(axis=1) / period
    return up, down


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder's RMA: seed with mean of first `period` true ranges."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if (period < 1 or high.ndim != 1 or high.shape != low.shape or
            high.shape != close.shape):
        raise ValueError("ATR requires aligned 1-D OHLC arrays")
    prev = np.concatenate(([close[0]], close[:-1])) if len(close) else close
    tr = np.maximum.reduce((high-low, np.abs(high-prev), np.abs(low-prev)))
    atr = np.full(len(close), np.nan)
    if len(close) >= period:
        atr[period-1] = tr[:period].mean()
        for i in range(period, len(close)):
            atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    return atr


def prepare_trend(path: str | Path) -> TrendData:
    """Construct 15 causal candle features; no MACD, RSI, or SMA50."""
    df = read_candles(path)
    if len(df) < 100:
        raise ValueError("At least 100 completed M15 candles are required")
    close = df["close"].astype(float)
    previous = close.shift(1)
    mid = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    sigma = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    upper = mid + BB_STD * sigma
    lower = mid - BB_STD * sigma
    span = upper - lower
    # Flat markets should map exactly to the center of a zero-width band.
    percent_b = (close - lower) / span.replace(0, np.nan)
    percent_b = percent_b.where(span != 0, 0.5)
    ema = close.ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean()
    up, down = aroon_channels(df["high"].to_numpy(), df["low"].to_numpy())
    atr = wilder_atr(df["high"].to_numpy(), df["low"].to_numpy(), close.to_numpy())
    f = pd.DataFrame(index=df.index)
    for column in ("open", "high", "low", "close"):
        f[column + "_return"] = df[column] / previous - 1.0
    f["log_volume"] = np.log1p(df["volume"])
    f["bb_percent_b"] = percent_b
    f["bb_bandwidth"] = span / mid
    f["bb_middle_slope"] = mid / mid.shift(1) - 1.0
    f["ema10_vs_bb_middle"] = (ema - mid) / mid
    f["ema10_slope"] = ema / ema.shift(1) - 1.0
    f["aroon_up"] = up
    f["aroon_down"] = down
    f["aroon_spread"] = up - down
    f["atr14_ratio"] = atr / close.to_numpy()
    f["log_gap"] = np.log1p(df["time"].diff() / (TIMEFRAME_MINUTES*60))
    f = f[list(FEATURE_NAMES)]
    valid = np.isfinite(f.to_numpy(dtype=np.float64)).all(axis=1)
    if not valid.any():
        raise ValueError("Not enough history to initialize indicators")
    first = int(np.argmax(valid))
    if not valid[first:].all():
        raise ValueError("Nonfinite trend features after indicator warmup")
    return TrendData(
        timestamps=df["time"].to_numpy(dtype=np.int64)[first:],
        features=f.to_numpy(dtype=np.float64)[first:],
        closes=close.to_numpy(dtype=np.float64)[first:],
        atr14=atr[first:].copy(),
    )


def price_trend_reference(data: TrendData, ends: np.ndarray,
                          period: int = LABEL_PERIOD):
    """Past-only OLS slope of log(close) over `period` bars, in ATR units.

    Returns (binary labels, signed strength). Zero slope is forced up because
    the user explicitly requested two classes; weak strengths are flagged.
    This synthetic target is not a validated ground-truth trend annotation.
    """
    ends = np.asarray(ends, dtype=np.int64)
    if period < 2 or ends.ndim != 1 or np.any(ends < period - 1) or np.any(ends >= len(data.closes)):
        raise ValueError("Invalid end indices for price-only trend reference")
    from numpy.lib.stride_tricks import sliding_window_view
    log_closes = np.log(data.closes)
    windows = sliding_window_view(log_closes, period)[ends - (period-1)]
    t = np.arange(period, dtype=float)
    t -= t.mean()
    slope = (windows @ t) / (t @ t)
    atr_price = data.atr14[ends]
    if not np.isfinite(atr_price).all() or np.any(atr_price < 0):
        raise ValueError("ATR values must be finite and nonnegative")
    # For a perfectly flat market, use zero strength (not a division error).
    strength = slope * (period-1) * data.closes[ends] / np.maximum(atr_price, 1e-12)
    return (slope >= 0).astype(np.float64), strength


def trend_samples(data: TrendData):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(data.closes) < LOOKBACK + 50:
        raise ValueError("Not enough candles for trend training")
    ends = np.arange(LOOKBACK-1, len(data.closes))
    # slide axis=0 yields (samples, features, lookback); transpose to (n,lookback,f)
    windows = sliding_window_view(data.features, LOOKBACK, axis=0).transpose(0, 2, 1)
    labels, strength = price_trend_reference(data, ends)
    return windows, labels, strength, ends


def trend_metrics(model: MLP, x: np.ndarray, y: np.ndarray) -> dict:
    probabilities = np.clip(model.forward(x), 1e-12, 1-1e-12)
    pred = (probabilities >= 0.5).astype(int)
    truth = y.astype(int)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (truth, pred), 1)
    precision = {}
    recall = {}
    f1 = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = confusion[i, i]
        actual = confusion[i].sum()
        predicted = confusion[:, i].sum()
        r = float(tp / actual) if actual else 0.0
        p = float(tp / predicted) if predicted else 0.0
        recall[cls], precision[cls] = r, p
        f1[cls] = 2*r*p/(r+p) if r+p else 0.0
    return {
        "loss": float(np.mean(-y*np.log(probabilities) - (1-y)*np.log1p(-probabilities))),
        "accuracy": float(np.mean(truth == pred)),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(list(f1.values()))),
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion.tolist(),
        "samples": int(len(y)),
        "up_fraction": float(y.mean()),
        "predicted_up_fraction": float(pred.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1-y.mean())),
    }


def train_trend(csv: str, out: str, settings: Settings | None = None,
                hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> dict:
    cfg = settings or Settings()
    if (cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or
            not cfg.seeds or not hidden or any(not isinstance(n, int) or n < 1 for n in hidden)):
        raise ValueError("Invalid training settings")
    data = prepare_trend(csv)
    x, y, strengths, ends = trend_samples(data)
    a, b = split_positions(len(y))
    # Only the earliest 70% contributes scaler statistics and weight fitting.
    mu, std = fit_scaler(x[:a])
    xt = transform(x[:a], mu, std)
    xv = transform(x[a:b], mu, std)
    yt, yv = y[:a], y[a:b]
    sizes = (LOOKBACK*len(FEATURE_NAMES), *hidden, 1)
    best_loss, best_state, best_seed, best_epoch = np.inf, None, None, None
    trials = []
    for seed in cfg.seeds:
        model = MLP(sizes=sizes, seed=seed)
        rng = np.random.default_rng(seed)
        trial_loss, trial_state, trial_epoch = np.inf, None, None
        misses = 0
        for epoch in range(1, cfg.epochs+1):
            order = rng.permutation(len(xt))
            for offset in range(0, len(order), cfg.batch_size):
                indices = order[offset:offset+cfg.batch_size]
                model.train_batch(xt[indices], yt[indices], cfg.lr, cfg.weight_decay)
            score = trend_metrics(model, xv, yv)["loss"]
            if score < trial_loss - 1e-6:
                trial_loss, trial_state, trial_epoch = score, model.state_dict(), epoch
                misses = 0
            else:
                misses += 1
            if misses >= cfg.patience:
                break
        trials.append({"seed": seed, "best_epoch": trial_epoch, "validation_loss": float(trial_loss)})
        if trial_loss < best_loss:
            best_loss, best_state, best_seed, best_epoch = trial_loss, trial_state, seed, trial_epoch
    best = MLP(sizes=sizes, seed=best_seed)
    best.load_state_dict(best_state)
    prior = float(np.clip(yt.mean(), 1e-12, 1-1e-12))
    validation = trend_metrics(best, xv, yv)
    validation["train_prior_baseline_loss"] = float(
        np.mean(-yv*np.log(prior)-(1-yv)*np.log1p(-prior)))
    validation["persistence_baseline_accuracy"] = float(np.mean(y[a:b] == y[a-1:b-1]))
    ema_index = FEATURE_NAMES.index("ema10_vs_bb_middle")
    validation["ema10_vs_bb_middle_baseline_accuracy"] = float(np.mean(
        (x[a:b, -1, ema_index] >= 0) == yv.astype(bool)))
    report = {
        "model_version": MODEL_VERSION,
        "task": "current_trend_classification_not_forecasting",
        "classes": list(CLASS_NAMES),
        "reference_label": "sign of trailing 32-bar OLS slope of log(close), no future bars",
        "reference_is_proxy_not_ground_truth": True,
        "architecture": list(sizes),
        "parameters": int(sum(n*m + m for n,m in zip(sizes[:-1], sizes[1:]))),
        "selected_seed": best_seed,
        "selected_epoch": best_epoch,
        "trials": trials,
        "train": trend_metrics(best, xt, yt),
        "validation": validation,
        "weak_reference_fraction_train": float(np.mean(np.abs(strengths[:a]) < WEAK_SLOPE_ATR)),
        "weak_reference_fraction_validation": float(np.mean(np.abs(strengths[a:b]) < WEAK_SLOPE_ATR)),
        "split_windows": {"train": a, "validation": b-a,
                          "internal_holdout_reserved": len(y)-b},
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
    }
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root/"model.npz", **best_state)
    cfg_saved = {
        "model_version": MODEL_VERSION, "task": report["task"],
        "architecture": list(sizes), "lookback": LOOKBACK,
        "label_period": LABEL_PERIOD, "timeframe": TIMEFRAME_MINUTES,
        "features": list(FEATURE_NAMES), "classes": list(CLASS_NAMES),
        "indicators": {"bollinger": [BB_PERIOD, BB_STD], "ema": EMA_PERIOD,
                       "aroon": AROON_PERIOD, "atr": ATR_PERIOD},
        "weak_trend_threshold_atr": WEAK_SLOPE_ATR,
        "training": asdict(cfg),
    }
    prep_saved = {"version": 7, "mean": mu.tolist(), "std": std.tolist(),
                  "reference_definition": report["reference_label"],
                  "time_zone": "UTC; Unix seconds"}
    (root/"config.json").write_text(json.dumps(cfg_saved, indent=2), encoding="utf-8")
    (root/"preprocessing.json").write_text(json.dumps(prep_saved, indent=2), encoding="utf-8")
    (root/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def detect_trend(csv: str, model_dir: str) -> dict:
    root = Path(model_dir)
    cfg = json.loads((root/"config.json").read_text(encoding="utf-8"))
    prep = json.loads((root/"preprocessing.json").read_text(encoding="utf-8"))
    sizes = cfg.get("architecture", [])
    required_indicators = {"bollinger": [BB_PERIOD, BB_STD], "ema": EMA_PERIOD,
                           "aroon": AROON_PERIOD, "atr": ATR_PERIOD}
    if (cfg.get("model_version") != MODEL_VERSION or cfg.get("task") != "current_trend_classification_not_forecasting" or
            cfg.get("lookback") != LOOKBACK or cfg.get("label_period") != LABEL_PERIOD or
            cfg.get("timeframe") != TIMEFRAME_MINUTES or cfg.get("features") != list(FEATURE_NAMES) or
            cfg.get("classes") != list(CLASS_NAMES) or cfg.get("indicators") != required_indicators or
            cfg.get("weak_trend_threshold_atr") != WEAK_SLOPE_ATR or
            prep.get("version") != 7 or not isinstance(sizes, list) or len(sizes) < 2 or
            sizes[0] != LOOKBACK*len(FEATURE_NAMES) or sizes[-1] != 1 or
            any(not isinstance(n, int) or n < 1 for n in sizes) or
            len(prep.get("mean", [])) != len(FEATURE_NAMES) or
            len(prep.get("std", [])) != len(FEATURE_NAMES)):
        raise ValueError("Incompatible v0.7 trend model configuration")
    data = prepare_trend(csv)
    if len(data.features) < LOOKBACK:
        raise ValueError("Insufficient completed candles for trend detection")
    mu = np.asarray(prep["mean"], dtype=float)
    std = np.asarray(prep["std"], dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid training scaler parameters")
    model = MLP(sizes=tuple(sizes))
    with np.load(root/"model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    x = transform(data.features[-LOOKBACK:][None, ...], mu, std)
    p_up = float(model.forward(x)[0])
    current_y, signed_strength = price_trend_reference(
        data, np.array([len(data.closes)-1]))
    last = data.features[-1]
    prev = data.features[-2]
    ema_idx = FEATURE_NAMES.index("ema10_vs_bb_middle")
    crossed_up = prev[ema_idx] <= 0 < last[ema_idx]
    crossed_down = prev[ema_idx] >= 0 > last[ema_idx]
    cross = "bullish" if crossed_up else "bearish" if crossed_down else "none"
    # Expose a weak-trend *flag*, never an invented third class.
    return {
        "model_version": MODEL_VERSION,
        "task": "current_trend_detection",
        "last_closed_candle_utc": pd.Timestamp(int(data.timestamps[-1]), unit="s", tz="UTC").isoformat(),
        "trend": CLASS_NAMES[int(p_up >= 0.5)],
        "up_score_uncalibrated": p_up,
        "down_score_uncalibrated": 1.0-p_up,
        "weak_price_trend": bool(abs(signed_strength[0]) < WEAK_SLOPE_ATR),
        "price_slope_in_atr_units": float(signed_strength[0]),
        "indicators": {
            "bollinger_percent_b": float(last[FEATURE_NAMES.index("bb_percent_b")]),
            "bollinger_bandwidth_ratio": float(last[FEATURE_NAMES.index("bb_bandwidth")]),
            "ema10_minus_bb_middle_ratio": float(last[ema_idx]),
            "ema10_bb_middle_cross": cross,
            "aroon_up": float(100*last[FEATURE_NAMES.index("aroon_up")]),
            "aroon_down": float(100*last[FEATURE_NAMES.index("aroon_down")]),
            "atr14_ratio": float(last[FEATURE_NAMES.index("atr14_ratio")]),
        },
        "note": "Current historical trend proxy, not future prediction. Scores are uncalibrated.",
    }
