"""Visual validation of v0.7 current-trend recognition on completed M15 candles.

Plots only the chronological VALIDATION segment of a development CSV. It never
reads the reserved internal holdout or final_test.csv for predictions or scoring.
The model is compared with a constructed, past-only price-slope reference,
not independent ground-truth labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import split_positions, transform
from .features import read_candles
from .network import MLP
from .trend_detection import (
    ATR_PERIOD, AROON_PERIOD, BB_PERIOD, BB_STD, CLASS_NAMES,
    EMA_PERIOD, FEATURE_NAMES, LABEL_PERIOD, LOOKBACK, MODEL_VERSION,
    TIMEFRAME_MINUTES, WEAK_SLOPE_ATR, prepare_trend, price_trend_reference,
)


def validation_chart_data(csv: str, model_dir: str, candles: int = 500) -> dict:
    """Load a trained v0.7 model; infer only the last N VALIDATION samples.

    All predictions, chart prices and reference labels end on the SAME closed
    candle. Chronological split is identical to `train_trend`; the last 15%
    internal holdout is not loaded into inference windows.
    """
    if not isinstance(candles, int) or candles < 1:
        raise ValueError("--candles must be a positive integer")

    root = Path(model_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    architecture = config.get("architecture")
    indicators = {"bollinger": [BB_PERIOD, BB_STD], "ema": EMA_PERIOD,
                  "aroon": AROON_PERIOD, "atr": ATR_PERIOD}
    if (config.get("model_version") != MODEL_VERSION or
            config.get("task") != "current_trend_classification_not_forecasting" or
            config.get("lookback") != LOOKBACK or
            config.get("label_period") != LABEL_PERIOD or
            config.get("timeframe") != TIMEFRAME_MINUTES or
            config.get("features") != list(FEATURE_NAMES) or
            config.get("classes") != list(CLASS_NAMES) or
            config.get("indicators") != indicators or
            config.get("weak_trend_threshold_atr") != WEAK_SLOPE_ATR or
            prep.get("version") != 7 or
            not isinstance(architecture, list) or len(architecture) < 2 or
            architecture[0] != LOOKBACK * len(FEATURE_NAMES) or
            architecture[-1] != 1 or
            any(not isinstance(n, int) or n < 1 for n in architecture) or
            len(prep.get("mean", [])) != len(FEATURE_NAMES) or
            len(prep.get("std", [])) != len(FEATURE_NAMES)):
        raise ValueError("Incompatible v0.7 trend model configuration")

    data = prepare_trend(csv)
    total = len(data.closes) - LOOKBACK + 1
    a, b = split_positions(total)
    if candles > b - a:
        raise ValueError(f"Requested {candles} candles, but validation has only {b-a}")
    # Sample index 0 ends at LOOKBACK-1. Stop at b: this excludes holdout.
    start, stop = b - candles, b
    ends = np.arange(start + LOOKBACK - 1, stop + LOOKBACK - 1)

    from numpy.lib.stride_tricks import sliding_window_view
    features = sliding_window_view(data.features, LOOKBACK, axis=0)
    batch = features[start:stop].transpose(0, 2, 1)
    mu = np.asarray(prep["mean"], dtype=float)
    std = np.asarray(prep["std"], dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid training scaler")
    model = MLP(tuple(architecture))
    with np.load(root / "model.npz", allow_pickle=False) as saved:
        model.load_state_dict(saved)
    scores = np.asarray(model.forward(transform(batch, mu, std))).reshape(-1)
    detected = (scores >= 0.5).astype(np.int64)
    reference, strengths = price_trend_reference(data, ends)

    # Map the exact selected timestamps back to original untrimmed OHLC rows.
    raw = read_candles(csv)
    loc = pd.Index(raw["time"]).get_indexer(data.timestamps[ends])
    if np.any(loc < 0):
        raise ValueError("Validation candle timestamps did not match original CSV")
    window = raw.iloc[loc].copy()
    full_close = raw["close"].astype(float)
    bb_middle = full_close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    bb_sigma = full_close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    ema = full_close.ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean()
    # Exact same indicator definitions as current-trend training.
    return {
        "timestamps": data.timestamps[ends].copy(),
        "open": window["open"].to_numpy(dtype=float),
        "high": window["high"].to_numpy(dtype=float),
        "low": window["low"].to_numpy(dtype=float),
        "close": window["close"].to_numpy(dtype=float),
        "bb_middle": bb_middle.iloc[loc].to_numpy(dtype=float),
        "bb_upper": (bb_middle + BB_STD * bb_sigma).iloc[loc].to_numpy(dtype=float),
        "bb_lower": (bb_middle - BB_STD * bb_sigma).iloc[loc].to_numpy(dtype=float),
        "ema10": ema.iloc[loc].to_numpy(dtype=float),
        "detected": detected,
        "reference": reference.astype(np.int64),
        "scores_uncalibrated": scores,
        "weak": np.abs(strengths) < WEAK_SLOPE_ATR,
        "validation_sample_range": [int(start), int(stop)],
        "reserved_holdout_start_sample": int(b),
        "train_end_sample": int(a),
    }


def plot_validation_trend(csv: str, model_dir: str, out: str,
                          candles: int = 500) -> dict:
    """Render price candles, trend bands, and the historical price reference."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless Linux/servers and regular desktops
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch, Rectangle
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib: pip install -e '.[plot]'") from exc

    chart = validation_chart_data(csv, model_dir, candles)
    x = np.arange(candles)
    is_up = chart["close"] >= chart["open"]
    green, red = "#159a6a", "#cf4048"
    fig = plt.figure(figsize=(18, 8.2), facecolor="#ffffff")
    gs = fig.add_gridspec(2, 1, height_ratios=[7, 0.65], hspace=0.06,
                          left=0.065, right=0.985, top=0.88, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    bar = fig.add_subplot(gs[1], sharex=ax)
    ax.set_facecolor("#f9fafb")

    detected = chart["detected"]
    changes = np.r_[0, np.flatnonzero(np.diff(detected)) + 1, candles]
    for l, r in zip(changes[:-1], changes[1:]):
        ax.axvspan(l-0.5, r-0.5, color=green if detected[l] else red,
                   alpha=0.10, lw=0, zorder=0)

    # Price candlesticks, with constant width even across session/weekend gaps.
    for i in x:
        color = green if is_up[i] else red
        ax.vlines(i, chart["low"][i], chart["high"][i],
                  color=color, lw=0.8, zorder=2)
        o, c = chart["open"][i], chart["close"][i]
        base = min(o, c)
        height = max(abs(o-c), 1e-6)
        ax.add_patch(Rectangle((i-0.34, base), 0.68, height,
                               color=color, lw=0, zorder=3))

    ax.plot(x, chart["bb_upper"], color="#8895a7", lw=0.8, alpha=0.6, ls="--")
    ax.plot(x, chart["bb_lower"], color="#8895a7", lw=0.8, alpha=0.6, ls="--")
    ax.plot(x, chart["bb_middle"], color="#536d8b", lw=1.4, alpha=0.85)
    ax.plot(x, chart["ema10"], color="#ce8710", lw=1.25, alpha=0.92)
    ax.grid(axis="y", alpha=0.18)
    ax.set_ylabel("XAUUSD price", fontsize=11)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlim(-0.5, candles - 0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Two independent ribbons: model classification vs constructed OLS reference.
    rows = np.stack([detected, chart["reference"]])
    bar.imshow(rows, cmap=ListedColormap([red, green]), vmin=0, vmax=1,
               origin="upper", aspect="auto", interpolation="nearest",
               extent=(-0.5, candles-0.5, 1.5, -0.5))
    bar.set_yticks([0, 1], ["AI v0.7", "OLS32 proxy"])
    bar.set_ylim(1.5, -0.5)
    bar.tick_params(axis="y", length=0, pad=9, labelsize=10)
    bar.spines[:].set_visible(False)
    tick_positions = np.unique(np.round(np.linspace(0, candles-1, min(9, candles))).astype(int))
    stamp_labels = pd.to_datetime(chart["timestamps"][tick_positions], unit="s", utc=True)
    bar.set_xticks(tick_positions, [t.strftime("%d %b\n%H:%M") for t in stamp_labels],
                   fontsize=9)
    bar.set_xlabel("UTC dates · consecutive observed M15 candles (market closures omitted)",
                   fontsize=9, labelpad=11)

    agreement = float(np.mean(detected == chart["reference"]))
    weak_count = int(chart["weak"].sum())
    fig.suptitle("AI Finance v0.7  |  XAUUSD M15 current-trend recognition",
                 x=0.065, y=0.98, ha="left", fontsize=16, weight="bold")
    ax.set_title(f"Last {candles} VALIDATION candles  ·  AI / historical-slope agreement: {agreement:.1%}  ·  weak reference: {weak_count}",
                 loc="left", fontsize=10, color="#475569", pad=16)
    legend_items = [
        Patch(facecolor=green, alpha=0.45, label="Uptrend"),
        Patch(facecolor=red, alpha=0.45, label="Downtrend"),
        Line2D([0], [0], color="#536d8b", lw=1.5, label="Bollinger middle (SMA20)"),
        Line2D([0], [0], color="#ce8710", lw=1.5, label="EMA10"),
    ]
    ax.legend(handles=legend_items, ncol=4, loc="upper right", frameon=False, fontsize=9)
    fig.text(0.065, 0.023,
             "Past-only labels (OLS slope of last 32 bars). Validation was used to choose the model; "
             "this is not an independent accuracy test or future forecast.",
             ha="left", va="center", color="#64748b", fontsize=9)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)

    timestamps = pd.to_datetime(chart["timestamps"][[0, -1]], unit="s", utc=True)
    return {
        "model_version": MODEL_VERSION,
        "chart": str(output),
        "segment": "development_validation_only",
        "candles": candles,
        "first_candle_utc": timestamps[0].isoformat(),
        "last_candle_utc": timestamps[1].isoformat(),
        "ai_vs_historical_slope_agreement": agreement,
        "weak_reference_candles": weak_count,
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
        "note": "Visual agreement with a constructed past-only reference, not prediction performance.",
    }
