"""Plot causal v0.8 reversal confirmations exclusively within development validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import transform
from .features import read_candles
from .reversal_detection import (
    VERSION, LOOKBACK, FEATURES, CLASSES, BalancedSoftmax,
    _selections, prepare_reversal, reversal_samples,
)


def reversal_validation_data(csv: str, model_dir: str, candles: int = 500) -> dict:
    if not isinstance(candles, int) or candles < 1:
        raise ValueError("--candles must be positive")
    root = Path(model_dir)
    cfg = json.loads((root/"config.json").read_text(encoding="utf-8"))
    pre = json.loads((root/"preprocessing.json").read_text(encoding="utf-8"))
    sizes = cfg.get("architecture")
    if (cfg.get("model_version") != VERSION or pre.get("version") != 8 or
            cfg.get("features") != list(FEATURES) or cfg.get("classes") != list(CLASSES) or
            cfg.get("lookback") != LOOKBACK or cfg.get("timeframe") != 15 or
            cfg.get("task") != "past_only_confirmed_swing_reversal_reference_not_forecasting" or
            not isinstance(sizes, list) or len(sizes) < 2 or
            sizes[0] != LOOKBACK*len(FEATURES) or sizes[-1] != 3 or
            len(pre.get("mean", [])) != len(FEATURES) or
            len(pre.get("std", [])) != len(FEATURES)):
        raise ValueError("Incompatible v0.8 reversal checkpoint")
    data = prepare_reversal(csv)
    x, y, strength, ends = reversal_samples(data)
    train_sel, val_sel, hold_sel = _selections(len(y))
    if candles > val_sel.stop-val_sel.start:
        raise ValueError("Requested candles exceed available validation candles")
    start, stop = val_sel.stop-candles, val_sel.stop
    mean = np.asarray(pre["mean"], dtype=float)
    std = np.asarray(pre["std"], dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid v0.8 scaler")
    model = BalancedSoftmax(tuple(sizes), np.ones(3))
    with np.load(root/"model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    probs = model.forward(transform(x[start:stop], mean, std))
    predicted = probs.argmax(axis=1)
    raw = read_candles(csv)
    idx = pd.Index(raw["time"]).get_indexer(data.timestamps[ends[start:stop]])
    if (idx < 0).any():
        raise ValueError("Missing mapped validation candle")
    prices = raw.iloc[idx]
    close = raw["close"].astype(float)
    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    return {
        "timestamps": data.timestamps[ends[start:stop]].copy(),
        "open": prices["open"].to_numpy(dtype=float),
        "high": prices["high"].to_numpy(dtype=float),
        "low": prices["low"].to_numpy(dtype=float),
        "close": prices["close"].to_numpy(dtype=float),
        "bb_upper": (mid+2*sd).iloc[idx].to_numpy(dtype=float),
        "bb_middle": mid.iloc[idx].to_numpy(dtype=float),
        "bb_lower": (mid-2*sd).iloc[idx].to_numpy(dtype=float),
        "rsi14": data.features[ends[start:stop], FEATURES.index("rsi14")]*100,
        "reference": y[start:stop].copy(),
        "predicted": predicted,
        "scores_uncalibrated": probs,
        "pre_move_strength_atr": strength[start:stop],
        "validation_sample_range": [int(start), int(stop)],
        "holdout_start_sample": int(hold_sel.start),
        "validation_end_sample": int(val_sel.stop),
        "train_end_sample": int(train_sel.stop),
    }


def plot_validation_reversals(csv: str, model_dir: str, out: str, candles: int = 500) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("Install plotting support: pip install -e '.[plot]'") from exc

    chart = reversal_validation_data(csv, model_dir, candles)
    x = np.arange(candles)
    bull, bear = "#059669", "#dc2626"
    fig = plt.figure(figsize=(18, 8.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.1], hspace=0.05,
                          left=.065, right=.985, top=.87, bottom=.13)
    ax = fig.add_subplot(gs[0]); lower = fig.add_subplot(gs[1], sharex=ax)
    ax.set_facecolor("#fafafa")
    for i in x:
        color = bull if chart["close"][i] >= chart["open"][i] else bear
        ax.vlines(i, chart["low"][i], chart["high"][i], color=color, lw=.65)
        low = min(chart["open"][i], chart["close"][i])
        ax.add_patch(Rectangle((i-.34, low), .68,
                              max(abs(chart["close"][i]-chart["open"][i]), 1e-6),
                              color=color, linewidth=0))
    ax.plot(x, chart["bb_middle"], color="#64748b", lw=1.0, alpha=.8)
    ax.plot(x, chart["bb_upper"], color="#94a3b8", lw=.85, ls="--", alpha=.7)
    ax.plot(x, chart["bb_lower"], color="#94a3b8", lw=.85, ls="--", alpha=.7)
    # IMPORTANT: draw markers at confirmation candle t (not pivot t-1).
    for cls, marker, edge, value in ((1, "^", bull, "low"), (2, "v", bear, "high")):
        actual_idx = np.flatnonzero(chart["reference"] == cls)
        pred_idx = np.flatnonzero(chart["predicted"] == cls)
        offset = .55*(1 if cls == 1 else -1)
        ax.scatter(actual_idx, chart[value][actual_idx]+offset,
                   s=55, marker="o", facecolors="none", edgecolors=edge,
                   linewidths=1.4, zorder=5)
        ax.scatter(pred_idx, chart[value][pred_idx]+offset,
                   s=40, marker=marker, c=edge, zorder=6)
    ax.grid(axis="y", alpha=.15)
    ax.set_xlim(-.5, candles-.5)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("XAUUSD price")
    lower.plot(x, chart["rsi14"], lw=1.1, color="#4338ca", label="RSI14")
    for level in (20, 60, 70):
        lower.axhline(level, color="#a1a1aa", lw=.8, ls="--")
        lower.text(candles-1, level+1.5, str(level), ha="right", fontsize=8, color="#64748b")
    lower.set_ylim(0, 100)
    lower.set_ylabel("RSI14")
    lower.grid(axis="y", alpha=.16)
    ticks = np.unique(np.round(np.linspace(0, candles-1, min(9, candles))).astype(int))
    times = pd.to_datetime(chart["timestamps"][ticks], unit="s", utc=True)
    lower.set_xticks(ticks, [t.strftime("%d %b\n%H:%M") for t in times], fontsize=9)
    lower.set_xlabel("UTC · consecutive completed M15 candles (market gaps preserved)")
    for pane in (ax, lower):
        pane.spines[["top", "right"]].set_visible(False)
    fig.suptitle("AI Finance v0.8  |  XAUUSD M15 confirmed reversal candidates",
                 x=.065, y=.98, ha="left", fontsize=16, weight="bold")
    bull_pred, bear_pred = int(np.sum(chart["predicted"] == 1)), int(np.sum(chart["predicted"] == 2))
    bull_ref, bear_ref = int(np.sum(chart["reference"] == 1)), int(np.sum(chart["reference"] == 2))
    ax.set_title(f"Last {candles} VALIDATION candles · Model: {bull_pred} bullish / {bear_pred} bearish · Price reference: {bull_ref} bullish / {bear_ref} bearish",
                 loc="left", fontsize=10, pad=17, color="#475569")
    ax.legend(handles=[Line2D([0], [0], color=bull, marker="^", linestyle="None", label="AI bullish"),
                       Line2D([0], [0], color=bear, marker="v", linestyle="None", label="AI bearish"),
                       Line2D([0], [0], marker="o", color="#64748b", markerfacecolor="none",
                              linestyle="None", label="Price-only reference (outline)"),
                       Line2D([0], [0], color="#64748b", lw=1, label="BB20 middle")],
              ncol=4, loc="upper right", fontsize=9, frameon=False)
    fig.text(.065, .035,
             "Markers are at the CLOSED CONFIRMATION candle, not at an earlier actionable pivot. "
             "Only validation is shown; rare-event proxy labels, not proven turning points or future forecasts.",
             fontsize=8.8, ha="left", color="#64748b")
    output = Path(out); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return {
        "model_version": VERSION,
        "chart": str(output),
        "candles": candles,
        "validation_only": True,
        "reference_count": {"bullish_reversal": bull_ref, "bearish_reversal": bear_ref},
        "predicted_count": {"bullish_reversal": bull_pred, "bearish_reversal": bear_pred},
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
    }
