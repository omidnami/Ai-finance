"""Compare 1/4/8-candle targets on identical historical inputs.

Only training and validation slices are used to tune/select horizons. The
internal development holdout and external final_test.csv are not evaluated.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .experiment import (Settings, fit_scaler, fit_trend_scaler, metrics,
                         split_positions, transform_v2)
from .features import FEATURE_NAMES, LOOKBACK, PreparedData, labeled_windows_v2, prepare
from .network import LAYERS, MLP
from .trend import TREND_FEATURE_NAMES

DEFAULT_HORIZONS = (1, 4, 8)


def build_comparison_samples(data: PreparedData, horizons: tuple[int, ...]):
    """Return identical input windows/end times, and one target per horizon.

    A horizon h predicts close[t+h] > close[t], where t is the last observed
    closed candle. Using the maximum h makes all comparisons fair.
    """
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must not contain duplicates")
    max_h = max(horizons)
    if len(data.closes) < LOOKBACK + max_h + 150:
        raise ValueError("Too few candles for train, validation, and holdout with these horizons")
    all_x, all_extra, _, _ = labeled_windows_v2(data)
    count = len(data.closes) - LOOKBACK - max_h + 1
    x, extra = all_x[:count], all_extra[:count]
    ends = np.arange(LOOKBACK - 1, LOOKBACK - 1 + count)
    targets = {
        h: (data.closes[ends + h] > data.closes[ends]).astype(np.float64)
        for h in horizons
    }
    return x, extra, ends, targets


def comparison_slices(n: int, max_h: int):
    """Fixed slices for all horizons; purge max_h bars before boundaries.

    Training targets must precede the first validation input timestamp;
    validation targets must precede the internal holdout input timestamp.
    """
    a, b = split_positions(n)
    if a - max_h < 50 or b - a - max_h < 20:
        raise ValueError("Not enough samples after purging horizon boundaries")
    return slice(0, a - max_h), slice(a, b - max_h), slice(b, n)


def compare_horizons(csv: str, out: str, settings: Settings | None = None,
                     horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> dict:
    """Run independent initializations for each target, choose on val loss only."""
    cfg = settings or Settings()
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or not cfg.seeds:
        raise ValueError("Invalid training settings")
    data = prepare(csv)
    x, extra, ends, targets = build_comparison_samples(data, horizons)
    train_sel, val_sel, hold_sel = comparison_slices(len(x), max(horizons))
    mu, sigma = fit_scaler(x[train_sel])
    trend_mu, trend_sigma = fit_trend_scaler(extra[train_sel])
    xt = transform_v2(x[train_sel], extra[train_sel], mu, sigma, trend_mu, trend_sigma)
    xv = transform_v2(x[val_sel], extra[val_sel], mu, sigma, trend_mu, trend_sigma)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for horizon in horizons:
        yt = targets[horizon][train_sel]
        yv = targets[horizon][val_sel]
        best_val, best_seed, best_epoch, best_state = float("inf"), None, None, None
        trials = []
        for seed in cfg.seeds:
            model = MLP(seed=seed)
            rng = np.random.default_rng(seed)
            trial_val, trial_state, trial_epoch = float("inf"), None, None
            misses = 0
            for epoch in range(1, cfg.epochs + 1):
                order = rng.permutation(len(xt))
                for batch in np.array_split(order, range(cfg.batch_size, len(order), cfg.batch_size)):
                    model.train_batch(xt[batch], yt[batch], cfg.lr, cfg.weight_decay)
                val_loss = metrics(model, xv, yv)["loss"]
                if val_loss < trial_val - 1e-6:
                    trial_val, trial_epoch = val_loss, epoch
                    trial_state = model.state_dict()
                    misses = 0
                else:
                    misses += 1
                if misses >= cfg.patience:
                    break
            trials.append({"seed": seed, "best_epoch": trial_epoch,
                           "validation_loss": float(trial_val)})
            if trial_val < best_val:
                best_val, best_seed, best_epoch, best_state = (
                    trial_val, seed, trial_epoch, trial_state)
        selected = MLP(seed=best_seed)
        selected.load_state_dict(best_state)
        subdir = root / f"h{horizon}"
        subdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(subdir / "model.npz", **best_state)
        config = {
            "model_version": "0.3.0", "architecture": list(LAYERS),
            "lookback": LOOKBACK, "timeframe": 15,
            "features": list(FEATURE_NAMES), "trend_features": list(TREND_FEATURE_NAMES),
            "horizon_candles": horizon,
            "target_definition": f"close[t+{horizon}] > close[t]",
            "training": asdict(cfg),
        }
        prep = {
            "version": 3, "mean": mu.tolist(), "std": sigma.tolist(),
            "trend_mean": trend_mu.tolist(), "trend_std": trend_sigma.tolist(),
            "indicator_settings": {"macd": [12, 26, 9], "rsi": 14,
                                   "sma": [20, 50], "atr": 14},
            "time_zone": "UTC; Unix seconds",
            "sessions": "Asia/Tokyo 09-18, Europe/London 08-17, America/New_York 08-17; weekdays only",
        }
        result = {
            "horizon_candles": horizon,
            "nominal_minutes": 15 * horizon,
            "selected_seed": best_seed,
            "selected_epoch": best_epoch,
            "trials": trials,
            "train": metrics(selected, xt, yt),
            "validation": metrics(selected, xv, yv),
            "checkpoint": str(subdir),
        }
        (subdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (subdir / "preprocessing.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
        (subdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summaries.append(result)
    winner = min(summaries, key=lambda d: d["validation"]["loss"])
    # No test labels or prices are read into any scoring function.
    report = {
        "model_version": "0.3.0", "selection_metric": "validation_loss",
        "selected_horizon": winner["horizon_candles"],
        "selected_model": winner["checkpoint"],
        "comparisons": summaries,
        "split_windows": {
            "train": len(ends[train_sel]), "validation": len(ends[val_sel]),
            "internal_holdout_reserved": len(ends[hold_sel]),
        },
        "split_target_safety": "8-candle purged boundaries prevent cross-split future labels",
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
    }
    (root / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
