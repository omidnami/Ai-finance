"""Finance AI v0.4: controlled CPU architecture-size comparison.

Only training and validation data participate in selection. The internal
holdout and separately reserved final_test.csv are never evaluated here.
The one-bar target, input features, and chronological boundaries are fixed.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .experiment import (Settings, fit_scaler, fit_trend_scaler, metrics,
                         transform_v2)
from .features import FEATURE_NAMES, LOOKBACK, prepare
from .horizons import build_comparison_samples, comparison_slices
from .network import LAYERS, MLP
from .trend import TREND_FEATURE_NAMES

# Keep the original v0.2/v0.3 architecture as a control, plus three smaller ones.
ARCHITECTURES = {
    "large": LAYERS,
    "medium": (485, 64, 32, 1),
    "small": (485, 32, 16, 1),
    "linear": (485, 1),
}
HORIZON = 1
COMPARISON_MAX_HORIZON = 8  # Same samples/purged split as v0.3 for comparability.


def _validate(settings: Settings):
    if (settings.epochs < 1 or settings.batch_size < 1 or settings.patience < 1 or
            not settings.seeds or any(not isinstance(s, int) for s in settings.seeds)):
        raise ValueError("Invalid training settings")


def compare_architectures(csv: str, out: str, settings: Settings | None = None) -> dict:
    """Train multiple sizes on exactly the same windows; choose via validation loss."""
    cfg = settings or Settings()
    _validate(cfg)
    data = prepare(csv)
    # Use the same eight-candle horizon purge as v0.3 to hold input samples
    # and the one-candle labels constant for the large-network control.
    x, extra, ends, labels = build_comparison_samples(data, (1, 4, 8))
    train_sel, val_sel, hold_sel = comparison_slices(len(x), COMPARISON_MAX_HORIZON)
    mu, sigma = fit_scaler(x[train_sel])
    trend_mu, trend_sigma = fit_trend_scaler(extra[train_sel])
    xt = transform_v2(x[train_sel], extra[train_sel], mu, sigma, trend_mu, trend_sigma)
    xv = transform_v2(x[val_sel], extra[val_sel], mu, sigma, trend_mu, trend_sigma)
    yt = labels[HORIZON][train_sel]
    yv = labels[HORIZON][val_sel]
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name, sizes in ARCHITECTURES.items():
        best_loss, best_state, best_epoch, best_seed = float("inf"), None, None, None
        trials = []
        for seed in cfg.seeds:
            model = MLP(sizes=sizes, seed=seed)
            rng = np.random.default_rng(seed)
            trial_loss, trial_state, trial_epoch = float("inf"), None, None
            misses = 0
            for epoch in range(1, cfg.epochs + 1):
                order = rng.permutation(len(xt))
                for batch in np.array_split(order, range(cfg.batch_size, len(order), cfg.batch_size)):
                    model.train_batch(xt[batch], yt[batch], cfg.lr, cfg.weight_decay)
                val_loss = metrics(model, xv, yv)["loss"]
                if val_loss < trial_loss - 1e-6:
                    trial_loss, trial_epoch = val_loss, epoch
                    trial_state = model.state_dict()
                    misses = 0
                else:
                    misses += 1
                if misses >= cfg.patience:
                    break
            trials.append({"seed": seed, "best_epoch": trial_epoch,
                           "validation_loss": float(trial_loss)})
            if trial_loss < best_loss:
                best_loss, best_state, best_epoch, best_seed = (
                    trial_loss, trial_state, trial_epoch, seed)

        chosen = MLP(sizes=sizes, seed=best_seed)
        chosen.load_state_dict(best_state)
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "model.npz", **best_state)
        config = {
            "model_version": "0.4.0", "architecture": list(sizes),
            "architecture_name": name, "lookback": LOOKBACK, "timeframe": 15,
            "features": list(FEATURE_NAMES), "trend_features": list(TREND_FEATURE_NAMES),
            "horizon_candles": HORIZON,
            "target_definition": "close[t+1] > close[t]",
            "training": asdict(cfg),
        }
        prep = {
            "version": 4, "mean": mu.tolist(), "std": sigma.tolist(),
            "trend_mean": trend_mu.tolist(), "trend_std": trend_sigma.tolist(),
            "indicator_settings": {"macd": [12, 26, 9], "rsi": 14,
                                   "sma": [20, 50], "atr": 14},
            "time_zone": "UTC; Unix seconds",
            "sessions": "Asia/Tokyo 09-18, Europe/London 08-17, America/New_York 08-17; weekdays only",
        }
        result = {
            "architecture_name": name,
            "architecture": list(sizes),
            "parameters": int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))),
            "horizon_candles": HORIZON,
            "selected_seed": best_seed,
            "selected_epoch": best_epoch,
            "trials": trials,
            "train": metrics(chosen, xt, yt),
            "validation": metrics(chosen, xv, yv),
            "checkpoint": str(directory),
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (directory / "preprocessing.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
        (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summaries.append(result)

    winner = min(summaries, key=lambda r: r["validation"]["loss"])
    report = {
        "model_version": "0.4.0",
        "selection_metric": "validation_loss",
        "horizon_candles": HORIZON,
        "selected_architecture": winner["architecture_name"],
        "selected_model": winner["checkpoint"],
        "comparisons": summaries,
        "split_windows": {
            "train": len(ends[train_sel]),
            "validation": len(ends[val_sel]),
            "internal_holdout_reserved": len(ends[hold_sel]),
        },
        "split_target_safety": "8-candle purged boundaries identical to v0.3; h=1 labels",
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
    }
    (root / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
