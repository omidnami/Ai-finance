"""Chronological split, train-only scaling, multi-seed selection and held-out test."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .features import (FEATURE_NAMES, LOOKBACK, prepare, labeled_windows,
                       labeled_windows_v2, latest_window, latest_window_v2)
from .trend import TREND_FEATURE_NAMES
from .network import LAYERS, MLP


@dataclass
class Settings:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.001
    patience: int = 12
    seeds: tuple[int, ...] = (13, 42, 97)
    weight_decay: float = 0.0001


def metrics(model: MLP, x: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(model.forward(x), 1e-12, 1 - 1e-12)
    pred = p >= 0.5
    truth = y.astype(bool)
    return {
        "loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        "accuracy": float(np.mean(pred == truth)),
        "error_rate": float(np.mean(pred != truth)),
        "up_accuracy": float(np.mean(pred[truth])) if truth.any() else None,
        "down_accuracy": float(np.mean(~pred[~truth])) if (~truth).any() else None,
        "samples": int(len(y)),
        "up_fraction": float(y.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1-y.mean())),
    }


def split_positions(n: int):
    a, b = int(n * .70), int(n * .85)
    if a < 50 or b-a < 20 or n-b < 20:
        raise ValueError("Dataset too small for train/validation/test splits")
    return a, b


def fit_scaler(x_train: np.ndarray):
    # All feature values used to fit statistics originate in the training period.
    mu = x_train.mean(axis=(0, 1))
    sigma = x_train.std(axis=(0, 1))
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def transform(x, mu, sigma):
    return np.clip((x - mu) / sigma, -10, 10).reshape(len(x), -1)


def fit_trend_scaler(extra_train: np.ndarray):
    """Fit the five new summary features on the train split ONLY."""
    mu = extra_train.mean(axis=0)
    sigma = extra_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def transform_v2(x, extra, mu, sigma, trend_mu, trend_sigma):
    """Preserve the v0.1 scaling of the 24 base features, append 5 new ones."""
    base = transform(x, mu, sigma)
    trend = np.clip((extra - trend_mu) / trend_sigma, -10, 10)
    return np.concatenate([base, trend], axis=1)


def train(excel: str, out: str, settings: Settings | None = None) -> dict:
    cfg = settings or Settings()
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or not cfg.seeds:
        raise ValueError("Invalid training settings")
    dataset = prepare(excel)
    x, extra, y, times = labeled_windows_v2(dataset)
    a, b = split_positions(len(x))
    mu, sigma = fit_scaler(x[:a])
    trend_mu, trend_sigma = fit_trend_scaler(extra[:a])
    xt, xv, xs = (
        transform_v2(x[start:end], extra[start:end], mu, sigma, trend_mu, trend_sigma)
        for start, end in ((0, a), (a, b), (b, len(x)))
    )
    yt, yv, ys = y[:a], y[a:b], y[b:]
    best_state, best_val, best_seed, best_epoch = None, float("inf"), None, None
    trials = []
    for seed in cfg.seeds:
        model = MLP(seed=seed)
        rng = np.random.default_rng(seed)
        trial_loss, trial_state, trial_epoch = float("inf"), None, None
        misses = 0
        for epoch in range(1, cfg.epochs + 1):
            order = rng.permutation(len(xt))
            for batch in np.array_split(order, range(cfg.batch_size, len(order), cfg.batch_size)):
                model.train_batch(xt[batch], yt[batch], cfg.lr, cfg.weight_decay)
            val_loss = metrics(model, xv, yv)["loss"]
            if val_loss < trial_loss - 1e-6:
                trial_loss, trial_state, trial_epoch = val_loss, model.state_dict(), epoch
                misses = 0
            else:
                misses += 1
            if misses >= cfg.patience:
                break
        trials.append({"seed": seed, "best_epoch": trial_epoch, "validation_loss": trial_loss})
        if trial_loss < best_val:
            best_state, best_val, best_seed, best_epoch = trial_state, trial_loss, seed, trial_epoch
    chosen = MLP(seed=best_seed)
    chosen.load_state_dict(best_state)
    results = {
        "model_version": "0.2.0",
        "input_features": int(LAYERS[0]),
        "selected_seed": best_seed,
        "selected_epoch": best_epoch,
        "trials": trials,
        "splits": {
            "train": [int(times[0]), int(times[a-1])],
            "validation": [int(times[a]), int(times[b-1])],
            "test": [int(times[b]), int(times[-1])],
        },
        "train": metrics(chosen, xt, yt),
        "validation": metrics(chosen, xv, yv),
        "test": metrics(chosen, xs, ys),
    }
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "model.npz", **best_state)
    (target / "config.json").write_text(json.dumps({"architecture": list(LAYERS), "lookback": LOOKBACK,
                                                       "timeframe": 15, "features": list(FEATURE_NAMES),
                                                       "trend_features": list(TREND_FEATURE_NAMES),
                                                       "model_version": "0.2.0",
                                                       "training": asdict(cfg)}, indent=2), encoding="utf-8")
    (target / "preprocessing.json").write_text(json.dumps({"mean": mu.tolist(), "std": sigma.tolist(),
        "trend_mean": trend_mu.tolist(), "trend_std": trend_sigma.tolist(),
        "indicator_settings": {"macd": [12, 26, 9], "rsi": 14, "sma": [20, 50], "atr": 14},
        "trend_definitions": {
            "trend_5": "close[t]/close[t-5] - 1",
            "trend_10": "close[t]/close[t-10] - 1",
            "trend_20": "close[t]/close[t-19] - 1 (20-bar window)",
            "trend_strength": "abs(close[t]-close[t-19])/sum(abs(delta(close[t-19:t])))",
            "atr14_ratio": "Wilder ATR14 / close[t]"},
        "sessions": "Asia/Tokyo 09-18, Europe/London 08-17, America/New_York 08-17; weekdays only",
        "time_zone": "UTC; Unix seconds", "version": 2}, indent=2), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def predict(excel: str, model_dir: str) -> float:
    """Use saved schemas; remain compatible with v0.1, v0.2, v0.3 and v0.4."""
    from .architectures import ARCHITECTURES

    root = Path(model_dir)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    common_ok = (cfg.get("features") == list(FEATURE_NAMES) and cfg.get("lookback") == LOOKBACK
                 and cfg.get("timeframe") == 15)
    v4 = (prep.get("version") == 4 and cfg.get("model_version") == "0.4.0"
          and cfg.get("horizon_candles") == 1
          and cfg.get("architecture_name") in ARCHITECTURES
          and cfg.get("architecture") == list(ARCHITECTURES.get(cfg.get("architecture_name"), ()))
          and cfg.get("trend_features") == list(TREND_FEATURE_NAMES))
    v3 = (prep.get("version") == 3 and cfg.get("model_version") == "0.3.0"
          and cfg.get("horizon_candles") in (1, 4, 8)
          and cfg.get("architecture") == list(LAYERS)
          and cfg.get("trend_features") == list(TREND_FEATURE_NAMES))
    v2 = (prep.get("version") == 2 and cfg.get("architecture") == list(LAYERS)
          and cfg.get("trend_features") == list(TREND_FEATURE_NAMES))
    v1_layers = (480, 128, 64, 32, 1)
    v1 = (prep.get("version") == 1 and cfg.get("architecture") == list(v1_layers)
          and "trend_features" not in cfg)
    if not common_ok or not (v1 or v2 or v3 or v4):
        raise ValueError("Incompatible model configuration")
    sizes = tuple(cfg["architecture"])
    model = MLP(sizes=sizes)
    with np.load(root / "model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    data = prepare(excel)
    mean = np.asarray(prep["mean"], dtype=np.float64)
    std = np.asarray(prep["std"], dtype=np.float64)
    if v2 or v3 or v4:
        x, extra = latest_window_v2(data)
        x = transform_v2(x[None, ...], extra[None, ...], mean, std,
                         np.asarray(prep["trend_mean"], dtype=np.float64),
                         np.asarray(prep["trend_std"], dtype=np.float64))
    else:
        x = transform(latest_window(data)[None, ...], mean, std)
    return float(model.forward(x)[0])
