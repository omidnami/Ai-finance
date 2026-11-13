"""Finance AI v0.5: three-way next-candle classification using a causal ATR threshold.

Train and validate chronologically; neither the development holdout nor the
separately reserved final_test.csv is evaluated or used for model selection.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .experiment import Settings, fit_scaler, fit_trend_scaler, transform_v2
from .features import (FEATURE_NAMES, FEATURE_NAMES_NO_MACD, LOOKBACK, PreparedData, labeled_windows_v2,
                       latest_window_v2, prepare)
from .horizons import comparison_slices
from .network import MLP
from .trend import TREND_FEATURE_NAMES

CLASS_NAMES = ("down", "neutral", "up")
DEFAULT_HIDDEN = (512, 256, 128, 64)
DEFAULT_ATR_MULTIPLIER = 0.25
COMPARISON_PURGE = 8  # Keep v0.3/v0.4 train/val boundaries and windows.


class SoftmaxMLP(MLP):
    """Reuse the existing custom AdamW/backprop engine with 3-way softmax BCE replacement."""

    def __init__(self, sizes: tuple[int, ...], seed: int = 42):
        if len(sizes) < 2 or sizes[-1] != 3 or any(n < 1 for n in sizes):
            raise ValueError("SoftmaxMLP requires positive layers ending in three outputs")
        super().__init__(sizes=sizes, seed=seed)

    def forward(self, x, cache=False):
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.maximum(activations[-1] @ w + b, 0))
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        centered = z - z.max(axis=1, keepdims=True)
        exp = np.exp(centered)
        p = exp / exp.sum(axis=1, keepdims=True)
        return (z, p, activations) if cache else p

    def gradients(self, x, y):
        z, p, acts = self.forward(x, cache=True)
        y = np.asarray(y, dtype=np.int64)
        if len(x) == 0 or y.shape != (len(x),) or np.any((y < 0) | (y > 2)):
            raise ValueError("Expected one class label (0, 1, or 2) per example")
        # logsumexp is stable for arbitrary logits; do not clip probabilities.
        centered = z - z.max(axis=1, keepdims=True)
        loss = np.mean(np.log(np.exp(centered).sum(axis=1)) - centered[np.arange(len(y)), y])
        delta = p.copy()
        delta[np.arange(len(y)), y] -= 1.0
        delta /= len(y)
        gw, gb = [None] * len(self.weights), [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            gw[i] = acts[i].T @ delta
            gb[i] = np.sum(delta, axis=0, keepdims=True)
            if i:
                delta = (delta @ self.weights[i].T) * (acts[i] > 0)
        return float(loss), gw, gb


def atr_labels(data: PreparedData, ends: np.ndarray, multiplier: float) -> np.ndarray:
    """Classify close[t+1]-close[t] against +/- multiplier * ATR14[t].

    The label can use the *future close*, but the threshold only uses ATR14
    known at t. Class IDs: 0 down, 1 neutral, 2 up.
    """
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("ATR multiplier must be positive and finite")
    if data.atr14_ratio is None:
        raise ValueError("ATR14 is required")
    ends = np.asarray(ends, dtype=np.int64)
    if (ends < 0).any() or (ends + 1 >= len(data.closes)).any():
        raise ValueError("Sample end requires an available next candle")
    change = data.closes[ends + 1] - data.closes[ends]
    atr_at_end = data.atr14_ratio[ends] * data.closes[ends]
    if not np.isfinite(atr_at_end).all() or (atr_at_end <= 0).any():
        raise ValueError("ATR14 must be positive and finite at every sample end")
    threshold = multiplier * atr_at_end
    return np.where(change > threshold, 2, np.where(change < -threshold, 0, 1)).astype(np.int64)


def classification_metrics(model: SoftmaxMLP, x: np.ndarray, y: np.ndarray) -> dict:
    p = model.forward(x)
    predictions = p.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (y, predictions), 1)  # rows=true, columns=predicted
    recall, precision, f1 = {}, {}, {}
    for i, label in enumerate(CLASS_NAMES):
        tp = confusion[i, i]
        actual, predicted = confusion[i].sum(), confusion[:, i].sum()
        r = float(tp / actual) if actual else 0.0
        pr = float(tp / predicted) if predicted else 0.0
        recall[label] = r
        precision[label] = pr
        f1[label] = float(2 * pr * r / (pr + r)) if pr + r else 0.0
    fractions = np.bincount(y, minlength=3) / len(y)
    predicted_fraction = np.bincount(predictions, minlength=3) / len(y)
    return {
        "loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(list(f1.values()))),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion.tolist(),
        "samples": int(len(y)),
        "class_fraction": dict(zip(CLASS_NAMES, map(float, fractions))),
        "predicted_fraction": dict(zip(CLASS_NAMES, map(float, predicted_fraction))),
        "majority_baseline_accuracy": float(fractions.max()),
    }


def train_atr(csv: str, out: str, settings: Settings | None = None,
              hidden: tuple[int, ...] = DEFAULT_HIDDEN,
              atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
              *, include_macd: bool = False) -> dict:
    cfg = settings or Settings()
    if (cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patience < 1 or not cfg.seeds or
            any(not isinstance(seed, int) for seed in cfg.seeds)):
        raise ValueError("Invalid training settings")
    if not hidden or any(not isinstance(n, int) or n < 1 for n in hidden):
        raise ValueError("Hidden layer sizes must be positive integers")
    if not np.isfinite(atr_multiplier) or atr_multiplier <= 0:
        raise ValueError("ATR multiplier must be positive and finite")

    selected_features = FEATURE_NAMES if include_macd else FEATURE_NAMES_NO_MACD
    model_version = "0.5.0" if include_macd else "0.6.0"
    preprocessing_version = 5 if include_macd else 6
    sizes = (LOOKBACK * len(selected_features) + len(TREND_FEATURE_NAMES), *hidden, 3)
    data = prepare(csv) if include_macd else prepare(csv, include_macd=False)
    # Match the v0.4 sample endpoints exactly. Do not use the reserved holdout
    # labels or prices for either training or validation scoring.
    all_x, all_extra, _, _ = labeled_windows_v2(data)
    count = len(data.closes) - LOOKBACK - COMPARISON_PURGE + 1
    x, extra = all_x[:count], all_extra[:count]
    ends = np.arange(LOOKBACK - 1, LOOKBACK - 1 + count)
    train_sel, val_sel, hold_sel = comparison_slices(len(ends), COMPARISON_PURGE)
    mu, sigma = fit_scaler(x[train_sel])
    trend_mu, trend_sigma = fit_trend_scaler(extra[train_sel])
    xt = transform_v2(x[train_sel], extra[train_sel], mu, sigma, trend_mu, trend_sigma)
    xv = transform_v2(x[val_sel], extra[val_sel], mu, sigma, trend_mu, trend_sigma)
    yt = atr_labels(data, ends[train_sel], atr_multiplier)
    yv = atr_labels(data, ends[val_sel], atr_multiplier)

    trials = []
    best_loss, best_state, best_seed, best_epoch = float("inf"), None, None, None
    for seed in cfg.seeds:
        model = SoftmaxMLP(sizes, seed=seed)
        rng = np.random.default_rng(seed)
        trial_loss, trial_state, trial_epoch = float("inf"), None, None
        misses = 0
        for epoch in range(1, cfg.epochs + 1):
            order = rng.permutation(len(xt))
            for batch in np.array_split(order, range(cfg.batch_size, len(order), cfg.batch_size)):
                model.train_batch(xt[batch], yt[batch], cfg.lr, cfg.weight_decay)
            val_loss = classification_metrics(model, xv, yv)["loss"]
            if val_loss < trial_loss - 1e-6:
                trial_loss, trial_state, trial_epoch = val_loss, model.state_dict(), epoch
                misses = 0
            else:
                misses += 1
            if misses >= cfg.patience:
                break
        trials.append({"seed": seed, "best_epoch": trial_epoch,
                       "validation_loss": float(trial_loss)})
        if trial_loss < best_loss:
            best_loss, best_state, best_seed, best_epoch = (
                trial_loss, trial_state, seed, trial_epoch)

    chosen = SoftmaxMLP(sizes, seed=best_seed)
    chosen.load_state_dict(best_state)
    report = {
        "model_version": model_version, "prediction_type": "three_class_atr",
        "architecture": list(sizes),
        "parameters": int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))),
        "horizon_candles": 1,
        "atr_multiplier": float(atr_multiplier),
        "selected_seed": best_seed, "selected_epoch": best_epoch,
        "trials": trials,
        "train": classification_metrics(chosen, xt, yt),
        "validation": classification_metrics(chosen, xv, yv),
        "split_windows": {"train": len(ends[train_sel]), "validation": len(ends[val_sel]),
                          "internal_holdout_reserved": len(ends[hold_sel])},
        "internal_holdout_evaluated": False,
        "external_final_test_evaluated": False,
    }
    # Baseline log loss from train class prevalences is a fair validation comparator.
    train_prior = np.bincount(yt, minlength=3).astype(float) / len(yt)
    report["validation"]["train_prior_baseline_loss"] = float(
        -np.log(np.maximum(train_prior[yv], 1e-15)).mean())

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "model.npz", **best_state)
    config = {
        "model_version": model_version, "architecture": list(sizes),
        "lookback": LOOKBACK, "timeframe": 15,
        "features": list(selected_features), "trend_features": list(TREND_FEATURE_NAMES),
        "horizon_candles": 1, "classes": list(CLASS_NAMES),
        "atr_multiplier": float(atr_multiplier),
        "target_definition": "down if close[t+1]-close[t] < -k*ATR14[t]; up if > k*ATR14[t]; else neutral",
        "training": asdict(cfg),
    }
    preprocessing = {
        "version": preprocessing_version, "mean": mu.tolist(), "std": sigma.tolist(),
        "trend_mean": trend_mu.tolist(), "trend_std": trend_sigma.tolist(),
        "indicator_settings": ({"macd": [12, 26, 9], "rsi": 14,
                                "sma": [20, 50], "atr": 14} if include_macd
                               else {"rsi": 14, "sma": [20, 50], "atr": 14}),
        "time_zone": "UTC; Unix seconds",
        "sessions": "Asia/Tokyo 09-18, Europe/London 08-17, America/New_York 08-17; weekdays only",
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (directory / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    (directory / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_atr(csv: str, model_dir: str) -> dict:
    root = Path(model_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    prep = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    sizes = config.get("architecture")
    version = config.get("model_version")
    legacy = version == "0.5.0" and prep.get("version") == 5
    no_macd = version == "0.6.0" and prep.get("version") == 6
    expected_features = FEATURE_NAMES if legacy else FEATURE_NAMES_NO_MACD
    expected_inputs = LOOKBACK * len(expected_features) + len(TREND_FEATURE_NAMES)
    if (not (legacy or no_macd) or
            not isinstance(sizes, list) or len(sizes) < 3 or
            sizes[0] != expected_inputs or sizes[-1] != 3 or
            any(not isinstance(n, int) or n < 1 for n in sizes) or
            config.get("features") != list(expected_features) or
            config.get("trend_features") != list(TREND_FEATURE_NAMES) or
            config.get("classes") != list(CLASS_NAMES) or
            config.get("lookback") != LOOKBACK or config.get("timeframe") != 15 or
            config.get("horizon_candles") != 1):
        raise ValueError("Incompatible ATR model configuration")
    model = SoftmaxMLP(tuple(sizes))
    with np.load(root / "model.npz", allow_pickle=False) as checkpoint:
        model.load_state_dict(checkpoint)
    data = prepare(csv) if legacy else prepare(csv, include_macd=False)
    x, extra = latest_window_v2(data)
    scaled = transform_v2(x[None], extra[None],
                          np.asarray(prep["mean"], dtype=float),
                          np.asarray(prep["std"], dtype=float),
                          np.asarray(prep["trend_mean"], dtype=float),
                          np.asarray(prep["trend_std"], dtype=float))
    p = model.forward(scaled)[0]
    return {"model_version": version, "horizon_candles": 1,
            "predicted_direction": CLASS_NAMES[int(np.argmax(p))],
            "probabilities_uncalibrated": dict(zip(CLASS_NAMES, map(float, p))),
            "atr_multiplier": float(config["atr_multiplier"])}
